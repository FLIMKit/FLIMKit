import numpy as np
from flimkit.GPU._base import _BackendMixin
from flimkit.FLIM.fit_tools import estimate_bg, coates_pileup_correction


class TorchBackend(_BackendMixin):
    """GPU backend using PyTorch — works on CUDA (NVIDIA), MPS (Apple), and ROCm (AMD)."""

    def __init__(self, device="cuda"):
        import torch
        self._torch = torch
        self.device = torch.device(device)

    def __repr__(self):
        return f"TorchBackend(device='{self.device}')"

    def batch_fixed_tau(
        self,
        stack,
        A,
        taus_fixed,
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback=None,
    ):
        """Fit all pixels at once using fixed lifetimes — GPU does one big matmul."""
        torch = self._torch
        ny, nx, n_bins = stack.shape
        n_exp = A.shape[1]
        taus_ns = taus_fixed * 1e9

        # Compute pinv on CPU — linalg_svd (used internally by pinv) is not
        # supported on MPS and would silently fall back, triggering a UserWarning.
        A_cpu   = torch.as_tensor(A, dtype=torch.float32, device="cpu")
        A_pinv  = torch.linalg.pinv(A_cpu).to(self.device)  # (n_exp, n_bins)

        flat    = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask = intensity_flat >= min_photons

        bg_flat = self._estimate_bg_batch(flat, valid_mask)

        data_corr = np.maximum(flat - bg_flat[:, None], 0.0)

        if correct_pileup and n_sync_px > 0:
            for idx in np.where(valid_mask)[0]:
                data_corr[idx] = coates_pileup_correction(data_corr[idx], n_sync_px)

        data_t   = torch.as_tensor(data_corr, dtype=torch.float32, device=self.device)
        amps_raw = data_t @ A_pinv.T              # (N_pix, n_exp)
        amps_t   = torch.clamp(amps_raw, min=0.0)  # enforce non-negative amplitudes

        amps_np = amps_t.cpu().numpy()            # (N_pix, n_exp)
        bg_np   = bg_flat                         # (N_pix,)

        valid_idx = np.where(valid_mask)[0]

        maps = self._init_maps(
            ny, nx, n_exp,
            intensity = stack.sum(axis=2),
            taus_fixed_ns = taus_ns,
            free_tau = False,
        )
        if valid_idx.size == 0:
            return maps

        self._scatter_fixed_tau(
            maps,
            valid_idx  = valid_idx,
            amps       = amps_np[valid_idx],
            bg         = bg_np[valid_idx],
            decay_valid= flat[valid_idx],
            A          = A,
            taus_ns    = taus_ns,
            ny = ny, nx = nx,
        )
        return maps

    def batch_grid_scan_1exp(
        self,
        stack,
        basis_grid,
        bb_grid,
        tau_grid,
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback=None,
    ):
        """Fit all pixels by finding the best τ from a log-spaced grid — n_exp=1 only."""
        torch = self._torch
        ny, nx, n_bins = stack.shape
        N_GRID = len(tau_grid)

        flat           = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask     = intensity_flat >= min_photons
        valid_idx      = np.where(valid_mask)[0]

        maps = self._init_maps(
            ny, nx, n_exp=1,
            intensity=stack.sum(axis=2),
            taus_fixed_ns=np.array([tau_grid[N_GRID // 2] * 1e9]),
            free_tau=True,
        )
        if valid_idx.size == 0:
            return maps

        bg_flat   = self._estimate_bg_batch(flat, valid_mask)
        dc_flat   = np.maximum(flat - bg_flat[:, None], 0.0)

        if correct_pileup and n_sync_px > 0:
            for idx in valid_idx:
                dc_flat[idx] = coates_pileup_correction(dc_flat[idx], n_sync_px)

        dc_valid  = dc_flat[valid_idx]              # (N_valid, n_bins)

        bg_t      = torch.as_tensor(bb_grid, dtype=torch.float32, device=self.device)
        basis_t   = torch.as_tensor(basis_grid, dtype=torch.float32, device=self.device)
        dc_t      = torch.as_tensor(dc_valid, dtype=torch.float32, device=self.device)

        bd        = dc_t @ basis_t.T               # (N_valid, N_GRID)
        dc_sq     = (dc_t ** 2).sum(dim=1)         # (N_valid,)
        # cost = ||d||^2 - max(d·b, 0)^2 / ||b||^2; minimise → best τ per pixel
        costs     = dc_sq[:, None] - torch.clamp(bd, min=0.0) ** 2 / bg_t[None, :]
        best_g    = costs.argmin(dim=1).cpu().numpy()          # (N_valid,)
        bd_np     = bd.cpu().numpy()
        bb_np     = bb_grid

        tau_v     = tau_grid[best_g]               # (N_valid,) seconds
        amp_v     = np.maximum(bd_np[np.arange(len(valid_idx)), best_g]
                               / bb_np[best_g], 0.0)
        basis_best = basis_grid[best_g]            # (N_valid, n_bins)

        self._scatter_1exp(
            maps,
            valid_idx  = valid_idx,
            tau_v      = tau_v,
            amp_v      = amp_v,
            bg_v       = bg_flat[valid_idx],
            decay_valid= flat[valid_idx],
            basis_best = basis_best,
            ny = ny, nx = nx,
            n_bins = n_bins,
        )
        return maps

    @staticmethod
    def _estimate_bg_batch(flat, valid_mask):
        """Per-pixel background estimate using the same logic as estimate_bg()."""
        n_pix, n_bins = flat.shape
        bg = np.zeros(n_pix, dtype=np.float32)
        peak_bins = flat.argmax(axis=1)               # (N_pix,)
        for i in np.where(valid_mask)[0]:
            bg[i] = estimate_bg(flat[i], int(peak_bins[i]))
        return bg