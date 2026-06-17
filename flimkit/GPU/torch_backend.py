import numpy as np
from flimkit.GPU._base import _BackendMixin
from flimkit.FLIM.fit_tools import estimate_bg, coates_pileup_correction


class TorchBackend(_BackendMixin):

    def __init__(self, device='cuda'):
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
        torch = self._torch
        ny, nx, n_bins = stack.shape
        n_exp = A.shape[1]
        taus_ns = taus_fixed * 1e9

        # Compute pinv on CPU - linalg_svd (used internally by pinv) is not
        # supported on MPS and would silently fall back, triggering a UserWarning.
        A_cpu   = torch.as_tensor(A, dtype=torch.float32, device='cpu')
        A_pinv  = torch.linalg.pinv(A_cpu).to(self.device)

        flat    = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask = intensity_flat >= min_photons

        bg_flat = self._estimate_bg_batch(flat, valid_mask)

        data_corr = np.maximum(flat - bg_flat[:, None], 0.0)

        if correct_pileup and n_sync_px > 0:
            for idx in np.where(valid_mask)[0]:
                data_corr[idx] = coates_pileup_correction(data_corr[idx], n_sync_px)

        data_t   = torch.as_tensor(data_corr, dtype=torch.float32, device=self.device)
        amps_raw = data_t @ A_pinv.T
        amps_t   = torch.clamp(amps_raw, min=0.0)

        amps_np = amps_t.cpu().numpy()
        bg_np   = bg_flat

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

        dc_valid  = dc_flat[valid_idx]

        bg_t      = torch.as_tensor(bb_grid, dtype=torch.float32, device=self.device)
        basis_t   = torch.as_tensor(basis_grid, dtype=torch.float32, device=self.device)
        dc_t      = torch.as_tensor(dc_valid, dtype=torch.float32, device=self.device)

        bd        = dc_t @ basis_t.T
        dc_sq     = (dc_t ** 2).sum(dim=1)
        # cost = ||d||^2 - max(d·b, 0)^2 / ||b||^2; minimise → best τ per pixel
        costs     = dc_sq[:, None] - torch.clamp(bd, min=0.0) ** 2 / bg_t[None, :]
        best_g    = costs.argmin(dim=1).cpu().numpy()
        bd_np     = bd.cpu().numpy()
        bb_np     = bb_grid

        tau_v     = tau_grid[best_g]
        amp_v     = np.maximum(bd_np[np.arange(len(valid_idx)), best_g]
                               / bb_np[best_g], 0.0)
        basis_best = basis_grid[best_g]

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
        n_pix, n_bins = flat.shape
        bg = np.zeros(n_pix, dtype=np.float32)
        peak_bins = flat.argmax(axis=1)
        for i in np.where(valid_mask)[0]:
            bg[i] = estimate_bg(flat[i], int(peak_bins[i]))
        return bg

    def batch_free_tau_fit(
        self,
        stack,
        irf_array,
        tcspc_res,
        taus_init,
        tau_min_s,
        tau_max_s,
        n_exp,
        min_photons,
        correct_pileup,
        n_sync_px,
        n_steps=50,
        lr=None,
    ):
        torch = self._torch
        ny, nx, n_bins = stack.shape
        taus_ns_init   = taus_init * 1e9

        flat           = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask     = intensity_flat >= min_photons
        valid_idx      = np.where(valid_mask)[0]

        maps = self._init_maps(
            ny, nx, n_exp,
            intensity=stack.sum(axis=2),
            taus_fixed_ns=taus_ns_init,
            free_tau=True,
        )
        if valid_idx.size == 0:
            return maps

        bg_flat = self._estimate_bg_batch(flat, valid_mask)
        dc_flat = np.maximum(flat - bg_flat[:, None], 0.0)
        if correct_pileup and n_sync_px > 0:
            for idx in valid_idx:
                dc_flat[idx] = coates_pileup_correction(dc_flat[idx], n_sync_px)

        raw_valid = flat[valid_idx].astype(np.float32)
        bg_valid  = bg_flat[valid_idx].astype(np.float32)
        B         = len(valid_idx)

        taus_out, amps_out, chi2r_out, _, valid_b = self._scipy_parallel_free_tau_fit(
            raw_valid, bg_valid, irf_array, tcspc_res,
            taus_init, tau_min_s, tau_max_s, n_exp, n_bins,
        )

        self._scatter_free_tau(
            maps, valid_idx=valid_idx[valid_b],
            taus_s=taus_out[valid_b], amps=amps_out[valid_b],
            chi2_r=chi2r_out[valid_b],
            ny=ny, nx=nx, n_exp=n_exp,
        )
        return maps

    def batch_dist_scan_unimodal(
        self,
        stack,
        basis,
        bb_grid,
        param_pairs,
        irf_fixed,
        tcspc_res,
        n_bins,
        dist_type,
        min_photons,
        progress_callback=None,
    ):
        torch = self._torch
        ny, nx, _ = stack.shape

        flat           = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask     = intensity_flat >= min_photons
        valid_idx      = np.where(valid_mask)[0]

        maps = dict(
            intensity    = stack.sum(axis=2),
            tau_mean_amp = np.full((ny, nx), np.nan),
            tau_mean_int = np.full((ny, nx), np.nan),
            chi2_r       = np.full((ny, nx), np.nan),
            tau_center_1 = np.full((ny, nx), np.nan),
            width_1      = np.full((ny, nx), np.nan),
            alpha_1      = np.full((ny, nx), np.nan),
            frac_1       = np.full((ny, nx), np.nan),
        )
        if valid_idx.size == 0:
            return maps

        bg_flat  = self._estimate_bg_batch(flat, valid_mask)
        dc_flat  = np.maximum(flat - bg_flat[:, None], 0.0)
        dc_valid = dc_flat[valid_idx]

        basis_t = torch.as_tensor(basis,    dtype=torch.float32, device=self.device)
        bb_t    = torch.as_tensor(bb_grid,  dtype=torch.float32, device=self.device)
        dc_t    = torch.as_tensor(dc_valid, dtype=torch.float32, device=self.device)

        bd_t    = dc_t @ basis_t.T
        dc_sq_t = (dc_t ** 2).sum(dim=1)
        costs_t = dc_sq_t[:, None] - torch.clamp(bd_t, min=0.0) ** 2 / bb_t[None, :]
        best_g  = costs_t.argmin(dim=1).cpu().numpy()
        bd_np   = bd_t.cpu().numpy()

        tau_v   = param_pairs[best_g, 0]
        w_v     = param_pairs[best_g, 1]
        amp_v   = np.maximum(
            bd_np[np.arange(len(valid_idx)), best_g] / bb_grid[best_g].astype(np.float64), 0.0)

        good       = amp_v > 0
        tau_amp_ns = tau_v * 1e9
        tau_int_ns = (tau_v + w_v ** 2 / np.maximum(tau_v, 1e-15)) * 1e9

        basis_best = basis[best_g].astype(np.float64)
        model_v    = amp_v[:, None] * basis_best + bg_flat[valid_idx, None]
        resid_v    = dc_valid.astype(np.float64) - model_v
        chi2_v     = (resid_v ** 2 / np.maximum(model_v, 1.0)).sum(axis=1) / max(n_bins - 3, 1)

        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))
        maps['tau_center_1'][yi_arr[good], xi_arr[good]] = tau_amp_ns[good]
        maps['width_1'][yi_arr[good], xi_arr[good]]      = w_v[good] * 1e9
        maps['alpha_1'][yi_arr[good], xi_arr[good]]      = amp_v[good]
        maps['frac_1'][yi_arr[good], xi_arr[good]]       = 1.0
        maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_amp_ns[good]
        maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_int_ns[good]
        maps['chi2_r'][yi_arr[good], xi_arr[good]]       = chi2_v[good]
        return maps