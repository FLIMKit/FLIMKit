import numpy as np
from flimkit.GPU._base import _BackendMixin
from flimkit.FLIM.fit_tools import estimate_bg, coates_pileup_correction

class MLXBackend(_BackendMixin):

    def __init__(self):
        import mlx.core as mx
        self._mx = mx
        self.device = mx.Device(mx.gpu)

    def __repr__(self):
        return "MLXBackend(device='metal')"

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
        mx = self._mx
        ny, nx, n_bins = stack.shape
        n_exp = A.shape[1]
        taus_ns = taus_fixed * 1e9

        flat           = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask     = intensity_flat >= min_photons
        valid_idx      = np.where(valid_mask)[0]

        maps = self._init_maps(
            ny, nx, n_exp,
            intensity=stack.sum(axis=2),
            taus_fixed_ns=taus_ns,
            free_tau=False,
        )
        if valid_idx.size == 0:
            return maps

        bg_flat   = self._estimate_bg_batch(flat, valid_mask)
        dc_flat   = np.maximum(flat - bg_flat[:, None], 0.0)

        if correct_pileup and n_sync_px > 0:
            for idx in valid_idx:
                dc_flat[idx] = coates_pileup_correction(dc_flat[idx], n_sync_px)

        A_mx      = mx.array(A.astype(np.float32))
        A_pinv_mx = mx.linalg.pinv(A_mx)                   # (n_exp, n_bins)
        dc_mx     = mx.array(dc_flat)                       # (N_pix, n_bins)
        amps_raw  = dc_mx @ A_pinv_mx.T                     # (N_pix, n_exp)
        amps_mx   = mx.maximum(amps_raw, 0.0)               # clamp negatives
        mx.eval(amps_mx)                                    # force lazy eval

        amps_np = np.array(amps_mx)                         # zero-copy on Apple Silicon

        self._scatter_fixed_tau(
            maps,
            valid_idx  = valid_idx,
            amps       = amps_np[valid_idx],
            bg         = bg_flat[valid_idx],
            decay_valid= flat[valid_idx],
            A          = A,
            taus_ns    = taus_ns,
            ny=ny, nx=nx,
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
        mx = self._mx
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

        basis_mx  = mx.array(basis_grid.astype(np.float32))   # (N_GRID, n_bins)
        bb_mx     = mx.array(bb_grid.astype(np.float32))       # (N_GRID,)
        dc_mx     = mx.array(dc_valid)                         # (N_valid, n_bins)

        bd_mx     = dc_mx @ basis_mx.T                         # (N_valid, N_GRID)
        dc_sq_mx  = (dc_mx ** 2).sum(axis=1)                   # (N_valid,)
        # cost = ||d||^2 - max(d·b, 0)^2 / ||b||^2; argmin gives best τ per pixel
        costs_mx  = dc_sq_mx[:, None] - mx.maximum(bd_mx, 0.0) ** 2 / bb_mx[None, :]
        best_g_mx = costs_mx.argmin(axis=1)
        mx.eval(best_g_mx, bd_mx)                              # force lazy eval

        best_g    = np.array(best_g_mx)                        # (N_valid,)
        bd_np     = np.array(bd_mx)

        tau_v     = tau_grid[best_g]
        amp_v     = np.maximum(
            bd_np[np.arange(len(valid_idx)), best_g] / bb_grid[best_g], 0.0
        )
        basis_best = basis_grid[best_g]

        self._scatter_1exp(
            maps,
            valid_idx  = valid_idx,
            tau_v      = tau_v,
            amp_v      = amp_v,
            bg_v       = bg_flat[valid_idx],
            decay_valid= flat[valid_idx],
            basis_best = basis_best,
            ny=ny, nx=nx,
            n_bins=n_bins,
        )
        return maps

    # Internal helpers

    @staticmethod
    def _estimate_bg_batch(flat, valid_mask):
        n_pix = flat.shape[0]
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
        mx = self._mx
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

        raw_valid = flat[valid_idx].astype(np.float32)         # (B, n_bins)
        bg_valid  = bg_flat[valid_idx].astype(np.float32)      # (B,)
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

