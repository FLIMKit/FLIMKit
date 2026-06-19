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
        tvb_profile=None,
        fit_tvb=False,
    ):
        mx = self._mx
        ny, nx, n_bins = stack.shape
        n_exp = A.shape[1]
        taus_ns = taus_fixed * 1e9
        flat = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask = intensity_flat >= min_photons
        valid_idx = np.where(valid_mask)[0]
        maps = self._init_maps(
            ny, nx, n_exp,
            intensity=stack.sum(axis=2),
            taus_fixed_ns=taus_ns,
            free_tau=False,
        )
        if valid_idx.size == 0:
            return maps
        tvb_np = None
        if fit_tvb and tvb_profile is not None:
            B_col = np.asarray(tvb_profile, dtype=np.float32)
            A_aug = np.column_stack([A, B_col, np.ones(n_bins, dtype=np.float32)]).astype(np.float32)
            data_in = flat.copy()
            if correct_pileup and n_sync_px > 0:
                for idx in valid_idx:
                    data_in[idx] = coates_pileup_correction(data_in[idx], n_sync_px)
            A_pinv_mx = mx.linalg.pinv(mx.array(A_aug), stream=mx.cpu)
            coeffs_mx = mx.maximum(mx.array(data_in) @ A_pinv_mx.T, 0.0)
            mx.eval(coeffs_mx)
            coeffs = np.array(coeffs_mx)
            amps_np = coeffs[:, :n_exp]
            tvb_np = coeffs[:, n_exp]
            bg_np = coeffs[:, n_exp + 1]
        else:
            bg_flat = self._estimate_bg_batch(flat, valid_mask)
            dc_flat = np.maximum(flat - bg_flat[:, None], 0.0)
            if correct_pileup and n_sync_px > 0:
                for idx in valid_idx:
                    dc_flat[idx] = coates_pileup_correction(dc_flat[idx], n_sync_px)
            A_mx = mx.array(A.astype(np.float32))
            A_pinv_mx = mx.linalg.pinv(A_mx, stream=mx.cpu)
            amps_mx = mx.maximum(mx.array(dc_flat) @ A_pinv_mx.T, 0.0)
            mx.eval(amps_mx)
            amps_np = np.array(amps_mx)
            bg_np = bg_flat

        self._scatter_fixed_tau(
            maps,
            valid_idx = valid_idx,
            amps = amps_np[valid_idx],
            bg = bg_np[valid_idx],
            decay_valid= flat[valid_idx],
            A = A,
            taus_ns = taus_ns,
            ny=ny, nx=nx,
            tvb = tvb_np[valid_idx] if tvb_np is not None else None,
            tvb_profile = np.asarray(tvb_profile, dtype=np.float32)
                         if (fit_tvb and tvb_profile is not None) else None,
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
        tvb_profile=None,
        fit_tvb=False,
    ):
        mx = self._mx
        ny, nx, n_bins = stack.shape
        N_GRID = len(tau_grid)
        flat = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask = intensity_flat >= min_photons
        valid_idx = np.where(valid_mask)[0]
        maps = self._init_maps(
            ny, nx, n_exp=1,
            intensity=stack.sum(axis=2),
            taus_fixed_ns=np.array([tau_grid[N_GRID // 2] * 1e9]),
            free_tau=True,
        )
        if valid_idx.size == 0:
            return maps
        if fit_tvb and tvb_profile is not None:
            U, U_pinv, basis_perp, bb_perp = self._tvb_grid_prep(basis_grid, tvb_profile, n_bins)
            data_in = flat.copy()
            if correct_pileup and n_sync_px > 0:
                for idx in valid_idx:
                    data_in[idx] = coates_pileup_correction(data_in[idx], n_sync_px)
            d_valid = data_in[valid_idx]
            d_perp = self._tvb_project_data(d_valid.astype(np.float64), U, U_pinv).astype(np.float32)
            basis_mx = mx.array(basis_perp)
            bbp_mx = mx.array(bb_perp)
            dperp_mx = mx.array(d_perp)
            bd_mx = dperp_mx @ basis_mx.T
            dsq_mx = (dperp_mx ** 2).sum(axis=1)
            costs_mx = dsq_mx[:, None] - mx.maximum(bd_mx, 0.0) ** 2 / bbp_mx[None, :]
            best_g_mx = costs_mx.argmin(axis=1)
            mx.eval(best_g_mx, bd_mx)
            best_g = np.array(best_g_mx)
            bd_np = np.array(bd_mx)
            amp_v = np.maximum(bd_np[np.arange(len(valid_idx)), best_g] / bb_perp[best_g], 0.0)
            basis_best = basis_grid[best_g]
            resid_after = d_valid.astype(np.float64) - amp_v[:, None] * basis_best
            vz = resid_after @ U_pinv.T
            tvb_v = np.maximum(vz[:, 0], 0.0).astype(np.float32)
            bg_z = vz[:, 1].astype(np.float32)
            self._scatter_1exp(
                maps, valid_idx=valid_idx, tau_v=tau_grid[best_g], amp_v=amp_v,
                bg_v=bg_z, decay_valid=flat[valid_idx], basis_best=basis_best,
                ny=ny, nx=nx, n_bins=n_bins,
                tvb=tvb_v, tvb_profile=np.asarray(tvb_profile, dtype=np.float32),
            )
            return maps
        bg_flat = self._estimate_bg_batch(flat, valid_mask)
        dc_flat = np.maximum(flat - bg_flat[:, None], 0.0)
        if correct_pileup and n_sync_px > 0:
            for idx in valid_idx:
                dc_flat[idx] = coates_pileup_correction(dc_flat[idx], n_sync_px)
        dc_valid = dc_flat[valid_idx]
        basis_mx = mx.array(basis_grid.astype(np.float32))
        bb_mx = mx.array(bb_grid.astype(np.float32))
        dc_mx = mx.array(dc_valid)
        bd_mx = dc_mx @ basis_mx.T
        dc_sq_mx = (dc_mx ** 2).sum(axis=1)
        # cost = ||d||^2 - max(d·b, 0)^2 / ||b||^2; argmin gives best τ per pixel
        costs_mx = dc_sq_mx[:, None] - mx.maximum(bd_mx, 0.0) ** 2 / bb_mx[None, :]
        best_g_mx = costs_mx.argmin(axis=1)
        mx.eval(best_g_mx, bd_mx)
        best_g = np.array(best_g_mx)
        bd_np = np.array(bd_mx)
        tau_v = tau_grid[best_g]
        amp_v = np.maximum(bd_np[np.arange(len(valid_idx)), best_g] / bb_grid[best_g], 0.0)
        basis_best = basis_grid[best_g]
        self._scatter_1exp(
            maps,
            valid_idx = valid_idx,
            tau_v = tau_v,
            amp_v = amp_v,
            bg_v = bg_flat[valid_idx],
            decay_valid = flat[valid_idx],
            basis_best = basis_best,
            ny=ny, nx=nx,
            n_bins=n_bins,
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
        mx = self._mx
        ny, nx, _ = stack.shape
        flat = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask = intensity_flat >= min_photons
        valid_idx = np.where(valid_mask)[0]
        maps = dict(
            intensity = stack.sum(axis=2),
            tau_mean_amp = np.full((ny, nx), np.nan),
            tau_mean_int = np.full((ny, nx), np.nan),
            chi2_r = np.full((ny, nx), np.nan),
            tau_center_1 = np.full((ny, nx), np.nan),
            width_1 = np.full((ny, nx), np.nan),
            alpha_1 = np.full((ny, nx), np.nan),
            frac_1 = np.full((ny, nx), np.nan),
        )
        if valid_idx.size == 0:
            return maps
        bg_flat = self._estimate_bg_batch(flat, valid_mask)
        dc_flat = np.maximum(flat - bg_flat[:, None], 0.0)
        dc_valid = dc_flat[valid_idx]
        basis_mx = mx.array(basis)
        bb_mx = mx.array(bb_grid)
        dc_mx = mx.array(dc_valid)
        bd_mx = dc_mx @ basis_mx.T
        dc_sq_mx = (dc_mx ** 2).sum(axis=1)
        costs_mx = dc_sq_mx[:, None] - mx.maximum(bd_mx, 0.0) ** 2 / bb_mx[None, :]
        best_g_mx = costs_mx.argmin(axis=1)
        mx.eval(best_g_mx, bd_mx)
        best_g = np.array(best_g_mx)
        bd_np = np.array(bd_mx)
        tau_v = param_pairs[best_g, 0]
        w_v = param_pairs[best_g, 1]
        amp_v = np.maximum(bd_np[np.arange(len(valid_idx)), best_g] / bb_grid[best_g].astype(np.float64), 0.0)
        good = amp_v > 0
        tau_amp_ns = tau_v * 1e9
        tau_int_ns = (tau_v + w_v ** 2 / np.maximum(tau_v, 1e-15)) * 1e9
        basis_best = basis[best_g].astype(np.float64)
        model_v = amp_v[:, None] * basis_best + bg_flat[valid_idx, None]
        resid_v = dc_valid.astype(np.float64) - model_v
        chi2_v = (resid_v ** 2 / np.maximum(model_v, 1.0)).sum(axis=1) / max(n_bins - 3, 1)
        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))
        maps['tau_center_1'][yi_arr[good], xi_arr[good]] = tau_amp_ns[good]
        maps['width_1'][yi_arr[good], xi_arr[good]] = w_v[good] * 1e9
        maps['alpha_1'][yi_arr[good], xi_arr[good]] = amp_v[good]
        maps['frac_1'][yi_arr[good], xi_arr[good]] = 1.0
        maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_amp_ns[good]
        maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_int_ns[good]
        maps['chi2_r'][yi_arr[good], xi_arr[good]] = chi2_v[good]
        return maps


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
        tvb_profile=None,
        fit_tvb=False,
    ):
        mx = self._mx
        ny, nx, n_bins = stack.shape
        taus_ns_init = taus_init * 1e9
        flat = stack.reshape(ny * nx, n_bins).astype(np.float32)
        intensity_flat = flat.sum(axis=1)
        valid_mask = intensity_flat >= min_photons
        valid_idx = np.where(valid_mask)[0]
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
        bg_valid = bg_flat[valid_idx].astype(np.float32)
        B = len(valid_idx)
        taus_out, amps_out, chi2r_out, _, valid_b, tvb_out = self._scipy_parallel_free_tau_fit(
            raw_valid, bg_valid, irf_array, tcspc_res,
            taus_init, tau_min_s, tau_max_s, n_exp, n_bins,
            tvb_profile=tvb_profile, fit_tvb=fit_tvb,
        )
        self._scatter_free_tau(
            maps, valid_idx=valid_idx[valid_b],
            taus_s=taus_out[valid_b], amps=amps_out[valid_b],
            chi2_r=chi2r_out[valid_b],
            ny=ny, nx=nx, n_exp=n_exp,
            tvb=tvb_out[valid_b] if fit_tvb else None,
        )
        return maps