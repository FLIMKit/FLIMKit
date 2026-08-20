import numpy as np
from scipy.optimize import least_squares
from ..FLIM.fit_tools import calibrated_chi2, calibrated_from_terms, chi2_terms
from ..FLIM.models import reconvolution_model

def fit_window(fit_idx, n_bins):
    if fit_idx is None:
        return None
    raw = np.asarray(fit_idx)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError('fit_idx must be a non-empty one-dimensional array')
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('fit_idx must contain integer bin indices')
    idx = raw.astype(int, copy=False)
    if np.any(idx < 0) or np.any(idx >= n_bins):
        raise ValueError(f'fit_idx entries must be between 0 and {n_bins - 1}')
    if np.unique(idx).size != idx.size:
        raise ValueError('fit_idx must not contain duplicate bins')
    if np.array_equal(idx, np.arange(n_bins)):
        return None
    return idx


GPU_BLOCK_BYTES = 256 * 1024 * 1024
CUDA_BLOCK_BYTES = 32 * 1024 * 1024

def gpu_block_bytes(default=GPU_BLOCK_BYTES):
    import os
    override = os.environ.get('FLIMKIT_GPU_BLOCK_BYTES')
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return default

def pixel_blocks(n_pixels, bytes_per_pixel, budget=None):
    budget = gpu_block_bytes() if budget is None else budget
    step = max(1, int(budget // max(int(bytes_per_pixel), 1)))
    for start in range(0, n_pixels, step):
        yield start, min(start + step, n_pixels)


class GPUBackend:

    def batch_fixed_tau(
        self,
        stack,
        A,
        taus_fixed,
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback,
        fit_idx=None,
    ):
        raise NotImplementedError

    def batch_grid_scan_1exp(
        self,
        stack,
        basis_grid,
        bb_grid,
        tau_grid,
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback,
        fit_idx=None,
    ):
        raise NotImplementedError

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
        n_steps,
        lr,
        fit_idx=None,
    ):
        raise NotImplementedError

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
        progress_callback,
        tvb_profile=None,
        fit_tvb=False,
        fit_idx=None,
    ):
        raise NotImplementedError

class _BackendMixin:

    def block_bytes(self):
        return gpu_block_bytes()


    @staticmethod
    def _estimate_bg_batch(flat, valid_mask, pre_gap=5):
        n_pix, n_bins = flat.shape
        bg = np.zeros(n_pix, dtype=np.float32)
        wanted = np.asarray(valid_mask, dtype=bool)
        if not wanted.any():
            return bg
        ends = np.maximum(flat.argmax(axis=1) - pre_gap, 0)
        for end in np.unique(ends[wanted]):
            rows = np.flatnonzero(wanted & (ends == end))
            if end >= 5:
                region = flat[rows, :end]
            else:
                region = flat[rows, -30:]
            bg[rows] = np.maximum(np.median(region, axis=1), 0.0)
        return bg
    @staticmethod
    def _init_maps(ny, nx, n_exp, intensity, taus_fixed_ns, free_tau):
        maps = dict(
            intensity    = intensity,
            tau_mean_int = np.full((ny, nx), np.nan),
            tau_mean_amp = np.full((ny, nx), np.nan),
            chi2_r       = np.full((ny, nx), np.nan),
            calibrated_chi2_r = np.full((ny, nx), np.nan),
        )
        for i in range(n_exp):
            maps[f"alpha_{i+1}"] = np.full((ny, nx), np.nan)
            maps[f"frac_{i+1}"] = np.full((ny, nx), np.nan)
            maps[f"tau_{i+1}"] = (np.full((ny, nx), np.nan) if (n_exp == 1 or free_tau)
                else np.full((ny, nx), taus_fixed_ns[i]))
            maps[f"a{i+1}"] = maps[f"alpha_{i+1}"]
        return maps

    @staticmethod
    def _scatter_fixed_tau(
        maps,
        valid_idx,
        amps,
        bg,
        decay_valid,
        A,
        taus_ns,
        ny, nx,
        tvb=None,
        tvb_profile=None,
    ):
        n_exp = A.shape[1]
        n_bins = A.shape[0]
        amp_sum = amps.sum(axis=1)
        good = amp_sum > 0
        if not good.any():
            return
        fracs = np.where(good[:, None], amps / np.maximum(amp_sum[:, None], 1e-30), 0.0)
        taus_ns2 = taus_ns ** 2
        tau_amp = (fracs * taus_ns[None, :]).sum(axis=1)
        denom = (amps * taus_ns[None, :]).sum(axis=1)
        tau_int = np.where(denom > 0,(amps * taus_ns2[None, :]).sum(axis=1) / np.maximum(denom, 1e-30), np.nan)
        model = amps @ A.T + bg[:, None]
        if tvb is not None and tvb_profile is not None:
            model = model + tvb[:, None] * tvb_profile[None, :]
        numerator, expected, row_ok = chi2_terms(decay_valid, model, axis=1)
        chi2 = numerator
        calibrated = calibrated_from_terms(numerator, expected, row_ok)
        dof = max(n_bins - n_exp, 1)

        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))

        maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_amp[good]
        maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_int[good]
        maps['chi2_r'][yi_arr[good], xi_arr[good]]       = chi2[good] / dof
        maps['calibrated_chi2_r'][yi_arr[good], xi_arr[good]] = calibrated[good]
        for i in range(n_exp):
            maps[f"alpha_{i+1}"][yi_arr[good], xi_arr[good]] = amps[good, i]
            maps[f"frac_{i+1}"][yi_arr[good], xi_arr[good]]  = fracs[good, i]
        if tvb is not None:
            maps.setdefault('tvb_scale', np.full((ny, nx), np.nan))
            maps['tvb_scale'][yi_arr[good], xi_arr[good]] = tvb[good]

    @staticmethod
    def _tvb_grid_prep(basis_grid, tvb_profile, n_bins):
        U = np.column_stack([np.asarray(tvb_profile, dtype=np.float64), np.ones(n_bins)])
        U_pinv = np.linalg.pinv(U)
        basis_perp = basis_grid - (basis_grid @ U_pinv.T) @ U.T
        bb_perp = np.maximum((basis_perp ** 2).sum(axis=1), 1e-20)
        return U, U_pinv, basis_perp.astype(np.float32), bb_perp.astype(np.float32)

    @staticmethod
    def _tvb_project_data(data, U, U_pinv):
        return data - (data @ U_pinv.T) @ U.T

    @staticmethod
    def _scatter_1exp(
        maps,
        valid_idx,
        tau_v,
        amp_v,
        bg_v,
        decay_valid,
        basis_best,
        ny, nx,
        n_bins,
        tvb=None,
        tvb_profile=None,
    ):
        good   = amp_v > 0
        tau_ns = tau_v * 1e9
        model  = amp_v[:, None] * basis_best + bg_v[:, None]
        if tvb is not None and tvb_profile is not None:
            model = model + tvb[:, None] * tvb_profile[None, :]
        numerator, expected, row_ok = chi2_terms(decay_valid, model, axis=1)
        chi2 = numerator / max(n_bins - 2, 1)
        calibrated = calibrated_from_terms(numerator, expected, row_ok)

        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))
        maps['tau_1'][yi_arr[good], xi_arr[good]]        = tau_ns[good]
        maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_ns[good]
        maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_ns[good]
        maps['alpha_1'][yi_arr[good], xi_arr[good]]      = amp_v[good]
        maps['frac_1'][yi_arr[good], xi_arr[good]]       = 1.0
        maps['chi2_r'][yi_arr[good], xi_arr[good]]       = chi2[good]
        maps['calibrated_chi2_r'][yi_arr[good], xi_arr[good]] = calibrated[good]
        if tvb is not None:
            maps.setdefault('tvb_scale', np.full((ny, nx), np.nan))
            maps['tvb_scale'][yi_arr[good], xi_arr[good]] = tvb[good]

    @staticmethod
    def _scatter_free_tau(
        maps,
        valid_idx,
        taus_s,
        amps,
        chi2_r,
        calibrated_values,
        ny, nx,
        n_exp,
        tvb=None,
    ):
        amp_sum = amps.sum(axis=1)
        good    = amp_sum > 0
        if not good.any():
            return

        fracs   = np.where(good[:, None], amps / np.maximum(amp_sum[:, None], 1e-30), 0.0)
        taus_ns = taus_s * 1e9
        tau_amp = (fracs * taus_ns).sum(axis=1)
        denom   = (amps  * taus_ns).sum(axis=1)
        tau_int = np.where(
            denom > 0,
            (amps * taus_ns ** 2).sum(axis=1) / np.maximum(denom, 1e-30),
            np.nan,
        )

        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))
        maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_amp[good]
        maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_int[good]
        maps['chi2_r'][yi_arr[good], xi_arr[good]]       = chi2_r[good]
        maps['calibrated_chi2_r'][yi_arr[good], xi_arr[good]] = calibrated_values[good]
        for i in range(n_exp):
            maps[f"tau_{i+1}"][yi_arr[good], xi_arr[good]]   = taus_ns[good, i]
            maps[f"alpha_{i+1}"][yi_arr[good], xi_arr[good]] = amps[good, i]
            maps[f"frac_{i+1}"][yi_arr[good], xi_arr[good]]  = fracs[good, i]
        if tvb is not None:
            maps.setdefault('tvb_scale', np.full((ny, nx), np.nan))
            maps['tvb_scale'][yi_arr[good], xi_arr[good]] = tvb[good]

    @staticmethod
    def _scipy_parallel_free_tau_fit(
        raw_valid,
        bg_valid,
        irf_array,
        tcspc_res,
        taus_init,
        tau_min_s,
        tau_max_s,
        n_exp,
        n_bins,
        tvb_profile=None,
        fit_tvb=False,
        fit_idx=None,
    ):
        B = raw_valid.shape[0]
        win = fit_window(fit_idx, n_bins)
        n_fit = n_bins if win is None else len(win)

        amp0    = float(raw_valid.max()) / n_exp
        # Use the same bounds as the CPU free-tau path in fit_per_pixel
        amp_hi  = float(raw_valid.max()) * 10.0
        lo_px   = np.array([float(tau_min_s)] * n_exp + [0.0]      * n_exp)
        hi_px   = np.array([float(tau_max_s)] * n_exp + [amp_hi]   * n_exp)
        if fit_tvb:
            tvb_hi = float(raw_valid.sum(axis=1).max())
            lo_px  = np.concatenate([lo_px, [0.0]])
            hi_px  = np.concatenate([hi_px, [tvb_hi]])

        def _fit_pixel(b):
            decay_b = raw_valid[b].astype(np.float64)
            bg_b    = float(bg_valid[b])
            wt      = np.sqrt(np.maximum(decay_b, 1.0))
            p0      = np.concatenate([taus_init,
                                      np.full(n_exp, amp0)])
            if fit_tvb:
                p0 = np.concatenate([p0, [bg_b * n_bins]])

            def _resid(p):
                if fit_tvb:
                    full_p = np.concatenate([p[:n_exp], p[n_exp:2 * n_exp], [0.0], [p[2 * n_exp]]])
                    model  = reconvolution_model(
                        full_p, tcspc_res, n_bins, irf_array,
                        n_exp, 0.0, False, False, False,
                        tvb_profile=tvb_profile, fit_tvb=True)
                else:
                    full_p = np.concatenate([p[:n_exp], p[n_exp:], [0.0]])
                    model  = reconvolution_model(
                        full_p, tcspc_res, n_bins, irf_array,
                        n_exp, bg_b, False, False, False)
                if win is None:
                    return (model - decay_b) / wt
                return (model[win] - decay_b[win]) / wt[win]

            try:
                res = least_squares(_resid, p0, bounds=(lo_px, hi_px),
                                    method='trf', max_nfev=500,
                                    ftol=1e-8, xtol=1e-8, gtol=1e-8)
                return res.x
            except Exception:
                return None

        solutions = [_fit_pixel(b) for b in range(B)]

        taus_out  = np.zeros((B, n_exp), dtype=np.float32)
        amps_out  = np.zeros((B, n_exp), dtype=np.float32)
        chi2r_out = np.full(B, np.nan, dtype=np.float64)
        chi2c_out = np.full(B, np.nan, dtype=np.float64)
        model_out = np.zeros((B, n_bins), dtype=np.float32)
        tvb_out   = np.zeros(B, dtype=np.float32)
        valid_b   = np.zeros(B, dtype=bool)

        for b, p_sol in enumerate(solutions):
            if p_sol is None:
                continue
            taus_b = p_sol[:n_exp];  amps_b = p_sol[n_exp:2 * n_exp]
            if amps_b.sum() <= 0:
                continue
            tvb_b = float(p_sol[2 * n_exp]) if fit_tvb else 0.0
            # Sort ascending for output (matches CPU convention)
            order   = np.argsort(taus_b)
            taus_b  = taus_b[order];  amps_b = amps_b[order]
            bg_b    = float(bg_valid[b])
            if fit_tvb:
                full_p  = np.concatenate([taus_b, amps_b, [0.0], [tvb_b]])
                model_b = reconvolution_model(
                    full_p, tcspc_res, n_bins, irf_array,
                    n_exp, 0.0, False, False, False,
                    tvb_profile=tvb_profile, fit_tvb=True)
            else:
                full_p  = np.concatenate([taus_b, amps_b, [0.0]])
                model_b = reconvolution_model(
                    full_p, tcspc_res, n_bins, irf_array,
                    n_exp, bg_b, False, False, False)
            decay_fit = raw_valid[b].astype(np.float64)
            if win is None:
                model_fit = model_b
            else:
                decay_fit = decay_fit[win]
                model_fit = model_b[win]
            resid_b = decay_fit - model_fit
            chi2_b  = (resid_b ** 2 / np.maximum(model_fit, 1.0)).sum()
            dof     = max(n_fit - 2 * n_exp, 1)

            taus_out[b]  = taus_b.astype(np.float32)
            amps_out[b]  = amps_b.astype(np.float32)
            model_out[b] = model_b.astype(np.float32)
            tvb_out[b]   = tvb_b
            chi2r_out[b] = chi2_b / dof
            chi2c_out[b] = calibrated_chi2(decay_fit, model_fit)
            valid_b[b]   = True

        return taus_out, amps_out, chi2r_out, chi2c_out, model_out, valid_b, tvb_out
