import multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import least_squares

from ..FLIM.models import reconvolution_model


class GPUBackend:

    def batch_fixed_tau(
        self,
        stack,                      # (ny, nx, n_bins) float
        A,                          # (n_bins, n_exp) - IRF-convolved basis
        taus_fixed,                 # (n_exp,) seconds
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback,
    ):
        raise NotImplementedError

    def batch_grid_scan_1exp(
        self,
        stack,                      # (ny, nx, n_bins)
        basis_grid,                 # (N_GRID, n_bins)
        bb_grid,                    # (N_GRID,) - ||basis[j]||^2
        tau_grid,                   # (N_GRID,) seconds
        min_photons,
        correct_pileup,
        n_sync_px,
        progress_callback,
    ):
        raise NotImplementedError

    def batch_free_tau_fit(
        self,
        stack,                      # (ny, nx, n_bins) float
        irf_array,                  # (n_bins,) real - time-domain IRF (normalised)
        tcspc_res,                  # seconds per bin
        taus_init,                  # (n_exp,) seconds - initial guess from global fit
        tau_min_s,                  # float
        tau_max_s,                  # float
        n_exp,
        min_photons,
        correct_pileup,
        n_sync_px,
        n_steps,
        lr,
    ):
        raise NotImplementedError


class _BackendMixin:
    @staticmethod
    def _init_maps(ny, nx, n_exp, intensity, taus_fixed_ns, free_tau):
        maps = dict(
            intensity    = intensity,
            tau_mean_int = np.full((ny, nx), np.nan),
            tau_mean_amp = np.full((ny, nx), np.nan),
            chi2_r       = np.full((ny, nx), np.nan),
        )
        for i in range(n_exp):
            maps[f"alpha_{i+1}"] = np.full((ny, nx), np.nan)
            maps[f"frac_{i+1}"]  = np.full((ny, nx), np.nan)
            maps[f"tau_{i+1}"]   = (
                np.full((ny, nx), np.nan) if (n_exp == 1 or free_tau)
                else np.full((ny, nx), taus_fixed_ns[i])
            )
            maps[f"a{i+1}"] = maps[f"alpha_{i+1}"]   # alias
        return maps

    @staticmethod
    def _scatter_fixed_tau(
        maps,
        valid_idx,                  # (N_valid,) flat pixel indices
        amps,                       # (N_valid, n_exp) non-negative
        bg,                         # (N_valid,)
        decay_valid,                # (N_valid, n_bins) raw counts
        A,                          # (n_bins, n_exp)
        taus_ns,                    # (n_exp,)
        ny, nx,
    ):
        n_exp = A.shape[1]
        n_bins = A.shape[0]

        amp_sum = amps.sum(axis=1)                      # (N_valid,)
        good    = amp_sum > 0
        if not good.any():
            return

        fracs    = np.where(good[:, None], amps / np.maximum(amp_sum[:, None], 1e-30), 0.0)
        taus_ns2 = taus_ns ** 2
        tau_amp  = (fracs * taus_ns[None, :]).sum(axis=1)   # (N_valid,)
        denom    = (amps * taus_ns[None, :]).sum(axis=1)
        tau_int  = np.where(denom > 0,
                            (amps * taus_ns2[None, :]).sum(axis=1) / np.maximum(denom, 1e-30),
                            np.nan)

        model    = decay_valid.copy()                    # allocate
        for j in range(n_exp):
            model = amps[:, j:j+1] * A[:, j][None, :]  # placeholder
        # Recompute properly: model = amps @ A.T + bg
        model    = amps @ A.T + bg[:, None]             # (N_valid, n_bins)
        resid    = decay_valid - model
        chi2     = (resid ** 2 / np.maximum(model, 1.0)).sum(axis=1)
        dof      = max(n_bins - n_exp, 1)

        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))

        maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_amp[good]
        maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_int[good]
        maps['chi2_r'][yi_arr[good], xi_arr[good]]       = chi2[good] / dof
        for i in range(n_exp):
            maps[f"alpha_{i+1}"][yi_arr[good], xi_arr[good]] = amps[good, i]
            maps[f"frac_{i+1}"][yi_arr[good], xi_arr[good]]  = fracs[good, i]

    @staticmethod
    def _scatter_1exp(
        maps,
        valid_idx,                  # (N_valid,) flat pixel indices
        tau_v,                      # (N_valid,) seconds
        amp_v,                      # (N_valid,)
        bg_v,                       # (N_valid,)
        decay_valid,                # (N_valid, n_bins)
        basis_best,                 # (N_valid, n_bins)
        ny, nx,
        n_bins,
    ):
        good   = amp_v > 0
        tau_ns = tau_v * 1e9
        model  = amp_v[:, None] * basis_best + bg_v[:, None]
        resid  = decay_valid - model
        chi2   = (resid ** 2 / np.maximum(model, 1.0)).sum(axis=1) / max(n_bins - 2, 1)

        yi_arr, xi_arr = np.unravel_index(valid_idx, (ny, nx))
        maps['tau_1'][yi_arr[good], xi_arr[good]]        = tau_ns[good]
        maps['tau_mean_amp'][yi_arr[good], xi_arr[good]] = tau_ns[good]
        maps['tau_mean_int'][yi_arr[good], xi_arr[good]] = tau_ns[good]
        maps['alpha_1'][yi_arr[good], xi_arr[good]]      = amp_v[good]
        maps['frac_1'][yi_arr[good], xi_arr[good]]       = 1.0
        maps['chi2_r'][yi_arr[good], xi_arr[good]]       = chi2[good]

    @staticmethod
    def _scatter_free_tau(
        maps,
        valid_idx,                  # (N_valid,) flat pixel indices
        taus_s,                     # (N_valid, n_exp) seconds, sorted ascending
        amps,                       # (N_valid, n_exp) non-negative
        chi2_r,                     # (N_valid,)
        ny, nx,
        n_exp,
    ):
        amp_sum = amps.sum(axis=1)                          # (N_valid,)
        good    = amp_sum > 0
        if not good.any():
            return

        fracs   = np.where(good[:, None], amps / np.maximum(amp_sum[:, None], 1e-30), 0.0)
        taus_ns = taus_s * 1e9                              # (N_valid, n_exp)
        tau_amp = (fracs * taus_ns).sum(axis=1)             # (N_valid,)
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
        for i in range(n_exp):
            maps[f"tau_{i+1}"][yi_arr[good], xi_arr[good]]   = taus_ns[good, i]
            maps[f"alpha_{i+1}"][yi_arr[good], xi_arr[good]] = amps[good, i]
            maps[f"frac_{i+1}"][yi_arr[good], xi_arr[good]]  = fracs[good, i]

    @staticmethod
    def _scipy_parallel_free_tau_fit(
        raw_valid,      # (B, n_bins) float32
        bg_valid,       # (B,)        float32
        irf_array,      # (n_bins,)   float64-compatible
        tcspc_res,      # float
        taus_init,      # (n_exp,)    seconds - initial guess
        tau_min_s,      # float
        tau_max_s,      # float
        n_exp,
        n_bins,
    ):
        B = raw_valid.shape[0]

        amp0    = float(raw_valid.max()) / n_exp
        # Use the same bounds as the CPU free-tau path in fit_per_pixel
        amp_hi  = float(raw_valid.max()) * 10.0
        lo_px   = np.array([float(tau_min_s)] * n_exp + [0.0]      * n_exp)
        hi_px   = np.array([float(tau_max_s)] * n_exp + [amp_hi]   * n_exp)

        def _fit_pixel(b):
            decay_b = raw_valid[b].astype(np.float64)
            bg_b    = float(bg_valid[b])
            wt      = np.sqrt(np.maximum(decay_b, 1.0))
            p0      = np.concatenate([taus_init,
                                      np.full(n_exp, amp0)])

            def _resid(p):
                # Exactly mirrors the CPU _resid in fit_per_pixel:
                #   full_p = [tau1..tauN, amp1..ampN, shift=0]
                #   reconvolution_model sorts taus descending internally
                full_p = np.concatenate([p[:n_exp], p[n_exp:], [0.0]])
                model  = reconvolution_model(
                    full_p, tcspc_res, n_bins, irf_array,
                    n_exp, bg_b, False, False, False)
                return (model - decay_b) / wt

            try:
                res = least_squares(_resid, p0, bounds=(lo_px, hi_px),
                                    method='trf', max_nfev=500,
                                    ftol=1e-8, xtol=1e-8, gtol=1e-8)
                return res.x
            except Exception:
                return None

        n_workers = min(B, max(1, multiprocessing.cpu_count()))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            solutions = list(pool.map(_fit_pixel, range(B)))

        taus_out  = np.zeros((B, n_exp), dtype=np.float32)
        amps_out  = np.zeros((B, n_exp), dtype=np.float32)
        chi2r_out = np.full(B, np.nan, dtype=np.float64)
        model_out = np.zeros((B, n_bins), dtype=np.float32)
        valid_b   = np.zeros(B, dtype=bool)

        for b, p_sol in enumerate(solutions):
            if p_sol is None:
                continue
            taus_b = p_sol[:n_exp];  amps_b = p_sol[n_exp:]
            if amps_b.sum() <= 0:
                continue
            # Sort ascending for output (matches CPU convention)
            order   = np.argsort(taus_b)
            taus_b  = taus_b[order];  amps_b = amps_b[order]
            # Compute chi2_r using reconvolution_model (Pearson, model-denominator)
            # to match the CPU's chi2_r calculation exactly.
            bg_b    = float(bg_valid[b])
            full_p  = np.concatenate([taus_b, amps_b, [0.0]])
            model_b = reconvolution_model(
                full_p, tcspc_res, n_bins, irf_array,
                n_exp, bg_b, False, False, False)
            resid_b = raw_valid[b].astype(np.float64) - model_b
            chi2_b  = (resid_b ** 2 / np.maximum(model_b, 1.0)).sum()
            dof     = max(n_bins - 2 * n_exp, 1)

            taus_out[b]  = taus_b.astype(np.float32)
            amps_out[b]  = amps_b.astype(np.float32)
            model_out[b] = model_b.astype(np.float32)
            chi2r_out[b] = chi2_b / dof
            valid_b[b]   = True

        return taus_out, amps_out, chi2r_out, model_out, valid_b
