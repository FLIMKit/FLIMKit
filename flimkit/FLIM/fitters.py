import time
import os
import numpy as np
from tqdm import tqdm
tqdm.disable = True
from scipy.optimize import least_squares, differential_evolution, nnls
from scipy.stats.distributions import chi2 as chi2_dist
from ..FLIM.irf_tools import build_full_irf
from ..FLIM.fit_tools import (TAU_FIT_UNIT_S, estimate_bg, find_fit_start, find_fit_end, _build_bounds,
                              _pack_p0, coates_pileup_correction, bins_from_ns, build_fit_idx,
                              find_tail_fit_start, _build_bounds_tail, _pack_p0_tail,
                              calibrated_chi2, distribution_dof)
from ..FLIM.models import (reconvolution_model, _DECost, _DECostLogTau,
                           _DECostPoisson, _DECostPoissonLogTau,
                           dist_reconvolution_model, build_dist_basis_grid,
                           _DECostDist, _DECostDistLogParam,
                           _DECostDistPoisson, _DECostDistPoissonLogParam,
                           tail_model, tail_basis, unpack_tail_params,
                           _DECostTail, _DECostTailLogTau,
                           _DECostTailPoisson, _DECostTailPoissonLogTau)
from ..FLIM.fit_tools import (_build_bounds_dist, _pack_p0_dist)
from ..configs import MIN_PHOTONS_PERPIX

_GPU_BACKEND_UNSET = object()
_gpu_backend_cache = _GPU_BACKEND_UNSET
_GPU_MAX_DIST_STACK_BYTES = 1_000_000_000
_FREE_TAU_WARN_PIXELS = 50_000
_TAU_GRID_POINTS = 1600

def tau_grid_points():
    override = os.environ.get('FLIMKIT_TAU_GRID_POINTS')
    if override:
        try:
            return max(2, int(override))
        except ValueError:
            pass
    return _TAU_GRID_POINTS

def _init_gpu_backend():
    global _gpu_backend_cache
    if _gpu_backend_cache is not _GPU_BACKEND_UNSET:
        return _gpu_backend_cache
    try:
        from flimkit.GPU import get_backend
        _gpu_backend_cache = get_backend()
    except Exception:
        _gpu_backend_cache = None
    return _gpu_backend_cache

def warmup_gpu_backend():
    import threading
    global _gpu_backend_cache
    if _gpu_backend_cache is not _GPU_BACKEND_UNSET:
        return
    def _warmup():
        global _gpu_backend_cache
        if _gpu_backend_cache is not _GPU_BACKEND_UNSET:
            return
        try:
            from flimkit.GPU import get_backend
            _gpu_backend_cache = get_backend()
        except Exception:
            _gpu_backend_cache = None
    threading.Thread(target=_warmup, daemon=True).start()

def fit_summed(decay, tcspc_res, n_bins, irf_prompt,
               has_tail, fit_bg, fit_sigma,
               n_exp, tau_min_ns, tau_max_ns,
               optimizer='de', n_restarts=8,
               de_popsize=15, de_maxiter=1000,
               workers=-1, polish=True,
               cost_function='poisson',
               sigma_max=3.0,
               irf_shift_bins=2,
               tvb_profile=None, fit_tvb=False,
               fit_start_ns=None, fit_end_ns=None, exclude_ns=None,
               n_sync=None):
    warmup_gpu_backend()
    tau_min = tau_min_ns * 1e-9
    tau_max = tau_max_ns * 1e-9
    if cost_function not in ('chi2', 'poisson'):
        raise ValueError(f"Unknown cost_function: {cost_function!r}")
    decay_work = decay.astype(float)    
    if decay_work.max() == 0:
        raise ValueError('Decay has zero maximum - cannot fit.')
    scale = 1.0
    peak_bin = int(np.argmax(decay_work))
    bg_init = estimate_bg(decay_work, peak_bin)
    bg_fixed = bg_init if not fit_bg else 0.0
    fit_end = find_fit_end(decay_work, peak_bin, tau_max, tcspc_res, n_bins)
    standard_fit_end = int(round(44.9455 / (tcspc_res * 1e9)))
    fit_end = min(fit_end, standard_fit_end)
    fit_start = find_fit_start(decay_work, irf_prompt, tcspc_res)
    fit_start = max(0, min(fit_start, fit_end - 10))
    if fit_end_ns is not None:
        fit_end = min(n_bins, bins_from_ns(fit_end_ns, tcspc_res))
    if fit_start_ns is not None:
        fit_start = max(0, min(bins_from_ns(fit_start_ns, tcspc_res), fit_end - 2))
    exclude_bins = [(bins_from_ns(lo, tcspc_res), bins_from_ns(hi, tcspc_res))
                    for lo, hi in (exclude_ns or [])]
    fit_idx = build_fit_idx(fit_start, fit_end, n_bins, exclude_bins)
    bg_upper = max(bg_init * 2.0, bg_init + 10.0)
    if fit_tvb and tvb_profile is None:
        raise ValueError('fit_tvb=True requires a tvb_profile.')
    tvb_init = float(bg_init * n_bins) if fit_tvb else 0.0
    tvb_upper = float(decay_work.sum()) if fit_tvb else None
    if fit_tvb:
        print(f"  TVB: free scale on measured profile, init={tvb_init:.1f}, upper={tvb_upper:.1f}")
    print(f"  Cost function: {cost_function}")
    print(f"  bg initial guess = {bg_init:.3f} cts/bin"
          f", upper bound = {bg_upper:.3f} "
          f"({'free param' if fit_bg else 'fixed'})")
    print(f"  σ broadening: {'free param (σ≤' + f'{sigma_max:.1f})' if fit_sigma else 'fixed at 0'}")
    print(f"  Fit window: bins {fit_start}-{fit_end} "
          f"({fit_start*tcspc_res*1e9:.2f}-{fit_end*tcspc_res*1e9:.2f} ns), "
          f"{len(fit_idx)} bins fitted"
          f"{' (user-set)' if (fit_start_ns is not None or fit_end_ns is not None) else ' (auto)'}")
    if exclude_bins:
        for (lo_b, hi_b), (lo_n, hi_n) in zip(exclude_bins, exclude_ns):
            print(f"    Excluded: bins {lo_b}-{hi_b} ({lo_n:.2f}-{hi_n:.2f} ns)")
    if n_sync:
        print(f"  Pile-up: in forward model (N_sync={n_sync:,}), data left raw")
    lo, hi = _build_bounds(n_exp, tau_min, tau_max, decay_work.max(),
                             has_tail, fit_bg, fit_sigma,
                             bg_init=bg_init, bg_upper=bg_upper,
                             sigma_max=sigma_max, irf_shift_bins=irf_shift_bins,
                             fit_tvb=fit_tvb, tvb_init=tvb_init, tvb_upper=tvb_upper)
    bounds = list(zip(lo, hi))
    if cost_function == 'chi2':
        weights = np.sqrt(np.maximum(decay_work[fit_idx], 1.0))
        def residuals(params):
            model_vals = reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
            return (model_vals[fit_idx]
                    - decay_work[fit_idx]) / weights
    else:
        def residuals(params):
            model_vals = reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
            n = decay_work[fit_idx]
            m = np.maximum(model_vals[fit_idx], 1e-10)
            dev = m - n
            pos = n > 0
            dev[pos] += n[pos] * np.log(n[pos] / m[pos])
            dev = np.maximum(dev, 0.0)       
            r = np.sqrt(2.0 * dev)
            r[m < n] *= -1                   
            return r
    if optimizer == 'lm_multistart':
        n_log = n_exp
        lo_scaled = np.asarray(lo, dtype=float).copy()
        hi_scaled = np.asarray(hi, dtype=float).copy()
        lo_scaled[:n_log] = np.log10(lo_scaled[:n_log])
        hi_scaled[:n_log] = np.log10(hi_scaled[:n_log])
        def to_scaled(params):
            scaled = np.asarray(params, dtype=float).copy()
            scaled[:n_log] = np.log10(scaled[:n_log])
            return scaled
        def from_scaled(params):
            linear = np.asarray(params, dtype=float).copy()
            linear[:n_log] = 10.0 ** linear[:n_log]
            return linear
        def residuals_scaled(params):
            return residuals(from_scaled(params))
        rng = np.random.default_rng(42)
        best_res = None
        best_cost = np.inf
        for i in range(n_restarts + 1):
            tau_ov = None if i == 0 else np.sort(
                np.exp(rng.uniform(np.log(tau_min*1.001),
                                   np.log(tau_max*0.999), n_exp)))
            p0 = _pack_p0(n_exp, tau_min, tau_max, float(decay_work.max()),
                          has_tail, fit_bg, fit_sigma, bg_init,
                          tau_override=tau_ov, fit_tvb=fit_tvb, tvb_init=tvb_init)
            try:
                res = least_squares(residuals_scaled, to_scaled(p0),
                                    bounds=(lo_scaled, hi_scaled), method='trf',
                                    max_nfev=50000,
                                    ftol=1e-13, xtol=1e-13, gtol=1e-13)
            except Exception as exc:
                print(f"    Restart {i:2d}: failed ({exc})")
                continue
            tag = 'log-spaced' if i == 0 else 'random    '
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}  ← best")
            else:
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}")
        if best_res is None:
            raise RuntimeError('All restarts failed.')
        popt_work = from_scaled(best_res.x)
        message = best_res.message
    elif optimizer == 'de':
        print(f"  Differential evolution: popsize={de_popsize}, "
              f"maxiter={de_maxiter}, workers={workers}")
        bounds_log = list(bounds)
        for i in range(n_exp):
            lo_tau, hi_tau = bounds[i]
            bounds_log[i] = (np.log10(lo_tau), np.log10(hi_tau))
        if cost_function == 'poisson':
            cost_fn = _DECostPoissonLogTau(
                tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                has_tail, fit_bg, fit_sigma,
                fit_idx, decay_work,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
        else:
            cost_fn = _DECostLogTau(
                tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                has_tail, fit_bg, fit_sigma,
                fit_idx, decay_work, weights,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
        de_res = differential_evolution(
            cost_fn, bounds=bounds_log,
            maxiter=de_maxiter, popsize=de_popsize,
            workers=workers, seed=42,
            updating='deferred' if workers != 1 else 'immediate',
            init='sobol',
            disp=False)
        popt_work = de_res.x.copy()
        popt_work[:n_exp] = 10.0 ** popt_work[:n_exp]
        message = f"DE success={de_res.success}, fun={de_res.fun:.4e}"
        if polish:
            print('  Running final LM polish...')
            eps = 1e-10
            popt_work = np.clip(popt_work, np.asarray(lo) + eps, np.asarray(hi) - eps)
            try:
                pol = least_squares(residuals, popt_work, bounds=(lo, hi),
                                    method='trf', max_nfev=5000,
                                    ftol=1e-13, xtol=1e-13, gtol=1e-13)
                popt_work = pol.x
                message += f"; polished cost={pol.cost:.4e}"
            except ValueError as e:
                print(f"  Warning: LM polish failed ({e}) - using DE result")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    popt_original = popt_work.copy()
    summary = _make_summary(popt_original, decay, tcspc_res, n_bins, irf_prompt,
                            n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                            fit_idx, message, tvb_profile=tvb_profile,
                            fit_tvb=fit_tvb, n_sync=n_sync)
    return popt_original, summary


def _make_summary(popt, decay, tcspc_res, n_bins, irf_prompt,
                  n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                  fit_idx, message=None,
                  tvb_profile=None, fit_tvb=False, n_sync=None):
    fit_start = int(fit_idx[0])
    fit_end = int(fit_idx[-1]) + 1
    taus = popt[:n_exp]
    amps = popt[n_exp:2*n_exp]
    order = np.argsort(-taus)
    taus = taus[order]
    amps = amps[order]
    idx = 2 * n_exp
    shift = popt[idx]; idx += 1
    if fit_sigma:
        sigma = popt[idx]; idx += 1
    else:
        sigma = 0.0
    if fit_bg:
        bg_fit = popt[idx]; idx += 1
    else:
        bg_fit = bg_fixed
    if fit_tvb:
        tvb_scale = popt[idx]; idx += 1
    else:
        tvb_scale = 0.0
    if has_tail:
        tail_amp = popt[idx]
        tail_tau = popt[idx + 1]
    else:
        tail_amp = tail_tau = 0.0
    model = reconvolution_model(popt, tcspc_res, n_bins, irf_prompt,
                                   n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                                   tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
    d_win = decay[fit_idx].astype(float)
    m_win = model[fit_idx]
    sigma_w = np.sqrt(np.maximum(d_win, 1.0))
    chi2 = float(np.sum(((d_win - m_win) / sigma_w)**2))
    dof = max(len(fit_idx) - len(popt), 1)
    rchi2 = chi2 / dof
    p_val = float(1 - chi2_dist.cdf(chi2, df=dof))
    resid = (decay - model) / np.sqrt(np.maximum(model, 1.0))
    sigma_p = np.sqrt(np.maximum(m_win, 1.0))
    chi2_p = float(np.sum(((d_win - m_win) / sigma_p)**2))
    rchi2_p = chi2_p / dof
    calibrated_chi2_p = calibrated_chi2(d_win, m_win)
    peak_bin_loc = int(fit_idx[np.argmax(decay[fit_idx])])
    tail_start = peak_bin_loc + max(1, int(0.05 * (fit_end - peak_bin_loc)))
    tail_idx = fit_idx[fit_idx >= tail_start]
    d_tail = decay[tail_idx].astype(float)
    m_tail = model[tail_idx]
    sw_tail = np.sqrt(np.maximum(d_tail, 1.0))
    chi2_tail = float(np.sum(((d_tail - m_tail) / sw_tail)**2))
    dof_tail = max(len(tail_idx) - len(popt), 1)
    rchi2_tail = chi2_tail / dof_tail
    sp_tail = np.sqrt(np.maximum(m_tail, 1.0))
    chi2_tail_p = float(np.sum(((d_tail - m_tail) / sp_tail)**2))
    rchi2_tail_p = chi2_tail_p / dof_tail
    calibrated_chi2_tail_p = calibrated_chi2(d_tail, m_tail)
    amp_sum = amps.sum() if amps.sum() > 0 else 1.0
    fracs = amps / amp_sum
    tau_amp = float(np.dot(fracs, taus))
    tau_int = float(np.dot(amps, taus**2) / np.dot(amps, taus))
    above = np.where(irf_prompt >= irf_prompt.max() / 2)[0]
    fwhm_pr = (above[-1] - above[0]) if len(above) > 1 else 1
    fwhm_eff = np.sqrt(fwhm_pr**2 + (2.3548 * sigma)**2) * tcspc_res * 1e9
    return dict(
        tcspc_res = tcspc_res,
        taus_ns = taus * 1e9,
        amps = amps,
        fractions = fracs,
        bg_fit = bg_fit,
        tvb_scale = tvb_scale,
        tau_mean_amp_ns = tau_amp * 1e9,
        tau_mean_int_ns = tau_int * 1e9,
        chi2 = chi2,
        reduced_chi2 = rchi2,
        reduced_chi2_tail = rchi2_tail,
        chi2_pearson = chi2_p,
        reduced_chi2_pearson = rchi2_p,
        reduced_chi2_tail_pearson = rchi2_tail_p,
        calibrated_chi2_pearson = calibrated_chi2_p,
        calibrated_chi2_tail_pearson = calibrated_chi2_tail_p,
        tail_start_bin = tail_start,
        p_val = p_val,
        dof = dof,
        fit_window_bins = (fit_start, fit_end),
        fit_window_ns = (fit_start*tcspc_res*1e9, fit_end*tcspc_res*1e9),
        fit_idx = fit_idx,
        irf_shift_bins = shift,
        irf_sigma_bins = sigma,
        irf_fwhm_eff_ns = fwhm_eff,
        tail_amp = tail_amp,
        tail_tau_ns = tail_tau * tcspc_res * 1e9,
        model = model,
        residuals = resid,
        optimizer_msg = message,
    )


def _basis_rows(taus, t_axis, tcspc_res, n_bins, tail, irf_fft=None, t0=0.0):
    if tail:
        return tail_basis(tcspc_res, n_bins, taus, t0)
    basis = np.stack([np.exp(-t_axis / max(tau, 1e-15)) for tau in taus])
    return np.array([np.real(np.fft.ifft(np.fft.fft(b) * irf_fft)) for b in basis])


def fit_summed_tail(decay, tcspc_res, n_bins,
                    fit_bg, n_exp, tau_min_ns, tau_max_ns,
                    fit_t0=False, t0_range_bins=5.0,
                    optimizer='de', n_restarts=8,
                    de_popsize=15, de_maxiter=1000,
                    workers=-1, polish=True,
                    cost_function='poisson',
                    tvb_profile=None, fit_tvb=False,
                    fit_start_ns=None, fit_end_ns=None, exclude_ns=None,
                    n_sync=None):
    tau_min = tau_min_ns * 1e-9
    tau_max = tau_max_ns * 1e-9
    if cost_function not in ('chi2', 'poisson'):
        raise ValueError(f"Unknown cost_function: {cost_function!r}")
    decay_work = decay.astype(float)
    if decay_work.max() == 0:
        raise ValueError('Decay has zero maximum - cannot fit.')
    peak_bin = int(np.argmax(decay_work))
    bg_init = estimate_bg(decay_work, peak_bin)
    bg_fixed = bg_init if not fit_bg else 0.0
    t0_fixed = peak_bin * tcspc_res
    t0_range = float(t0_range_bins) * tcspc_res
    fit_end = find_fit_end(decay_work, peak_bin, tau_max, tcspc_res, n_bins)
    standard_fit_end = int(round(44.9455 / (tcspc_res * 1e9)))
    fit_end = min(fit_end, standard_fit_end)
    fit_start = find_tail_fit_start(decay_work, peak_bin, n_bins)
    fit_start = max(0, min(fit_start, fit_end - 10))
    if fit_end_ns is not None:
        fit_end = min(n_bins, bins_from_ns(fit_end_ns, tcspc_res))
    if fit_start_ns is not None:
        fit_start = max(0, min(bins_from_ns(fit_start_ns, tcspc_res), fit_end - 2))
    exclude_bins = [(bins_from_ns(lo, tcspc_res), bins_from_ns(hi, tcspc_res))
                    for lo, hi in (exclude_ns or [])]
    fit_idx = build_fit_idx(fit_start, fit_end, n_bins, exclude_bins)
    bg_upper = max(bg_init * 2.0, bg_init + 10.0)
    if fit_tvb and tvb_profile is None:
        raise ValueError('fit_tvb=True requires a tvb_profile.')
    tvb_init = float(bg_init * n_bins) if fit_tvb else 0.0
    tvb_upper = float(decay_work.sum()) if fit_tvb else None
    if fit_tvb:
        print(f"  TVB: free scale on measured profile, init={tvb_init:.1f}, upper={tvb_upper:.1f}")
    print(f"  Cost function: {cost_function}")
    print(f"  Tail fit: no IRF used, no reconvolution")
    print(f"  t0 = {t0_fixed*1e9:.3f} ns (decay peak, bin {peak_bin})"
          f"{f', free within ±{t0_range_bins:.0f} bins' if fit_t0 else ', fixed'}")
    print(f"  bg initial guess = {bg_init:.3f} cts/bin"
          f", upper bound = {bg_upper:.3f} "
          f"({'free param' if fit_bg else 'fixed'})")
    print(f"  Fit window: bins {fit_start}-{fit_end} "
          f"({fit_start*tcspc_res*1e9:.2f}-{fit_end*tcspc_res*1e9:.2f} ns), "
          f"{len(fit_idx)} bins fitted"
          f"{' (user-set)' if (fit_start_ns is not None or fit_end_ns is not None) else ' (auto)'}")
    if exclude_bins:
        for (lo_b, hi_b), (lo_n, hi_n) in zip(exclude_bins, exclude_ns):
            print(f"    Excluded: bins {lo_b}-{hi_b} ({lo_n:.2f}-{hi_n:.2f} ns)")
    if n_sync:
        print(f"  Pile-up: in forward model (N_sync={n_sync:,}), data left raw")
    lo, hi = _build_bounds_tail(n_exp, tau_min, tau_max, decay_work.max(), fit_bg,
                                bg_init=bg_init, bg_upper=bg_upper,
                                fit_t0=fit_t0, t0_init=t0_fixed, t0_range=t0_range,
                                fit_tvb=fit_tvb, tvb_init=tvb_init, tvb_upper=tvb_upper)
    bounds = list(zip(lo, hi))
    def _model_of(params):
        return tail_model(params, tcspc_res, n_bins, n_exp, bg_fixed, fit_bg,
                          fit_t0=fit_t0, t0_fixed=t0_fixed,
                          tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
    if cost_function == 'chi2':
        weights = np.sqrt(np.maximum(decay_work[fit_idx], 1.0))
        def residuals(params):
            model_vals = _model_of(params)
            return (model_vals[fit_idx] - decay_work[fit_idx]) / weights
    else:
        def residuals(params):
            model_vals = _model_of(params)
            n = decay_work[fit_idx]
            m = np.maximum(model_vals[fit_idx], 1e-10)
            dev = m - n
            pos = n > 0
            dev[pos] += n[pos] * np.log(n[pos] / m[pos])
            dev = np.maximum(dev, 0.0)
            r = np.sqrt(2.0 * dev)
            r[m < n] *= -1
            return r
    if optimizer == 'lm_multistart':
        rng = np.random.default_rng(42)
        best_res = None
        best_cost = np.inf
        for i in range(n_restarts + 1):
            tau_ov = None if i == 0 else np.sort(
                np.exp(rng.uniform(np.log(tau_min*1.001),
                                   np.log(tau_max*0.999), n_exp)))
            p0 = _pack_p0_tail(n_exp, tau_min, tau_max, float(decay_work.max()),
                               fit_bg, bg_init, tau_override=tau_ov,
                               fit_t0=fit_t0, t0_init=t0_fixed,
                               fit_tvb=fit_tvb, tvb_init=tvb_init)
            try:
                res = least_squares(residuals, p0, bounds=(lo, hi), method='trf',
                                    max_nfev=50000,
                                    ftol=1e-13, xtol=1e-13, gtol=1e-13)
            except Exception as exc:
                print(f"    Restart {i:2d}: failed ({exc})")
                continue
            tag = 'log-spaced' if i == 0 else 'random    '
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}  ← best")
            else:
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}")
        if best_res is None:
            raise RuntimeError('All restarts failed.')
        popt_work = best_res.x
        message = best_res.message
    elif optimizer == 'de':
        print(f"  Differential evolution: popsize={de_popsize}, "
              f"maxiter={de_maxiter}, workers={workers}")
        bounds_log = list(bounds)
        for i in range(n_exp):
            lo_tau, hi_tau = bounds[i]
            bounds_log[i] = (np.log10(lo_tau), np.log10(hi_tau))
        if cost_function == 'poisson':
            cost_fn = _DECostTailPoissonLogTau(
                tcspc_res, n_bins, n_exp, bg_fixed, fit_bg,
                fit_idx, decay_work,
                fit_t0=fit_t0, t0_fixed=t0_fixed,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
        else:
            cost_fn = _DECostTailLogTau(
                tcspc_res, n_bins, n_exp, bg_fixed, fit_bg,
                fit_idx, decay_work, weights,
                fit_t0=fit_t0, t0_fixed=t0_fixed,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
        de_res = differential_evolution(
            cost_fn, bounds=bounds_log,
            maxiter=de_maxiter, popsize=de_popsize,
            workers=workers, seed=42,
            updating='deferred' if workers != 1 else 'immediate',
            init='sobol',
            disp=False)
        popt_work = de_res.x.copy()
        popt_work[:n_exp] = 10.0 ** popt_work[:n_exp]
        message = f"DE success={de_res.success}, fun={de_res.fun:.4e}"
        if polish:
            print('  Running final LM polish...')
            eps = 1e-10
            popt_work = np.clip(popt_work, np.asarray(lo) + eps, np.asarray(hi) - eps)
            try:
                pol = least_squares(residuals, popt_work, bounds=(lo, hi),
                                    method='trf', max_nfev=5000,
                                    ftol=1e-13, xtol=1e-13, gtol=1e-13)
                popt_work = pol.x
                message += f"; polished cost={pol.cost:.4e}"
            except ValueError as e:
                print(f"  Warning: LM polish failed ({e}) - using DE result")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    summary = _make_summary_tail(popt_work, decay, tcspc_res, n_bins,
                                 n_exp, bg_fixed, fit_bg, fit_idx, message,
                                 fit_t0=fit_t0, t0_fixed=t0_fixed,
                                 tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
    return popt_work, summary


def _make_summary_tail(popt, decay, tcspc_res, n_bins,
                       n_exp, bg_fixed, fit_bg, fit_idx, message=None,
                       fit_t0=False, t0_fixed=0.0,
                       tvb_profile=None, fit_tvb=False, n_sync=None):
    fit_start = int(fit_idx[0])
    fit_end = int(fit_idx[-1]) + 1
    taus, amps, t0, bg_fit, tvb_scale = unpack_tail_params(
        popt, n_exp, fit_t0, fit_bg, fit_tvb,
        t0_fixed=t0_fixed, bg_fixed=bg_fixed)
    order = np.argsort(-taus)
    taus = taus[order]
    amps = amps[order]
    model = tail_model(popt, tcspc_res, n_bins, n_exp, bg_fixed, fit_bg,
                       fit_t0=fit_t0, t0_fixed=t0_fixed,
                       tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
    d_win = decay[fit_idx].astype(float)
    m_win = model[fit_idx]
    sigma_w = np.sqrt(np.maximum(d_win, 1.0))
    chi2 = float(np.sum(((d_win - m_win) / sigma_w)**2))
    dof = max(len(fit_idx) - len(popt), 1)
    rchi2 = chi2 / dof
    p_val = float(1 - chi2_dist.cdf(chi2, df=dof))
    resid = (decay - model) / np.sqrt(np.maximum(model, 1.0))
    sigma_p = np.sqrt(np.maximum(m_win, 1.0))
    chi2_p = float(np.sum(((d_win - m_win) / sigma_p)**2))
    rchi2_p = chi2_p / dof
    calibrated_chi2_p = calibrated_chi2(d_win, m_win)
    amp_sum = amps.sum() if amps.sum() > 0 else 1.0
    fracs = amps / amp_sum
    tau_amp = float(np.dot(fracs, taus))
    tau_int = float(np.dot(amps, taus**2) / np.dot(amps, taus))
    intensities = amps * taus / tcspc_res
    i_sum = float(intensities.sum())
    return dict(
        fit_model = 'tail',
        tcspc_res = tcspc_res,
        taus_ns = taus * 1e9,
        amps = amps,
        fractions = fracs,
        intensities = intensities,
        intensity_fractions = intensities / (i_sum if i_sum > 0 else 1.0),
        i_sum = i_sum,
        a_sum = float(amps.sum()),
        t0_ns = t0 * 1e9,
        bg_fit = bg_fit,
        tvb_scale = tvb_scale,
        tau_mean_amp_ns = tau_amp * 1e9,
        tau_mean_int_ns = tau_int * 1e9,
        chi2 = chi2,
        reduced_chi2 = rchi2,
        reduced_chi2_tail = rchi2,
        chi2_pearson = chi2_p,
        reduced_chi2_pearson = rchi2_p,
        reduced_chi2_tail_pearson = rchi2_p,
        calibrated_chi2_pearson = calibrated_chi2_p,
        calibrated_chi2_tail_pearson = calibrated_chi2_p,
        tail_start_bin = fit_start,
        p_val = p_val,
        dof = dof,
        fit_idx = fit_idx,
        fit_window_bins = (fit_start, fit_end),
        fit_window_ns = (fit_start*tcspc_res*1e9, fit_end*tcspc_res*1e9),
        model = model,
        residuals = resid,
        optimizer_msg = message,
    )


def fit_per_pixel(stack, tcspc_res, n_bins, irf_prompt,
                  has_tail, fit_bg, fit_sigma,
                  global_popt, n_exp,
                  min_photons=MIN_PHOTONS_PERPIX,
                  tau_min_ns=None, tau_max_ns=None,
                  correct_pileup=False, n_sync=None, pileup_in_model=False,
                  fit_idx=None,
                  progress_callback=None,
                  free_tau=False,
                  use_gpu='auto',
                  gpu_backend=None,
                  tvb_profile=None, fit_tvb=False,
                  fit_model='reconv', fit_t0=False, t0_fixed=0.0):
    ny, nx, _ = stack.shape
    _tail = fit_model == 'tail'
    if free_tau:
        n_valid = int((stack.sum(axis=2) >= min_photons).sum())
        if n_valid > _FREE_TAU_WARN_PIXELS:
            print(f'  [!] free-tau per-pixel on {n_valid:,} pixels is slow (iterative fit each); '
                  f'fixed-tau is ~100x faster - untick free-tau or draw a smaller ROI')
    _n_sync_px = 0
    if correct_pileup and pileup_in_model:
        raise ValueError('pick one pile-up route: correct_pileup rescales the measured '
                         'decay, pileup_in_model folds pile-up into the fitted model')
    if pileup_in_model and (not free_tau or n_exp == 1 or _tail):
        raise ValueError('pileup_in_model needs a free-tau reconvolution fit with two '
                         'or more components; the one-exponential and fixed-tau paths '
                         'project onto a precomputed basis, and pile-up is not linear '
                         'in the amplitudes')
    if correct_pileup or pileup_in_model:
        if not n_sync:
            raise ValueError('pile-up correction requested but this file exposes no '
                             'excitation-pulse count (N_sync); Coates is unavailable here')
        _n_sync_px = int(n_sync / max(ny * nx, 1))
        _max_px = float(stack.sum(axis=2).max())
        if _n_sync_px <= _max_px:
            raise ValueError(
                f'pile-up correction needs more pulses than photons per pixel; got '
                f'{_n_sync_px:,} pulses/px vs {_max_px:,.0f} photons in the brightest pixel. '
                f'Check the N_sync reported by the reader.')
    _n_sync_model = _n_sync_px if pileup_in_model else None
    tvb_on = bool(fit_tvb) and tvb_profile is not None
    if _tail:
        taus_fixed, _, t0_px, _, _ = unpack_tail_params(
            global_popt, n_exp, fit_t0, fit_bg, fit_tvb, t0_fixed=t0_fixed)
        shift = sigma = tamp = 0.0
        ttau = 1.0
        irf_fixed = None
        irf_fft = None
        if fit_idx is None:
            fit_idx = np.arange(int(round(t0_px / tcspc_res)), n_bins, dtype=int)
    else:
        t0_px = 0.0
        idx = 2 * n_exp
        shift = global_popt[idx]; idx += 1
        sigma = global_popt[idx] if fit_sigma else 0.0
        if fit_sigma: idx += 1
        if fit_bg: idx += 1
        tamp = global_popt[idx]     if has_tail else 0.0
        ttau = global_popt[idx + 1] if has_tail else 1.0
        taus_fixed = global_popt[:n_exp]
        irf_fixed = build_full_irf(irf_prompt, shift, sigma, tamp, ttau, n_bins)
        irf_fft = np.fft.fft(irf_fixed)
    fit_idx = np.arange(n_bins) if fit_idx is None else np.asarray(fit_idx, dtype=int)
    _windowed = len(fit_idx) != n_bins
    t_axis = np.arange(n_bins, dtype=float) * tcspc_res
    conv_basis = _basis_rows(taus_fixed, t_axis, tcspc_res, n_bins, _tail,
                             irf_fft=irf_fft, t0=t0_px)
    A = conv_basis.T
    A_fit = A[fit_idx]
    if _tail and use_gpu is not False:
        print(f'  [per-pixel] tail fit uses the CPU path; the projection has to be '
              f'restricted to bins past t0')
    _gpu_windowed_ok = not (_windowed and n_exp == 1 and tvb_on)
    if _windowed and use_gpu is not False and not _tail and n_exp == 1 and tvb_on:
        print(f'  [per-pixel] fit window with a time-varying background is not supported '
              f'on the GPU for one-exponential fits, using the CPU path')
    if use_gpu is not False and not _tail and _gpu_windowed_ok:
        _backend = gpu_backend if gpu_backend is not None else (
            None if _gpu_backend_cache is _GPU_BACKEND_UNSET else _gpu_backend_cache
        )
        if _backend is not None:
            if not free_tau or n_exp == 1:
                if n_exp == 1:
                    _lo = (tau_min_ns if tau_min_ns is not None
                           else max(taus_fixed[0] * 1e9 / 20.0, 0.05)) * 1e-9
                    _hi = (tau_max_ns if tau_max_ns is not None
                           else min(taus_fixed[0] * 1e9 * 20.0, 45.0)) * 1e-9
                    _N_GRID = tau_grid_points()
                    _tau_grid = np.logspace(np.log10(_lo), np.log10(_hi), _N_GRID)
                    _basis_grid = _basis_rows(_tau_grid, t_axis, tcspc_res, n_bins,
                                              _tail, irf_fft=irf_fft, t0=t0_px)
                    _bb_grid = np.maximum((_basis_grid ** 2).sum(axis=1), 1e-20)
                    return _backend.batch_grid_scan_1exp(
                        stack, _basis_grid, _bb_grid, _tau_grid,
                        min_photons, correct_pileup, _n_sync_px,
                        progress_callback,
                        tvb_profile=tvb_profile if tvb_on else None,
                        fit_tvb=tvb_on,
                        fit_idx=fit_idx if _windowed else None,
                    )
                else:
                    return _backend.batch_fixed_tau(
                        stack, A, taus_fixed,
                        min_photons, correct_pileup, _n_sync_px,
                        progress_callback,
                        tvb_profile=tvb_profile if tvb_on else None,
                        fit_tvb=tvb_on,
                        fit_idx=fit_idx if _windowed else None,
                    )
            else:
                _tau_min_s = (tau_min_ns if tau_min_ns is not None
                              else taus_fixed.min() * 1e9 * 0.1) * 1e-9
                _tau_max_s = (tau_max_ns if tau_max_ns is not None
                              else taus_fixed.max() * 1e9 * 10.0) * 1e-9
                return _backend.batch_free_tau_fit(
                    stack, irf_fixed, tcspc_res,
                    taus_fixed, _tau_min_s, _tau_max_s,
                    n_exp, min_photons, correct_pileup, _n_sync_px,
                    tvb_profile=tvb_profile if tvb_on else None,
                    fit_tvb=tvb_on,
                    fit_idx=fit_idx if _windowed else None,
                    n_sync_model=_n_sync_px if pileup_in_model else None,
                )
    if tvb_on:
        B_cpu = np.asarray(tvb_profile, dtype=float)
        A_aug = np.column_stack([A, B_cpu, np.ones(n_bins)])
    maps = dict(
        intensity = stack.sum(axis=2),
        tau_mean_int = np.full((ny, nx), np.nan),
        tau_mean_amp = np.full((ny, nx), np.nan),
        chi2_r = np.full((ny, nx), np.nan),
        calibrated_chi2_r = np.full((ny, nx), np.nan),
    )
    for i in range(n_exp):
        maps[f"alpha_{i+1}"] = np.full((ny, nx), np.nan)
        maps[f"frac_{i+1}"] = np.full((ny, nx), np.nan)
        maps[f"tau_{i+1}"] = (np.full((ny, nx), np.nan)
                              if n_exp == 1 or free_tau
                              else np.full((ny, nx), taus_fixed[i] * 1e9))
        maps[f"a{i+1}"] = maps[f"alpha_{i+1}"]
    if tvb_on:
        maps['tvb_scale'] = np.full((ny, nx), np.nan)
    fitted = skipped = 0
    t0 = time.time()
    if n_exp == 1:
        _lo = (tau_min_ns if tau_min_ns is not None
               else max(taus_fixed[0] * 1e9 / 20.0, 0.05)) * 1e-9
        _hi = (tau_max_ns if tau_max_ns is not None
               else min(taus_fixed[0] * 1e9 * 20.0, 45.0)) * 1e-9
        _N_GRID = tau_grid_points()
        tau_grid = np.logspace(np.log10(_lo), np.log10(_hi), _N_GRID)
        basis_grid = _basis_rows(tau_grid, t_axis, tcspc_res, n_bins, _tail,
                                 irf_fft=irf_fft, t0=t0_px)
        basis_fit = basis_grid[:, fit_idx]
        bb_grid = np.maximum((basis_fit ** 2).sum(axis=1), 1e-20)
        if tvb_on:
            _tvb_prof_f = np.asarray(tvb_profile, dtype=float)[fit_idx]
            _tvb_U = np.column_stack([_tvb_prof_f, np.ones(len(fit_idx))])
            _tvb_Up = np.linalg.pinv(_tvb_U)
            _basis_perp = basis_fit - (basis_fit @ _tvb_Up.T) @ _tvb_U.T
            _bb_perp = np.maximum((_basis_perp ** 2).sum(axis=1), 1e-20)
        for yi in tqdm(range(ny), desc='  Per-pixel rows', disable=True):
            if progress_callback is not None:
                progress_callback(yi, ny)
            decay_row = stack[yi, :, :].astype(float)  
            ph_counts = decay_row.sum(axis=1)          
            valid_xi = np.where(ph_counts >= min_photons)[0]
            skipped += nx - len(valid_xi)
            if len(valid_xi) == 0:
                continue
            if tvb_on:
                dvf = decay_row[valid_xi].astype(float)
                if correct_pileup and _n_sync_px > 0:
                    dvf = np.array([coates_pileup_correction(dvf[k], _n_sync_px)
                                    for k in range(len(valid_xi))])
                dvf_f = dvf[:, fit_idx]
                d_perp = dvf_f - (dvf_f @ _tvb_Up.T) @ _tvb_U.T
                bd_p = d_perp @ _basis_perp.T
                d_sq = (d_perp ** 2).sum(axis=1)
                costs_p = d_sq[:, np.newaxis] - np.maximum(bd_p, 0.0) ** 2 / _bb_perp
                best_g = np.argmin(costs_p, axis=1)
                amp_v = np.maximum(bd_p[np.arange(len(valid_xi)), best_g] / _bb_perp[best_g], 0.0)
                resid_after = dvf_f - amp_v[:, np.newaxis] * basis_fit[best_g]
                vz = resid_after @ _tvb_Up.T
                tvb_v = np.maximum(vz[:, 0], 0.0)
                bg_z = vz[:, 1]
                tau_v = tau_grid[best_g]
                for k, xi in enumerate(valid_xi):
                    if amp_v[k] <= 0:
                        skipped += 1
                        continue
                    tau_ns = float(tau_v[k] * 1e9)
                    maps['tau_1'][yi, xi] = tau_ns
                    maps['tau_mean_amp'][yi, xi] = tau_ns
                    maps['tau_mean_int'][yi, xi] = tau_ns
                    maps['alpha_1'][yi, xi] = float(amp_v[k])
                    maps['frac_1'][yi, xi] = 1.0
                    maps['tvb_scale'][yi, xi] = float(tvb_v[k])
                    model_px = amp_v[k] * basis_fit[best_g[k]] + tvb_v[k] * _tvb_prof_f + bg_z[k]
                    resid = dvf_f[k] - model_px
                    chi2_px = float(np.sum(resid ** 2 / np.maximum(model_px, 1.0)))
                    maps['chi2_r'][yi, xi] = chi2_px / max(len(fit_idx) - 2, 1)
                    maps['calibrated_chi2_r'][yi, xi] = calibrated_chi2(dvf_f[k], model_px)
                    fitted += 1
                continue
            dv = decay_row[valid_xi] 
            peak_b_v = np.argmax(dv, axis=1)  
            bg_v = np.array([estimate_bg(dv[k], int(peak_b_v[k]))
                             for k in range(len(valid_xi))])
            dc_v = np.maximum(dv - bg_v[:, np.newaxis], 0.0)  
            if correct_pileup and _n_sync_px > 0:
                dc_v = np.array([
                    coates_pileup_correction(dc_v[k], _n_sync_px)
                    for k in range(len(valid_xi))
                ])
            dc_f = dc_v[:, fit_idx]
            bd = dc_f @ basis_fit.T
            amps_g = np.maximum(bd / bb_grid, 0.0)
            dc_sq = (dc_f ** 2).sum(axis=1)
            costs = dc_sq[:, np.newaxis] - np.maximum(bd, 0.0) ** 2 / bb_grid
            best_g = np.argmin(costs, axis=1)                          
            tau_v = tau_grid[best_g]                                 
            amp_v = amps_g[np.arange(len(valid_xi)), best_g]        
            good = amp_v > 0
            skipped += int((~good).sum())
            for k, xi in enumerate(valid_xi):
                if not good[k]:
                    continue
                tau_ns = float(tau_v[k] * 1e9)
                maps['tau_1'][yi, xi] = tau_ns
                maps['tau_mean_amp'][yi, xi] = tau_ns
                maps['tau_mean_int'][yi, xi] = tau_ns
                maps['alpha_1'][yi, xi] = float(amp_v[k])
                maps['frac_1'][yi, xi] = 1.0
                best_b = basis_fit[best_g[k]]
                model_px = float(amp_v[k]) * best_b + bg_v[k]
                resid = dv[k][fit_idx] - model_px
                chi2_px = float(np.sum(resid ** 2 / np.maximum(model_px, 1.0)))
                maps['chi2_r'][yi, xi] = chi2_px / max(len(fit_idx) - 2, 1)
                maps['calibrated_chi2_r'][yi, xi] = calibrated_chi2(dv[k][fit_idx], model_px)
                fitted += 1
    elif not free_tau:
        for yi in tqdm(range(ny), desc='  Per-pixel rows', disable=True):
            if progress_callback is not None:
                progress_callback(yi, ny)
            for xi in range(nx):
                decay_px = stack[yi, xi, :]
                if decay_px.sum() < min_photons:
                    skipped += 1
                    continue
                if tvb_on:
                    dfit = decay_px.astype(float)
                    if correct_pileup and _n_sync_px > 0:
                        dfit = coates_pileup_correction(dfit, _n_sync_px)
                    coeffs_px, _ = nnls(A_aug[fit_idx], dfit[fit_idx])
                    amps_px = coeffs_px[:n_exp]
                    tvb_px = coeffs_px[n_exp]
                    bg_px = coeffs_px[n_exp + 1]
                    model_px = A_fit @ amps_px + tvb_px * B_cpu[fit_idx] + bg_px
                else:
                    bg_px = estimate_bg(decay_px, int(np.argmax(decay_px)))
                    data_corr = np.maximum(decay_px - bg_px, 0.0)
                    if correct_pileup and _n_sync_px > 0:
                        data_corr = coates_pileup_correction(data_corr, _n_sync_px)
                    amps_px, _ = nnls(A_fit, data_corr[fit_idx])
                    model_px = A_fit @ amps_px + bg_px
                resid = decay_px[fit_idx] - model_px
                chi2_px = float(np.sum(resid**2 / np.maximum(model_px, 1.0)))
                dof_px = max(len(fit_idx) - n_exp, 1)
                amp_sum = amps_px.sum()
                if amp_sum <= 0:
                    skipped += 1
                    continue
                fracs_px = amps_px / amp_sum
                taus_ns = taus_fixed * 1e9
                tau_amp = float(np.dot(fracs_px, taus_ns))
                denom = np.dot(amps_px, taus_ns)
                tau_int = float(np.dot(amps_px, taus_ns**2) / denom) \
                           if denom > 0 else np.nan
                maps['tau_mean_int'][yi, xi] = tau_int
                maps['tau_mean_amp'][yi, xi] = tau_amp
                maps['chi2_r'][yi, xi] = chi2_px / dof_px
                maps['calibrated_chi2_r'][yi, xi] = calibrated_chi2(
                    decay_px[fit_idx], model_px)
                for i in range(n_exp):
                    maps[f"alpha_{i+1}"][yi, xi] = amps_px[i]
                    maps[f"frac_{i+1}"][yi, xi] = fracs_px[i]
                if tvb_on:
                    maps['tvb_scale'][yi, xi] = tvb_px
                fitted += 1
    else:  
        tau_min_s = (tau_min_ns if tau_min_ns is not None
                     else taus_fixed.min() * 1e9 * 0.1) * 1e-9
        tau_max_s = (tau_max_ns if tau_max_ns is not None
                     else taus_fixed.max() * 1e9 * 10.0) * 1e-9
        amp_hi = float(stack.max()) * 10.0
        lo_px = np.array([tau_min_s / TAU_FIT_UNIT_S] * n_exp + [0.0] * n_exp)
        hi_px = np.array([tau_max_s / TAU_FIT_UNIT_S] * n_exp + [amp_hi] * n_exp)
        p0_px = np.concatenate([taus_fixed / TAU_FIT_UNIT_S,
                                np.full(n_exp, float(stack.max()) / n_exp)])
        if tvb_on:
            tvb_hi = float(stack.sum(axis=2).max())
            lo_px = np.concatenate([lo_px, [0.0]])
            hi_px = np.concatenate([hi_px, [tvb_hi]])
            p0_px = np.concatenate([p0_px, [float(stack.max())]])
        for yi in tqdm(range(ny), desc='  Per-pixel rows (free-τ)', disable=True):
            if progress_callback is not None:
                progress_callback(yi, ny)
            for xi in range(nx):
                decay_px = stack[yi, xi, :].astype(float)
                if decay_px.sum() < min_photons:
                    skipped += 1
                    continue
                fit_px = decay_px
                if correct_pileup and _n_sync_px > 0:
                    fit_px = coates_pileup_correction(decay_px, _n_sync_px)
                bg_px = estimate_bg(fit_px, int(np.argmax(fit_px)))
                def _make_full(p_px):
                    taus_p = p_px[:n_exp] * TAU_FIT_UNIT_S
                    amps_p = p_px[n_exp:2 * n_exp]
                    if _tail:
                        full = list(taus_p) + list(amps_p)
                        if tvb_on:
                            full.append(p_px[2 * n_exp])
                        return full
                    full = list(taus_p) + list(amps_p) + [shift]
                    if fit_sigma:
                        full.append(sigma)
                    if tvb_on:
                        full.append(p_px[2 * n_exp])
                    elif fit_bg:
                        full.append(bg_px)
                    if has_tail:
                        full.extend([tamp, ttau])
                    return full
                def _eval_model(full_p, _bg):
                    if _tail:
                        return tail_model(
                            full_p, tcspc_res, n_bins, n_exp,
                            0.0 if tvb_on else _bg, False,
                            fit_t0=False, t0_fixed=t0_px,
                            tvb_profile=tvb_profile if tvb_on else None,
                            fit_tvb=tvb_on)
                    if tvb_on:
                        return reconvolution_model(
                            full_p, tcspc_res, n_bins, irf_prompt,
                            n_exp, 0.0, has_tail, False, fit_sigma,
                            tvb_profile=tvb_profile, fit_tvb=True,
                            n_sync=_n_sync_model)
                    return reconvolution_model(
                        full_p, tcspc_res, n_bins, irf_prompt,
                        n_exp, _bg, has_tail, False, fit_sigma,
                        n_sync=_n_sync_model)
                w_px = np.sqrt(np.maximum(decay_px, 1.0))
                def _resid(p_px, _decay=fit_px, _bg=bg_px, _w=w_px):
                    model_vals = _eval_model(np.array(_make_full(p_px)), _bg)
                    return (model_vals[fit_idx] - _decay[fit_idx]) / _w[fit_idx]
                try:
                    res = least_squares(
                        _resid, p0_px,
                        bounds=(lo_px, hi_px),
                        method='trf', max_nfev=500,
                        ftol=1e-8, xtol=1e-8, gtol=1e-8)
                    p_sol = res.x
                except Exception:
                    skipped += 1
                    continue
                taus_sol = p_sol[:n_exp] * TAU_FIT_UNIT_S
                amps_sol = p_sol[n_exp:2 * n_exp]
                amp_sum = amps_sol.sum()
                if amp_sum <= 0:
                    skipped += 1
                    continue
                order = np.argsort(taus_sol)
                taus_sol = taus_sol[order]
                amps_sol = amps_sol[order]
                fracs_px = amps_sol / amp_sum
                taus_ns = taus_sol * 1e9
                tau_amp = float(np.dot(fracs_px, taus_ns))
                denom = np.dot(amps_sol, taus_ns)
                tau_int = float(np.dot(amps_sol, taus_ns**2) / denom) \
                           if denom > 0 else np.nan
                full_sol = np.array(_make_full(p_sol))
                model_sol = _eval_model(full_sol, bg_px)
                resid_sol = fit_px[fit_idx] - model_sol[fit_idx]
                chi2_px = float(np.sum(resid_sol**2 / np.maximum(model_sol[fit_idx], 1.0)))
                dof_px = max(len(fit_idx) - 2 * n_exp, 1)
                maps['tau_mean_int'][yi, xi] = tau_int
                maps['tau_mean_amp'][yi, xi] = tau_amp
                maps['chi2_r'][yi, xi] = chi2_px / dof_px
                maps['calibrated_chi2_r'][yi, xi] = calibrated_chi2(
                    fit_px[fit_idx], model_sol[fit_idx])
                for i in range(n_exp):
                    maps[f"tau_{i+1}"][yi, xi] = taus_ns[i]
                    maps[f"alpha_{i+1}"][yi, xi] = amps_sol[i]
                    maps[f"frac_{i+1}"][yi, xi] = fracs_px[i]
                if tvb_on:
                    maps['tvb_scale'][yi, xi] = float(p_sol[2 * n_exp])
                fitted += 1
    elapsed = time.time() - t0
    return maps

def fit_summed_dist(decay, tcspc_res, n_bins, irf_prompt,
                    n_components, dist_type,
                    fit_bg, fit_sigma,
                    tau_min_ns, tau_max_ns,
                    optimizer='de', n_restarts=8,
                    de_popsize=15, de_maxiter=1000,
                    workers=-1, polish=True,
                    cost_function='poisson',
                    sigma_max=3.0,
                    irf_shift_bins=2,
                    tvb_profile=None, fit_tvb=False,
                    fit_start_ns=None, fit_end_ns=None, exclude_ns=None,
                    n_sync=None):
    tau_min = tau_min_ns * 1e-9
    tau_max = tau_max_ns * 1e-9
    if cost_function not in ('chi2', 'poisson'):
        raise ValueError(f"Unknown cost_function: {cost_function!r}")
    decay_work = decay.astype(float)
    if decay_work.max() == 0:
        raise ValueError('Decay has zero maximum - cannot fit.')
    peak_bin = int(np.argmax(decay_work))
    bg_init = estimate_bg(decay_work, peak_bin)
    bg_fixed = bg_init if not fit_bg else 0.0
    fit_end = find_fit_end(decay_work, peak_bin, tau_max, tcspc_res, n_bins)
    standard_fit_end = int(round(44.9455 / (tcspc_res * 1e9)))
    fit_end = min(fit_end, standard_fit_end)
    fit_start = find_fit_start(decay_work, irf_prompt, tcspc_res)
    fit_start = max(0, min(fit_start, fit_end - 10))
    if fit_end_ns is not None:
        fit_end = min(n_bins, bins_from_ns(fit_end_ns, tcspc_res))
    if fit_start_ns is not None:
        fit_start = max(0, min(bins_from_ns(fit_start_ns, tcspc_res), fit_end - 2))
    exclude_bins = [(bins_from_ns(lo, tcspc_res), bins_from_ns(hi, tcspc_res))
                    for lo, hi in (exclude_ns or [])]
    fit_idx = build_fit_idx(fit_start, fit_end, n_bins, exclude_bins)
    bg_upper = max(bg_init * 2.0, bg_init + 10.0)
    if fit_tvb and tvb_profile is None:
        raise ValueError('fit_tvb=True requires a tvb_profile.')
    tvb_init = float(bg_init * n_bins) if fit_tvb else 0.0
    tvb_upper = float(decay_work.sum()) if fit_tvb else None
    if fit_tvb:
        print(f"  TVB: free scale on measured profile, init={tvb_init:.1f}, upper={tvb_upper:.1f}")
    print(f"  Cost function: {cost_function}")
    print(f"  bg initial guess = {bg_init:.3f} cts/bin, upper bound = {bg_upper:.3f} "
          f"({'free param' if fit_bg else 'fixed'})")
    lo, hi = _build_bounds_dist(
        n_components, tau_min, tau_max, decay_work.max(),
        fit_bg, fit_sigma, bg_init=bg_init, bg_upper=bg_upper,
        sigma_max=sigma_max, irf_shift_bins=irf_shift_bins,
        fit_tvb=fit_tvb, tvb_init=tvb_init, tvb_upper=tvb_upper)
    bounds = list(zip(lo, hi))
    if cost_function == 'chi2':
        weights = np.sqrt(np.maximum(decay_work[fit_idx], 1.0))
        def residuals(params):
            model_vals = dist_reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_components, dist_type, bg_fixed, fit_bg, fit_sigma,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb)
            return (model_vals[fit_idx] - decay_work[fit_idx]) / weights
    else:
        def residuals(params):
            model_vals = dist_reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_components, dist_type, bg_fixed, fit_bg, fit_sigma,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb)
            n = decay_work[fit_idx]
            m = np.maximum(model_vals[fit_idx], 1e-10)
            dev = m - n
            pos = n > 0
            dev[pos] += n[pos] * np.log(n[pos] / m[pos])
            dev = np.maximum(dev, 0.0)
            r = np.sqrt(2.0 * dev)
            r[m < n] *= -1
            return r
    if optimizer == 'lm_multistart':
        rng = np.random.default_rng(42)
        best_res = None
        best_cost = np.inf
        for i in range(n_restarts + 1):
            if i == 0:
                p0 = _pack_p0_dist(n_components, tau_min, tau_max,
                                   float(decay_work.max()), fit_bg, fit_sigma, bg_init,
                                   fit_tvb=fit_tvb, tvb_init=tvb_init)
            else:
                tau_ov = np.sort(np.exp(rng.uniform(
                    np.log(tau_min * 1.001), np.log(tau_max * 0.999), n_components)))
                p0 = _pack_p0_dist(n_components, tau_min, tau_max,
                                        float(decay_work.max()), fit_bg, fit_sigma, bg_init,
                                        fit_tvb=fit_tvb, tvb_init=tvb_init)
                p0[:n_components] = tau_ov
            try:
                res = least_squares(residuals, p0, bounds=(lo, hi), method='trf',
                                    max_nfev=50000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
            except Exception as exc:
                print(f"    Restart {i:2d}: failed ({exc})")
                continue
            if res.cost < best_cost:
                best_cost = res.cost
                best_res = res
                print(f"    Restart {i:2d}: cost={res.cost:.4e}  ← best")
            else:
                print(f"    Restart {i:2d}: cost={res.cost:.4e}")
        if best_res is None:
            raise RuntimeError('All restarts failed.')
        popt_work = best_res.x
        message = best_res.message
    elif optimizer == 'de':
        print(f"  Differential evolution: popsize={de_popsize}, maxiter={de_maxiter}, workers={workers}")
        n = n_components
        bounds_log = list(bounds)
        for i in range(n):
            lo_t, hi_t = bounds[i]
            bounds_log[i] = (np.log10(lo_t), np.log10(hi_t))
        for i in range(n, 2 * n):
            lo_w, hi_w = bounds[i]
            bounds_log[i] = (np.log10(lo_w), np.log10(hi_w))
        if cost_function == 'poisson':
            cost_fn = _DECostDistPoissonLogParam(
                tcspc_res, n_bins, irf_prompt, n_components, dist_type,
                bg_fixed, fit_bg, fit_sigma, fit_idx, decay_work,
                tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
        else:
            cost_fn = _DECostDistLogParam(
                tcspc_res, n_bins, irf_prompt, n_components, dist_type,
                bg_fixed, fit_bg, fit_sigma, fit_idx, decay_work,
                weights, tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
        de_res = differential_evolution(
            cost_fn, bounds=bounds_log,
            maxiter=de_maxiter, popsize=de_popsize,
            workers=workers, seed=42,
            updating='deferred' if workers != 1 else 'immediate',
            init='sobol', disp=False)
        popt_work = de_res.x.copy()
        popt_work[:n] = 10.0 ** popt_work[:n]
        popt_work[n:2*n] = 10.0 ** popt_work[n:2*n]
        message = f"DE success={de_res.success}, fun={de_res.fun:.4e}"
        if polish:
            print('  Running final LM polish...')
            eps = 1e-10
            popt_work = np.clip(popt_work, np.asarray(lo) + eps, np.asarray(hi) - eps)
            try:
                pol = least_squares(residuals, popt_work, bounds=(lo, hi), method='trf',
                                    max_nfev=5000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
                popt_work = pol.x
                message += f"; polished cost={pol.cost:.4e}"
            except ValueError as e:
                print(f"  Warning: LM polish failed ({e}) - using DE result")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    summary = _make_summary_dist(
        popt_work, decay, tcspc_res, n_bins, irf_prompt,
        n_components, dist_type, bg_fixed, fit_bg, fit_sigma,
        fit_idx, message, tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync)
    return popt_work, summary

def _make_summary_dist(popt, decay, tcspc_res, n_bins, irf_prompt,
                       n_components, dist_type, bg_fixed, fit_bg, fit_sigma,
                       fit_idx, message=None,
                       tvb_profile=None, fit_tvb=False, n_sync=None):
    fit_start = int(fit_idx[0])
    fit_end = int(fit_idx[-1]) + 1
    tau_centers = popt[:n_components]
    widths = popt[n_components:2 * n_components]
    amps = popt[2 * n_components:3 * n_components]
    idx = 3 * n_components
    shift = popt[idx]; idx += 1
    sigma = popt[idx] if fit_sigma else 0.0
    if fit_sigma:
        idx += 1
    if fit_bg:
        bg_fit = popt[idx]; idx += 1
    else:
        bg_fit = bg_fixed
    if fit_tvb:
        tvb_scale = popt[idx]; idx += 1
    else:
        tvb_scale = 0.0
    model = dist_reconvolution_model(popt, tcspc_res, n_bins, irf_prompt,
                                       n_components, dist_type, bg_fixed, fit_bg, fit_sigma,
                                       tvb_profile=tvb_profile, fit_tvb=fit_tvb)
    d_win = decay[fit_idx].astype(float)
    m_win = model[fit_idx]
    sigma_w = np.sqrt(np.maximum(d_win, 1.0))
    chi2 = float(np.sum(((d_win - m_win) / sigma_w) ** 2))
    dof = max(len(fit_idx) - len(popt), 1)
    rchi2 = chi2 / dof
    sigma_p = np.sqrt(np.maximum(m_win, 1.0))
    chi2_p = float(np.sum(((d_win - m_win) / sigma_p) ** 2))
    rchi2_p = chi2_p / dof
    calibrated_chi2_p = calibrated_chi2(d_win, m_win)
    peak_bin_loc = int(fit_idx[np.argmax(decay[fit_idx])])
    tail_start = peak_bin_loc + max(1, int(0.05 * (fit_end - peak_bin_loc)))
    tail_idx = fit_idx[fit_idx >= tail_start]
    d_tail = decay[tail_idx].astype(float)
    m_tail = model[tail_idx]
    sw_tail = np.sqrt(np.maximum(d_tail, 1.0))
    chi2_tail = float(np.sum(((d_tail - m_tail) / sw_tail) ** 2))
    dof_tail = max(len(tail_idx) - len(popt), 1)
    rchi2_tail = chi2_tail / dof_tail
    sp_tail = np.sqrt(np.maximum(m_tail, 1.0))
    chi2_tail_p = float(np.sum(((d_tail - m_tail) / sp_tail) ** 2))
    rchi2_tail_p = chi2_tail_p / dof_tail
    calibrated_chi2_tail_p = calibrated_chi2(d_tail, m_tail)
    resid = (decay - model) / np.sqrt(np.maximum(model, 1.0))
    from ..FLIM.models import _alpha_gaussian, _alpha_lorentzian, _N_QUAD
    alpha_fn = _alpha_gaussian if dist_type == 'gaussian' else _alpha_lorentzian
    tau_num_amp = 0.0;  tau_den_amp = 0.0
    tau_num_int = 0.0;  tau_den_int = 0.0
    for i in range(n_components):
        tau_c = tau_centers[i];  w = widths[i];  a = max(amps[i], 0.0)
        spread = 4.0 * w if dist_type == 'gaussian' else 8.0 * max(w / 2.0, 1e-15)
        tau_lo = max(tau_c - spread, 1e-12)
        tau_hi = max(tau_c + spread, tau_lo + 1e-12)
        tau_grid = np.linspace(tau_lo, tau_hi, _N_QUAD)
        alpha = alpha_fn(tau_grid, tau_c, w)
        alpha = alpha / max(alpha.sum(), 1e-30)
        tau_num_amp += a * float(np.dot(alpha, tau_grid))
        tau_den_amp += a
        tau_num_int += a * float(np.dot(alpha, tau_grid ** 2))
        tau_den_int += a * float(np.dot(alpha, tau_grid))
    tau_mean_amp_ns = (tau_num_amp / max(tau_den_amp, 1e-30)) * 1e9
    tau_mean_int_ns = (tau_num_int / max(tau_den_int, 1e-30)) * 1e9
    amp_sum = max(amps.sum(), 1e-30)
    fractions = amps / amp_sum
    if dist_type == 'gaussian':
        fwhms_ns = widths * 2.3548 * 1e9
    else:
        fwhms_ns = widths * 1e9 
    above = np.where(irf_prompt >= irf_prompt.max() / 2)[0]
    fwhm_pr = (above[-1] - above[0]) if len(above) > 1 else 1
    fwhm_eff = np.sqrt(fwhm_pr ** 2 + (2.3548 * sigma) ** 2) * tcspc_res * 1e9
    return dict(
        dist_type = dist_type,
        n_components = n_components,
        tcspc_res = tcspc_res,
        tau_centers_ns = tau_centers * 1e9,
        widths_ns = widths * 1e9,
        fwhms_ns = fwhms_ns,
        amps = amps,
        fractions = fractions,
        bg_fit = bg_fit,
        tvb_scale = tvb_scale,
        tau_mean_amp_ns = tau_mean_amp_ns,
        tau_mean_int_ns = tau_mean_int_ns,
        chi2 = chi2,
        reduced_chi2 = rchi2,
        reduced_chi2_tail = rchi2_tail,
        chi2_pearson = chi2_p,
        reduced_chi2_pearson = rchi2_p,
        reduced_chi2_tail_pearson = rchi2_tail_p,
        calibrated_chi2_pearson = calibrated_chi2_p,
        calibrated_chi2_tail_pearson = calibrated_chi2_tail_p,
        tail_start_bin = tail_start,
        dof = dof,
        fit_window_bins = (fit_start, fit_end),
        fit_window_ns = (fit_start * tcspc_res * 1e9, fit_end * tcspc_res * 1e9),
        fit_idx = fit_idx,
        irf_shift_bins = shift,
        irf_sigma_bins = sigma,
        irf_fwhm_eff_ns = fwhm_eff,
        model = model,
        residuals = resid,
        optimizer_msg = message,
    )


def fit_per_pixel_dist(stack, tcspc_res, n_bins, irf_prompt,
                       global_popt, n_components, dist_type,
                       fit_bg=True, fit_sigma=False,
                       min_photons=MIN_PHOTONS_PERPIX,
                       tau_min_ns=None, tau_max_ns=None,
                       n_tau_grid=50, n_width_grid=30,
                       progress_callback=None,
                       use_gpu='auto',
                       gpu_backend=None,
                       tvb_profile=None, fit_tvb=False,
                       fit_idx=None) -> dict:
    from ..GPU._base import fit_window

    ny, nx, _ = stack.shape
    window = fit_window(fit_idx, n_bins)
    fit_idx = np.arange(n_bins) if window is None else window
    n_fit = len(fit_idx)
    tvb_on = bool(fit_tvb) and tvb_profile is not None
    if tvb_on and n_components != 1:
        print('  Per-pixel distribution TVB only supported for unimodal (1 component); ignoring TVB here.')
        tvb_on = False
    idx = 3 * n_components
    shift = global_popt[idx]; idx += 1
    sigma = global_popt[idx] if fit_sigma else 0.0
    if fit_sigma:
        idx += 1
    tau_centers_g = global_popt[:n_components]
    widths_g = global_popt[n_components:2 * n_components]
    irf_fixed = build_full_irf(irf_prompt, shift, sigma, 0.0, 1.0, n_bins)
    tau_lo = max(tau_centers_g.min() / 5.0, 1e-12)   if tau_min_ns is None else tau_min_ns * 1e-9
    tau_hi = tau_centers_g.max() * 5.0               if tau_max_ns is None else tau_max_ns * 1e-9
    w_lo = widths_g.min() / 5.0
    w_hi = widths_g.max() * 5.0
    tau_grid = np.logspace(np.log10(max(tau_lo, 1e-12)), np.log10(tau_hi), n_tau_grid)
    width_grid = np.logspace(np.log10(max(w_lo, 1e-12)), np.log10(w_hi), n_width_grid)
    print(f"  Building distribution basis grid ({n_tau_grid}×{n_width_grid})...")
    basis, param_pairs = build_dist_basis_grid(
        tcspc_res, n_bins, irf_fixed, tau_grid, width_grid, dist_type)
    basis_fit = basis[:, fit_idx]
    bb_grid = np.maximum((basis_fit.astype(np.float64) ** 2).sum(axis=1), 1e-20).astype(np.float32)
    maps = dict(
        intensity = stack.sum(axis=2),
        tau_mean_amp = np.full((ny, nx), np.nan),
        tau_mean_int = np.full((ny, nx), np.nan),
        chi2_r = np.full((ny, nx), np.nan),
        calibrated_chi2_r = np.full((ny, nx), np.nan),
    )
    for i in range(n_components):
        maps[f"tau_center_{i+1}"] = np.full((ny, nx), np.nan)
        maps[f"width_{i+1}"] = np.full((ny, nx), np.nan)
        maps[f"alpha_{i+1}"] = np.full((ny, nx), np.nan)
        maps[f"frac_{i+1}"] = np.full((ny, nx), np.nan)
    if tvb_on:
        maps['tvb_scale'] = np.full((ny, nx), np.nan)
    if n_components == 1:
        backend = None
        if use_gpu is not False:
            backend = gpu_backend if gpu_backend is not None else (
                None if _gpu_backend_cache is _GPU_BACKEND_UNSET else _gpu_backend_cache
            )
        if backend is not None and stack.nbytes > _GPU_MAX_DIST_STACK_BYTES:
            print(f'  [per-pixel] {stack.nbytes/1e9:.1f} GB cube exceeds GPU limit '
                  f'({_GPU_MAX_DIST_STACK_BYTES/1e9:.1f} GB); the distribution scan is not blocked, using the CPU path')
            backend = None
        if backend is not None:
            return backend.batch_dist_scan_unimodal(
                stack, basis_fit, bb_grid, param_pairs,
                irf_fixed, tcspc_res, n_bins, dist_type,
                min_photons, progress_callback,
                tvb_profile=tvb_profile if tvb_on else None,
                fit_tvb=tvb_on, fit_idx=fit_idx)
        flat = stack.reshape(ny * nx, n_bins).astype(np.float32)
        ph_counts = flat.sum(axis=1)
        valid_idx = np.where(ph_counts >= min_photons)[0]
        if tvb_on and valid_idx.size > 0:
            tvb_fit = np.asarray(tvb_profile, dtype=float)[fit_idx]
            _U = np.column_stack([tvb_fit, np.ones(n_fit)])
            _Up = np.linalg.pinv(_U)
            _bp = basis_fit.astype(np.float64) - (basis_fit.astype(np.float64) @ _Up.T) @ _U.T
            _bbp = np.maximum((_bp ** 2).sum(axis=1), 1e-20)
            d_valid = flat[valid_idx][:, fit_idx].astype(np.float64)
            d_perp = d_valid - (d_valid @ _Up.T) @ _U.T
            bd = d_perp @ _bp.T
            costs = (d_perp ** 2).sum(axis=1)[:, None] - np.maximum(bd, 0.0) ** 2 / _bbp[None, :]
            best_g = np.argmin(costs, axis=1)
            amp_v = np.maximum(bd[np.arange(len(valid_idx)), best_g] / _bbp[best_g], 0.0)
            basis_best = basis_fit[best_g].astype(np.float64)
            resid_after = d_valid - amp_v[:, None] * basis_best
            vz = resid_after @ _Up.T
            tvb_v = np.maximum(vz[:, 0], 0.0)
            bg_z = vz[:, 1]
            tau_v = param_pairs[best_g, 0]
            w_v = param_pairs[best_g, 1]
            model_v = amp_v[:, None] * basis_best + tvb_v[:, None] * tvb_fit[None, :] + bg_z[:, None]
            chi2_v = ((d_valid - model_v) ** 2 / np.maximum(model_v, 1.0)).sum(axis=1)
            chi2_v /= distribution_dof(n_fit, 1, True)
            chi2_cal_v = calibrated_chi2(d_valid, model_v, axis=1)
            for k, fi in enumerate(valid_idx):
                if amp_v[k] <= 0:
                    continue
                yy, xx = divmod(int(fi), nx)
                maps['tau_center_1'][yy, xx] = tau_v[k] * 1e9
                maps['width_1'][yy, xx] = w_v[k] * 1e9
                maps['alpha_1'][yy, xx] = amp_v[k]
                maps['frac_1'][yy, xx] = 1.0
                maps['tau_mean_amp'][yy, xx] = tau_v[k] * 1e9
                maps['tau_mean_int'][yy, xx] = (tau_v[k] + w_v[k] ** 2 / max(tau_v[k], 1e-15)) * 1e9
                maps['chi2_r'][yy, xx] = chi2_v[k]
                maps['calibrated_chi2_r'][yy, xx] = chi2_cal_v[k]
                maps['tvb_scale'][yy, xx] = tvb_v[k]
            return maps
        t0 = time.time()
        for row_i in range(ny):
            if progress_callback is not None:
                progress_callback(row_i, ny)
            xi_range = range(nx)
            for xi in xi_range:
                flat_idx = row_i * nx + xi
                if ph_counts[flat_idx] < min_photons:
                    continue
                d = flat[flat_idx].astype(np.float64)
                d_fit = d[fit_idx]
                bg = estimate_bg(d, int(np.argmax(d)))
                dc = np.maximum(d_fit - bg, 0.0)
                bd = basis_fit.astype(np.float64) @ dc
                amps_g = np.maximum(bd / bb_grid.astype(np.float64), 0.0)
                costs = (dc ** 2).sum() - np.maximum(bd, 0.0) ** 2 / bb_grid.astype(np.float64)
                best = int(np.argmin(costs))
                tau_c_px = float(param_pairs[best, 0])
                w_px = float(param_pairs[best, 1])
                amp_px = float(amps_g[best])
                tau_amp_ns = tau_c_px * 1e9
                tau_int_ns = (tau_c_px + (w_px ** 2) / max(tau_c_px, 1e-15)) * 1e9
                model_px = amp_px * basis_fit[best].astype(np.float64) + bg
                resid_px = d_fit - model_px
                chi2_px = float(np.sum(resid_px ** 2 / np.maximum(model_px, 1.0)))
                maps['tau_center_1'][row_i, xi] = tau_c_px * 1e9
                maps['width_1'][row_i, xi] = w_px * 1e9
                maps['alpha_1'][row_i, xi] = amp_px
                maps['frac_1'][row_i, xi] = 1.0
                maps['tau_mean_amp'][row_i, xi] = tau_amp_ns
                maps['tau_mean_int'][row_i, xi] = tau_int_ns
                maps['chi2_r'][row_i, xi] = chi2_px / distribution_dof(n_fit, 1, False)
                maps['calibrated_chi2_r'][row_i, xi] = calibrated_chi2(d_fit, model_px)
    else:
        from ..FLIM.fit_tools import estimate_bg as _ebg
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        flat = stack.reshape(ny * nx, n_bins).astype(np.float32)
        ph_counts = flat.sum(axis=1)
        tau_lo_s = max(tau_centers_g.min() / 5.0, 1e-12)
        tau_hi_s = tau_centers_g.max() * 5.0
        w_lo_s = widths_g.min() / 5.0
        w_hi_s = widths_g.max() * 5.0
        fit_stack_max = float(stack[..., fit_idx].max())
        amp_hi = fit_stack_max * 10.0
        lo_px = np.array([tau_lo_s] * n_components + [w_lo_s] * n_components + [0.0] * n_components)
        hi_px = np.array([tau_hi_s] * n_components + [w_hi_s] * n_components + [amp_hi] * n_components)
        p0_px = np.concatenate([tau_centers_g, widths_g,
                                np.full(n_components, fit_stack_max / n_components)])
        def _fit_pixel_dist(flat_idx):
            d = flat[flat_idx].astype(np.float64)
            d_fit = d[fit_idx]
            bg = _ebg(d, int(np.argmax(d)))
            wt_fit = np.sqrt(np.maximum(d_fit, 1.0))
            def _resid(p):
                full_p = np.concatenate([p, [shift]])
                if fit_sigma:
                    full_p = np.concatenate([full_p, [sigma]])
                if fit_bg:
                    full_p = np.concatenate([full_p, [bg]])
                m = dist_reconvolution_model(
                    full_p, tcspc_res, n_bins, irf_prompt,
                    n_components, dist_type, bg, False, False)
                return (m[fit_idx] - d_fit) / wt_fit
            try:
                res = least_squares(_resid, p0_px, bounds=(lo_px, hi_px),
                                    method='trf', max_nfev=500,
                                    ftol=1e-8, xtol=1e-8, gtol=1e-8)
                return res.x
            except Exception:
                return None
        n_workers = min(ny * nx, max(1, multiprocessing.cpu_count()))
        valid_flat = [i for i in range(ny * nx) if ph_counts[i] >= min_photons]
        n_valid = len(valid_flat)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_fit_pixel_dist, i) for i in valid_flat]
            solutions = []
            for k, f in enumerate(futures):
                solutions.append(f.result())
                if progress_callback is not None and k % max(1, n_valid // 200) == 0:
                    progress_callback(k, n_valid)
        for sol, flat_idx in zip(solutions, valid_flat):
            if sol is None:
                continue
            yi, xi = divmod(flat_idx, nx)
            tau_cs = sol[:n_components]
            ws = sol[n_components:2 * n_components]
            amp_s = sol[2 * n_components:]
            amp_sum = max(amp_s.sum(), 1e-30)
            fracs = amp_s / amp_sum
            tau_amp_ns = float(np.dot(fracs, tau_cs)) * 1e9
            tau_int_ns = float(np.dot(amp_s, tau_cs ** 2) / max(np.dot(amp_s, tau_cs), 1e-30)) * 1e9
            maps['tau_mean_amp'][yi, xi] = tau_amp_ns
            maps['tau_mean_int'][yi, xi] = tau_int_ns
            d = flat[flat_idx].astype(np.float64)
            d_fit = d[fit_idx]
            bg = _ebg(d, int(np.argmax(d)))
            full_p = np.concatenate([sol, [shift]])
            if fit_sigma:
                full_p = np.concatenate([full_p, [sigma]])
            if fit_bg:
                full_p = np.concatenate([full_p, [bg]])
            model_px = dist_reconvolution_model(
                full_p, tcspc_res, n_bins, irf_prompt,
                n_components, dist_type, bg, False, False)
            model_fit = model_px[fit_idx]
            resid_fit = d_fit - model_fit
            chi2_px = float(np.sum(resid_fit ** 2 / np.maximum(model_fit, 1.0)))
            maps['chi2_r'][yi, xi] = chi2_px / distribution_dof(
                n_fit, n_components, False)
            maps['calibrated_chi2_r'][yi, xi] = calibrated_chi2(
                d_fit, model_fit)
            for i in range(n_components):
                maps[f"tau_center_{i+1}"][yi, xi] = tau_cs[i] * 1e9
                maps[f"width_{i+1}"][yi, xi] = ws[i] * 1e9
                maps[f"alpha_{i+1}"][yi, xi] = amp_s[i]
                maps[f"frac_{i+1}"][yi, xi] = fracs[i]
    return maps
