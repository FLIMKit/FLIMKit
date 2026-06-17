import time
import os
import numpy as np
from tqdm import tqdm

tqdm.disable = True

from scipy.optimize import least_squares, differential_evolution, nnls
from scipy.stats.distributions import chi2 as chi2_dist
from ..FLIM.irf_tools import build_full_irf
from ..FLIM.fit_tools import estimate_bg, find_fit_end, _build_bounds, _pack_p0, coates_pileup_correction
from ..FLIM.models import (reconvolution_model, _DECost, _DECostLogTau,
                           _DECostPoisson, _DECostPoissonLogTau,
                           dist_reconvolution_model, build_dist_basis_grid,
                           _DECostDist, _DECostDistLogParam,
                           _DECostDistPoisson, _DECostDistPoissonLogParam)
from ..FLIM.fit_tools import (_build_bounds_dist, _pack_p0_dist)
from ..configs import MIN_PHOTONS_PERPIX

_GPU_BACKEND_UNSET = object()
_gpu_backend_cache = _GPU_BACKEND_UNSET


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
               irf_shift_bins=2) -> tuple[np.ndarray, dict]:

    warmup_gpu_backend()
    tau_min  = tau_min_ns * 1e-9
    tau_max  = tau_max_ns * 1e-9

    if cost_function not in ('chi2', 'poisson'):
        raise ValueError(f"Unknown cost_function: {cost_function!r}")
    decay_work = decay.astype(float)    
    if decay_work.max() == 0:
        raise ValueError('Decay has zero maximum - cannot fit.')
    scale = 1.0

    peak_bin = int(np.argmax(decay_work))
    bg_init  = estimate_bg(decay_work, peak_bin)
    bg_fixed = bg_init if not fit_bg else 0.0

    fit_end   = find_fit_end(decay_work, peak_bin, tau_max, tcspc_res, n_bins)
    fit_start = 1

    standard_fit_end = int(round(44.9455 / (tcspc_res * 1e9)))
    fit_end = min(fit_end, standard_fit_end)

    bg_upper = max(bg_init * 2.0, bg_init + 10.0)

    print(f"  Cost function: {cost_function}")
    print(f"  bg initial guess = {bg_init:.3f} cts/bin"
          f", upper bound = {bg_upper:.3f} "
          f"({'free param' if fit_bg else 'fixed'})")
    print(f"  σ broadening: {'free param (σ≤' + f'{sigma_max:.1f})' if fit_sigma else 'fixed at 0'}")
    print(f"  Fit window: bins {fit_start}-{fit_end} "
          f"({fit_start*tcspc_res*1e9:.2f}-{fit_end*tcspc_res*1e9:.2f} ns), "
          f"{fit_end-fit_start} bins")

    lo, hi  = _build_bounds(n_exp, tau_min, tau_max, decay_work.max(),
                             has_tail, fit_bg, fit_sigma,
                             bg_init=bg_init, bg_upper=bg_upper,
                             sigma_max=sigma_max, irf_shift_bins=irf_shift_bins)
    bounds  = list(zip(lo, hi))

    if cost_function == 'chi2':
        weights = np.sqrt(np.maximum(decay_work[fit_start:fit_end], 1.0))

        def residuals(params):
            model_vals = reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
            return (model_vals[fit_start:fit_end]
                    - decay_work[fit_start:fit_end]) / weights

    else:
        def residuals(params):
            model_vals = reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
            n = decay_work[fit_start:fit_end]
            m = np.maximum(model_vals[fit_start:fit_end], 1e-10)
            dev = m - n
            pos = n > 0
            dev[pos] += n[pos] * np.log(n[pos] / m[pos])
            dev = np.maximum(dev, 0.0)       
            r = np.sqrt(2.0 * dev)
            r[m < n] *= -1                   
            return r

    if optimizer == 'lm_multistart':
        rng       = np.random.default_rng(42)
        best_res  = None
        best_cost = np.inf

        for i in range(n_restarts + 1):
            tau_ov = None if i == 0 else np.sort(
                np.exp(rng.uniform(np.log(tau_min*1.001),
                                   np.log(tau_max*0.999), n_exp)))
            p0 = _pack_p0(n_exp, tau_min, tau_max, float(decay_work.max()),
                          has_tail, fit_bg, fit_sigma, bg_init,
                          tau_override=tau_ov)
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
                best_res  = res
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}  ← best")
            else:
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}")

        if best_res is None:
            raise RuntimeError('All restarts failed.')
        popt_work = best_res.x
        message   = best_res.message

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
                fit_start, fit_end, decay_work)
        else:
            cost_fn = _DECostLogTau(
                tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                has_tail, fit_bg, fit_sigma,
                fit_start, fit_end, decay_work, weights)

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
                message  += f"; polished cost={pol.cost:.4e}"
            except ValueError as e:
                print(f"  Warning: LM polish failed ({e}) - using DE result")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")

    popt_original = popt_work.copy()

    summary = _make_summary(popt_original, decay, tcspc_res, n_bins, irf_prompt,
                            n_exp, bg_fixed,
                            has_tail, fit_bg, fit_sigma,
                            fit_start, fit_end, message)
    return popt_original, summary


def _make_summary(popt, decay, tcspc_res, n_bins, irf_prompt,
                  n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                  fit_start, fit_end, message=None) -> dict:

    taus  = popt[:n_exp]
    amps  = popt[n_exp:2*n_exp]
    order = np.argsort(-taus)
    taus = taus[order]
    amps = amps[order]
    idx   = 2 * n_exp

    shift = popt[idx]; idx += 1

    if fit_sigma:
        sigma = popt[idx]; idx += 1
    else:
        sigma = 0.0

    if fit_bg:
        bg_fit = popt[idx]; idx += 1
    else:
        bg_fit = bg_fixed

    if has_tail:
        tail_amp = popt[idx]
        tail_tau = popt[idx + 1]
    else:
        tail_amp = tail_tau = 0.0
    model   = reconvolution_model(popt, tcspc_res, n_bins, irf_prompt,
                                   n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
    d_win   = decay[fit_start:fit_end].astype(float)
    m_win   = model[fit_start:fit_end]
    sigma_w = np.sqrt(np.maximum(d_win, 1.0))
    chi2    = float(np.sum(((d_win - m_win) / sigma_w)**2))
    dof     = max((fit_end - fit_start) - len(popt), 1)
    rchi2   = chi2 / dof
    p_val   = float(1 - chi2_dist.cdf(chi2, df=dof))
    resid   = (decay - model) / np.sqrt(np.maximum(model, 1.0))
    sigma_p  = np.sqrt(np.maximum(m_win, 1.0))
    chi2_p   = float(np.sum(((d_win - m_win) / sigma_p)**2))
    rchi2_p  = chi2_p / dof
    peak_bin_loc = int(np.argmax(decay[fit_start:fit_end])) + fit_start
    tail_start   = peak_bin_loc + max(1, int(0.05 * (fit_end - peak_bin_loc)))
    d_tail  = decay[tail_start:fit_end].astype(float)
    m_tail  = model[tail_start:fit_end]
    sw_tail = np.sqrt(np.maximum(d_tail, 1.0))
    chi2_tail  = float(np.sum(((d_tail - m_tail) / sw_tail)**2))
    dof_tail   = max((fit_end - tail_start) - len(popt), 1)
    rchi2_tail = chi2_tail / dof_tail
    sp_tail       = np.sqrt(np.maximum(m_tail, 1.0))
    chi2_tail_p   = float(np.sum(((d_tail - m_tail) / sp_tail)**2))
    rchi2_tail_p  = chi2_tail_p / dof_tail
    amp_sum    = amps.sum() if amps.sum() > 0 else 1.0
    fracs      = amps / amp_sum
    tau_amp    = float(np.dot(fracs, taus))
    tau_int    = float(np.dot(amps, taus**2) / np.dot(amps, taus))
    above    = np.where(irf_prompt >= irf_prompt.max() / 2)[0]
    fwhm_pr  = (above[-1] - above[0]) if len(above) > 1 else 1
    fwhm_eff = np.sqrt(fwhm_pr**2 + (2.3548 * sigma)**2) * tcspc_res * 1e9

    return dict(
        tcspc_res        = tcspc_res,
        taus_ns          = taus * 1e9,
        amps             = amps,
        fractions        = fracs,
        bg_fit           = bg_fit,
        tau_mean_amp_ns  = tau_amp * 1e9,
        tau_mean_int_ns  = tau_int * 1e9,
        chi2             = chi2,
        reduced_chi2     = rchi2,
        reduced_chi2_tail= rchi2_tail,
        chi2_pearson           = chi2_p,
        reduced_chi2_pearson   = rchi2_p,
        reduced_chi2_tail_pearson = rchi2_tail_p,
        tail_start_bin   = tail_start,
        p_val            = p_val,
        dof              = dof,
        fit_window_bins  = (fit_start, fit_end),
        fit_window_ns    = (fit_start*tcspc_res*1e9, fit_end*tcspc_res*1e9),
        irf_shift_bins   = shift,
        irf_sigma_bins   = sigma,
        irf_fwhm_eff_ns  = fwhm_eff,
        tail_amp         = tail_amp,
        tail_tau_ns      = tail_tau * tcspc_res * 1e9,
        model            = model,
        residuals        = resid,
        optimizer_msg    = message,
    )


def fit_per_pixel(stack, tcspc_res, n_bins, irf_prompt,
                  has_tail, fit_bg, fit_sigma,
                  global_popt, n_exp,
                  min_photons=MIN_PHOTONS_PERPIX,
                  tau_min_ns=None, tau_max_ns=None,
                  correct_pileup=False, n_sync=0,
                  progress_callback=None,
                  free_tau=False,
                  use_gpu='auto',
                  gpu_backend=None) -> dict:
    ny, nx, _ = stack.shape
    _n_sync_px = int(n_sync / max(ny * nx, 1)) if correct_pileup and n_sync > 0 else 0

    idx   = 2 * n_exp
    shift = global_popt[idx]; idx += 1
    sigma = global_popt[idx] if fit_sigma else 0.0
    if fit_sigma: idx += 1
    if fit_bg: idx += 1
    tamp  = global_popt[idx]     if has_tail else 0.0
    ttau  = global_popt[idx + 1] if has_tail else 1.0
    taus_fixed = global_popt[:n_exp]

    irf_fixed  = build_full_irf(irf_prompt, shift, sigma, tamp, ttau, n_bins)
    t_axis     = np.arange(n_bins, dtype=float) * tcspc_res
    basis      = np.stack([np.exp(-t_axis / max(tau, 1e-15)) for tau in taus_fixed])
    irf_fft    = np.fft.fft(irf_fixed)
    conv_basis = np.array([
        np.real(np.fft.ifft(np.fft.fft(b) * irf_fft)) for b in basis
    ]) 
    A = conv_basis.T   

    if use_gpu is not False:
        _backend = gpu_backend if gpu_backend is not None else (
            None if _gpu_backend_cache is _GPU_BACKEND_UNSET else _gpu_backend_cache
        )

        if _backend is not None:
            if not free_tau:
                if n_exp == 1:
                    _lo = (tau_min_ns if tau_min_ns is not None
                           else max(taus_fixed[0] * 1e9 / 20.0, 0.05)) * 1e-9
                    _hi = (tau_max_ns if tau_max_ns is not None
                           else min(taus_fixed[0] * 1e9 * 20.0, 45.0)) * 1e-9
                    _N_GRID = 200
                    _tau_grid   = np.logspace(np.log10(_lo), np.log10(_hi), _N_GRID)
                    _irf_fft_g  = np.fft.fft(irf_fixed)
                    _basis_grid = np.array([
                        np.real(np.fft.ifft(
                            np.fft.fft(np.exp(-t_axis / max(tau, 1e-15))) * _irf_fft_g))
                        for tau in _tau_grid
                    ])
                    _bb_grid = np.maximum((_basis_grid ** 2).sum(axis=1), 1e-20)
                    return _backend.batch_grid_scan_1exp(
                        stack, _basis_grid, _bb_grid, _tau_grid,
                        min_photons, correct_pileup, _n_sync_px,
                        progress_callback,
                    )
                else:
                    return _backend.batch_fixed_tau(
                        stack, A, taus_fixed,
                        min_photons, correct_pileup, _n_sync_px,
                        progress_callback,
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
                )

    maps = dict(
        intensity    = stack.sum(axis=2),
        tau_mean_int = np.full((ny, nx), np.nan),
        tau_mean_amp = np.full((ny, nx), np.nan),
        chi2_r       = np.full((ny, nx), np.nan),
    )
    for i in range(n_exp):
        maps[f"alpha_{i+1}"] = np.full((ny, nx), np.nan)
        maps[f"frac_{i+1}"]  = np.full((ny, nx), np.nan)
        maps[f"tau_{i+1}"] = (np.full((ny, nx), np.nan)
                              if n_exp == 1 or free_tau
                              else np.full((ny, nx), taus_fixed[i] * 1e9))
        maps[f"a{i+1}"]      = maps[f"alpha_{i+1}"]
    fitted = skipped = 0
    t0 = time.time()

    if n_exp == 1:
        _lo = (tau_min_ns if tau_min_ns is not None
               else max(taus_fixed[0] * 1e9 / 20.0, 0.05)) * 1e-9
        _hi = (tau_max_ns if tau_max_ns is not None
               else min(taus_fixed[0] * 1e9 * 20.0, 45.0)) * 1e-9

        _N_GRID = 200
        tau_grid = np.logspace(np.log10(_lo), np.log10(_hi), _N_GRID)

        _irf_fft_g = np.fft.fft(irf_fixed)
        basis_grid = np.array([
            np.real(np.fft.ifft(
                np.fft.fft(np.exp(-t_axis / max(tau, 1e-15))) * _irf_fft_g))
            for tau in tau_grid
        ])  
        bb_grid = np.maximum((basis_grid ** 2).sum(axis=1), 1e-20)

        for yi in tqdm(range(ny), desc='  Per-pixel rows', disable=True):
            if progress_callback is not None:
                progress_callback(yi, ny)

            decay_row = stack[yi, :, :].astype(float)  
            ph_counts = decay_row.sum(axis=1)          
            valid_xi  = np.where(ph_counts >= min_photons)[0]
            skipped  += nx - len(valid_xi)

            if len(valid_xi) == 0:
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

            bd     = dc_v @ basis_grid.T             
            amps_g = np.maximum(bd / bb_grid, 0.0)  
            dc_sq  = (dc_v ** 2).sum(axis=1)        
            # cost[i,j] = ||dc_v[i]||² - max(bd[i,j],0)² / ||basis[j]||²
            # Minimising cost ↔ maximising the projection bd²/||b||²
            costs  = dc_sq[:, np.newaxis] - np.maximum(bd, 0.0) ** 2 / bb_grid
            best_g = np.argmin(costs, axis=1)                          
            tau_v  = tau_grid[best_g]                                 
            amp_v  = amps_g[np.arange(len(valid_xi)), best_g]        

            good = amp_v > 0
            skipped += int((~good).sum())

            for k, xi in enumerate(valid_xi):
                if not good[k]:
                    continue
                tau_ns = float(tau_v[k] * 1e9)
                maps['tau_1'][yi, xi]        = tau_ns
                maps['tau_mean_amp'][yi, xi] = tau_ns
                maps['tau_mean_int'][yi, xi] = tau_ns
                maps['alpha_1'][yi, xi]      = float(amp_v[k])
                maps['frac_1'][yi, xi]       = 1.0
                best_b   = basis_grid[best_g[k]]
                model_px = float(amp_v[k]) * best_b + bg_v[k]
                resid    = dv[k] - model_px
                chi2_px  = float(np.sum(resid ** 2 / np.maximum(model_px, 1.0)))
                maps['chi2_r'][yi, xi] = chi2_px / max(n_bins - 2, 1)
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

                bg_px   = estimate_bg(decay_px, int(np.argmax(decay_px)))
                data_corr = np.maximum(decay_px - bg_px, 0.0)
                if correct_pileup and _n_sync_px > 0:
                    data_corr = coates_pileup_correction(data_corr, _n_sync_px)
                amps_px, _ = nnls(A, data_corr)

                model_px = A @ amps_px + bg_px
                resid    = decay_px - model_px
                chi2_px  = float(np.sum(resid**2 / np.maximum(model_px, 1.0)))
                dof_px   = max(n_bins - n_exp, 1)

                amp_sum = amps_px.sum()
                if amp_sum <= 0:
                    skipped += 1
                    continue

                fracs_px = amps_px / amp_sum
                taus_ns  = taus_fixed * 1e9
                tau_amp  = float(np.dot(fracs_px, taus_ns))
                denom    = np.dot(amps_px, taus_ns)
                tau_int  = float(np.dot(amps_px, taus_ns**2) / denom) \
                           if denom > 0 else np.nan

                maps['tau_mean_int'][yi, xi] = tau_int
                maps['tau_mean_amp'][yi, xi] = tau_amp
                maps['chi2_r'][yi, xi]       = chi2_px / dof_px
                for i in range(n_exp):
                    maps[f"alpha_{i+1}"][yi, xi] = amps_px[i]
                    maps[f"frac_{i+1}"][yi, xi]  = fracs_px[i]
                fitted += 1

    else:  
        tau_min_s = (tau_min_ns if tau_min_ns is not None
                     else taus_fixed.min() * 1e9 * 0.1) * 1e-9
        tau_max_s = (tau_max_ns if tau_max_ns is not None
                     else taus_fixed.max() * 1e9 * 10.0) * 1e-9
        amp_hi    = float(stack.max()) * 10.0
        lo_px = np.array([tau_min_s] * n_exp + [0.0] * n_exp)
        hi_px = np.array([tau_max_s] * n_exp + [amp_hi]   * n_exp)
        p0_px = np.concatenate([taus_fixed, np.full(n_exp, float(stack.max()) / n_exp)])

        for yi in tqdm(range(ny), desc='  Per-pixel rows (free-τ)', disable=True):
            if progress_callback is not None:
                progress_callback(yi, ny)
            for xi in range(nx):
                decay_px = stack[yi, xi, :].astype(float)
                if decay_px.sum() < min_photons:
                    skipped += 1
                    continue

                bg_px     = estimate_bg(decay_px, int(np.argmax(decay_px)))
                data_corr = np.maximum(decay_px - bg_px, 0.0)
                if correct_pileup and _n_sync_px > 0:
                    data_corr = coates_pileup_correction(data_corr, _n_sync_px)

                def _make_full(p_px):
                    taus_p = p_px[:n_exp]
                    amps_p = p_px[n_exp:2 * n_exp]
                    full   = list(taus_p) + list(amps_p) + [shift]
                    if fit_sigma:
                        full.append(sigma)
                    if fit_bg:
                        full.append(bg_px)
                    if has_tail:
                        full.extend([tamp, ttau])
                    return full

                w_px = np.sqrt(np.maximum(decay_px, 1.0))

                def _resid(p_px, _decay=decay_px, _bg=bg_px, _w=w_px):
                    full_p = np.array(_make_full(p_px))
                    model_vals = reconvolution_model(
                        full_p, tcspc_res, n_bins, irf_prompt,
                        n_exp, _bg, has_tail, False, fit_sigma)
                    return (model_vals - _decay) / _w
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
                taus_sol = p_sol[:n_exp]
                amps_sol = p_sol[n_exp:2 * n_exp]
                amp_sum  = amps_sol.sum()
                if amp_sum <= 0:
                    skipped += 1
                    continue
                order    = np.argsort(taus_sol)
                taus_sol = taus_sol[order]
                amps_sol = amps_sol[order]
                fracs_px = amps_sol / amp_sum
                taus_ns  = taus_sol * 1e9
                tau_amp  = float(np.dot(fracs_px, taus_ns))
                denom    = np.dot(amps_sol, taus_ns)
                tau_int  = float(np.dot(amps_sol, taus_ns**2) / denom) \
                           if denom > 0 else np.nan
                full_sol   = np.array(_make_full(p_sol))
                model_sol  = reconvolution_model(
                    full_sol, tcspc_res, n_bins, irf_prompt,
                    n_exp, bg_px, has_tail, False, fit_sigma)
                resid_sol  = decay_px - model_sol
                chi2_px    = float(np.sum(resid_sol**2 / np.maximum(model_sol, 1.0)))
                dof_px     = max(n_bins - 2 * n_exp, 1)
                maps['tau_mean_int'][yi, xi] = tau_int
                maps['tau_mean_amp'][yi, xi] = tau_amp
                maps['chi2_r'][yi, xi] = chi2_px / dof_px
                for i in range(n_exp):
                    maps[f"tau_{i+1}"][yi, xi]   = taus_ns[i]
                    maps[f"alpha_{i+1}"][yi, xi] = amps_sol[i]
                    maps[f"frac_{i+1}"][yi, xi]  = fracs_px[i]
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
                    irf_shift_bins=2):

    tau_min = tau_min_ns * 1e-9
    tau_max = tau_max_ns * 1e-9
    if cost_function not in ('chi2', 'poisson'):
        raise ValueError(f"Unknown cost_function: {cost_function!r}")
    decay_work = decay.astype(float)
    if decay_work.max() == 0:
        raise ValueError('Decay has zero maximum - cannot fit.')
    peak_bin  = int(np.argmax(decay_work))
    bg_init   = estimate_bg(decay_work, peak_bin)
    bg_fixed  = bg_init if not fit_bg else 0.0
    fit_end   = find_fit_end(decay_work, peak_bin, tau_max, tcspc_res, n_bins)
    fit_start = 1
    standard_fit_end = int(round(44.9455 / (tcspc_res * 1e9)))
    fit_end = min(fit_end, standard_fit_end)
    bg_upper = max(bg_init * 2.0, bg_init + 10.0)
    print(f"  Cost function: {cost_function}")
    print(f"  bg initial guess = {bg_init:.3f} cts/bin, upper bound = {bg_upper:.3f} "
          f"({'free param' if fit_bg else 'fixed'})")
    lo, hi = _build_bounds_dist(
        n_components, tau_min, tau_max, decay_work.max(),
        fit_bg, fit_sigma, bg_init=bg_init, bg_upper=bg_upper,
        sigma_max=sigma_max, irf_shift_bins=irf_shift_bins)
    bounds = list(zip(lo, hi))
    if cost_function == 'chi2':
        weights = np.sqrt(np.maximum(decay_work[fit_start:fit_end], 1.0))
        def residuals(params):
            model_vals = dist_reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_components, dist_type, bg_fixed, fit_bg, fit_sigma)
            return (model_vals[fit_start:fit_end] - decay_work[fit_start:fit_end]) / weights
    else:
        def residuals(params):
            model_vals = dist_reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_components, dist_type, bg_fixed, fit_bg, fit_sigma)
            n = decay_work[fit_start:fit_end]
            m = np.maximum(model_vals[fit_start:fit_end], 1e-10)
            dev = m - n
            pos = n > 0
            dev[pos] += n[pos] * np.log(n[pos] / m[pos])
            dev = np.maximum(dev, 0.0)
            r   = np.sqrt(2.0 * dev)
            r[m < n] *= -1
            return r
    if optimizer == 'lm_multistart':
        rng       = np.random.default_rng(42)
        best_res  = None
        best_cost = np.inf
        for i in range(n_restarts + 1):
            if i == 0:
                p0 = _pack_p0_dist(n_components, tau_min, tau_max,
                                   float(decay_work.max()), fit_bg, fit_sigma, bg_init)
            else:
                tau_ov  = np.sort(np.exp(rng.uniform(
                    np.log(tau_min * 1.001), np.log(tau_max * 0.999), n_components)))
                p0      = _pack_p0_dist(n_components, tau_min, tau_max,
                                        float(decay_work.max()), fit_bg, fit_sigma, bg_init)
                p0[:n_components] = tau_ov
            try:
                res = least_squares(residuals, p0, bounds=(lo, hi), method='trf',
                                    max_nfev=50000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
            except Exception as exc:
                print(f"    Restart {i:2d}: failed ({exc})")
                continue
            if res.cost < best_cost:
                best_cost = res.cost
                best_res  = res
                print(f"    Restart {i:2d}: cost={res.cost:.4e}  ← best")
            else:
                print(f"    Restart {i:2d}: cost={res.cost:.4e}")
        if best_res is None:
            raise RuntimeError('All restarts failed.')
        popt_work = best_res.x
        message   = best_res.message
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
                bg_fixed, fit_bg, fit_sigma, fit_start, fit_end, decay_work)
        else:
            cost_fn = _DECostDistLogParam(
                tcspc_res, n_bins, irf_prompt, n_components, dist_type,
                bg_fixed, fit_bg, fit_sigma, fit_start, fit_end, decay_work,
                weights)
        de_res = differential_evolution(
            cost_fn, bounds=bounds_log,
            maxiter=de_maxiter, popsize=de_popsize,
            workers=workers, seed=42,
            updating='deferred' if workers != 1 else 'immediate',
            init='sobol', disp=False)
        popt_work = de_res.x.copy()
        popt_work[:n]     = 10.0 ** popt_work[:n]
        popt_work[n:2*n]  = 10.0 ** popt_work[n:2*n]
        message = f"DE success={de_res.success}, fun={de_res.fun:.4e}"
        if polish:
            print('  Running final LM polish...')
            eps = 1e-10
            popt_work = np.clip(popt_work, np.asarray(lo) + eps, np.asarray(hi) - eps)
            try:
                pol = least_squares(residuals, popt_work, bounds=(lo, hi), method='trf',
                                    max_nfev=5000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
                popt_work = pol.x
                message  += f"; polished cost={pol.cost:.4e}"
            except ValueError as e:
                print(f"  Warning: LM polish failed ({e}) - using DE result")
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    summary = _make_summary_dist(
        popt_work, decay, tcspc_res, n_bins, irf_prompt,
        n_components, dist_type, bg_fixed, fit_bg, fit_sigma,
        fit_start, fit_end, message)
    return popt_work, summary

def _make_summary_dist(popt, decay, tcspc_res, n_bins, irf_prompt,
                       n_components, dist_type, bg_fixed, fit_bg, fit_sigma,
                       fit_start, fit_end, message=None):
    tau_centers = popt[:n_components]
    widths      = popt[n_components:2 * n_components]
    amps        = popt[2 * n_components:3 * n_components]
    idx   = 3 * n_components
    shift = popt[idx]; idx += 1
    sigma  = popt[idx] if fit_sigma else 0.0
    if fit_sigma:
        idx += 1
    bg_fit = popt[idx] if fit_bg else bg_fixed
    model  = dist_reconvolution_model(popt, tcspc_res, n_bins, irf_prompt,
                                       n_components, dist_type, bg_fixed, fit_bg, fit_sigma)
    d_win  = decay[fit_start:fit_end].astype(float)
    m_win  = model[fit_start:fit_end]
    sigma_w = np.sqrt(np.maximum(d_win, 1.0))
    chi2    = float(np.sum(((d_win - m_win) / sigma_w) ** 2))
    dof     = max((fit_end - fit_start) - len(popt), 1)
    rchi2   = chi2 / dof
    sigma_p  = np.sqrt(np.maximum(m_win, 1.0))
    chi2_p   = float(np.sum(((d_win - m_win) / sigma_p) ** 2))
    rchi2_p  = chi2_p / dof
    peak_bin_loc = int(np.argmax(decay[fit_start:fit_end])) + fit_start
    tail_start   = peak_bin_loc + max(1, int(0.05 * (fit_end - peak_bin_loc)))
    d_tail  = decay[tail_start:fit_end].astype(float)
    m_tail  = model[tail_start:fit_end]
    sw_tail = np.sqrt(np.maximum(d_tail, 1.0))
    chi2_tail   = float(np.sum(((d_tail - m_tail) / sw_tail) ** 2))
    dof_tail    = max((fit_end - tail_start) - len(popt), 1)
    rchi2_tail  = chi2_tail / dof_tail
    sp_tail     = np.sqrt(np.maximum(m_tail, 1.0))
    chi2_tail_p  = float(np.sum(((d_tail - m_tail) / sp_tail) ** 2))
    rchi2_tail_p = chi2_tail_p / dof_tail
    resid = (decay - model) / np.sqrt(np.maximum(model, 1.0))
    from ..FLIM.models import _alpha_gaussian, _alpha_lorentzian, _N_QUAD
    alpha_fn = _alpha_gaussian if dist_type == 'gaussian' else _alpha_lorentzian
    tau_num_amp = 0.0;  tau_den_amp = 0.0
    tau_num_int = 0.0;  tau_den_int = 0.0
    for i in range(n_components):
        tau_c = tau_centers[i];  w = widths[i];  a = max(amps[i], 0.0)
        spread   = 4.0 * w if dist_type == 'gaussian' else 8.0 * max(w / 2.0, 1e-15)
        tau_lo   = max(tau_c - spread, 1e-12)
        tau_hi   = max(tau_c + spread, tau_lo + 1e-12)
        tau_grid = np.linspace(tau_lo, tau_hi, _N_QUAD)
        alpha    = alpha_fn(tau_grid, tau_c, w)
        alpha    = alpha / max(alpha.sum(), 1e-30)
        tau_num_amp += a * float(np.dot(alpha, tau_grid))
        tau_den_amp += a
        tau_num_int += a * float(np.dot(alpha, tau_grid ** 2))
        tau_den_int += a * float(np.dot(alpha, tau_grid))
    tau_mean_amp_ns = (tau_num_amp / max(tau_den_amp, 1e-30)) * 1e9
    tau_mean_int_ns = (tau_num_int / max(tau_den_int, 1e-30)) * 1e9
    amp_sum   = max(amps.sum(), 1e-30)
    fractions = amps / amp_sum
    if dist_type == 'gaussian':
        fwhms_ns = widths * 2.3548 * 1e9
    else:
        fwhms_ns = widths * 1e9 
    above    = np.where(irf_prompt >= irf_prompt.max() / 2)[0]
    fwhm_pr  = (above[-1] - above[0]) if len(above) > 1 else 1
    fwhm_eff = np.sqrt(fwhm_pr ** 2 + (2.3548 * sigma) ** 2) * tcspc_res * 1e9
    return dict(
        dist_type           = dist_type,
        n_components        = n_components,
        tcspc_res           = tcspc_res,
        tau_centers_ns      = tau_centers * 1e9,
        widths_ns           = widths * 1e9,
        fwhms_ns            = fwhms_ns,
        amps                = amps,
        fractions           = fractions,
        bg_fit              = bg_fit,
        tau_mean_amp_ns     = tau_mean_amp_ns,
        tau_mean_int_ns     = tau_mean_int_ns,
        chi2                = chi2,
        reduced_chi2        = rchi2,
        reduced_chi2_tail   = rchi2_tail,
        chi2_pearson             = chi2_p,
        reduced_chi2_pearson     = rchi2_p,
        reduced_chi2_tail_pearson = rchi2_tail_p,
        tail_start_bin      = tail_start,
        dof                 = dof,
        fit_window_bins     = (fit_start, fit_end),
        fit_window_ns       = (fit_start * tcspc_res * 1e9, fit_end * tcspc_res * 1e9),
        irf_shift_bins      = shift,
        irf_sigma_bins      = sigma,
        irf_fwhm_eff_ns     = fwhm_eff,
        model               = model,
        residuals           = resid,
        optimizer_msg       = message,
    )


def fit_per_pixel_dist(stack, tcspc_res, n_bins, irf_prompt,
                       global_popt, n_components, dist_type,
                       fit_bg=True, fit_sigma=False,
                       min_photons=MIN_PHOTONS_PERPIX,
                       tau_min_ns=None, tau_max_ns=None,
                       n_tau_grid=50, n_width_grid=30,
                       progress_callback=None,
                       use_gpu='auto',
                       gpu_backend=None) -> dict:
    ny, nx, _ = stack.shape
    idx   = 3 * n_components
    shift = global_popt[idx]; idx += 1
    sigma = global_popt[idx] if fit_sigma else 0.0
    if fit_sigma:
        idx += 1
    tau_centers_g = global_popt[:n_components]
    widths_g      = global_popt[n_components:2 * n_components]
    irf_fixed = build_full_irf(irf_prompt, shift, sigma, 0.0, 1.0, n_bins)
    tau_lo = max(tau_centers_g.min() / 5.0, 1e-12)   if tau_min_ns is None else tau_min_ns * 1e-9
    tau_hi = tau_centers_g.max() * 5.0               if tau_max_ns is None else tau_max_ns * 1e-9
    w_lo   = widths_g.min() / 5.0
    w_hi   = widths_g.max() * 5.0
    tau_grid   = np.logspace(np.log10(max(tau_lo, 1e-12)), np.log10(tau_hi), n_tau_grid)
    width_grid = np.logspace(np.log10(max(w_lo,   1e-12)), np.log10(w_hi),   n_width_grid)
    print(f"  Building distribution basis grid ({n_tau_grid}×{n_width_grid})...")
    basis, param_pairs = build_dist_basis_grid(
        tcspc_res, n_bins, irf_fixed, tau_grid, width_grid, dist_type)
    bb_grid = np.maximum((basis.astype(np.float64) ** 2).sum(axis=1), 1e-20).astype(np.float32)
    maps = dict(
        intensity    = stack.sum(axis=2),
        tau_mean_amp = np.full((ny, nx), np.nan),
        tau_mean_int = np.full((ny, nx), np.nan),
        chi2_r       = np.full((ny, nx), np.nan),
    )
    for i in range(n_components):
        maps[f"tau_center_{i+1}"] = np.full((ny, nx), np.nan)
        maps[f"width_{i+1}"]      = np.full((ny, nx), np.nan)
        maps[f"alpha_{i+1}"]      = np.full((ny, nx), np.nan)
        maps[f"frac_{i+1}"]       = np.full((ny, nx), np.nan)
    if n_components == 1:
        backend = gpu_backend if gpu_backend is not None else (
            None if _gpu_backend_cache is _GPU_BACKEND_UNSET else _gpu_backend_cache
        )
        if backend is not None:
            return backend.batch_dist_scan_unimodal(
                stack, basis, bb_grid, param_pairs,
                irf_fixed, tcspc_res, n_bins, dist_type,
                min_photons, progress_callback)
        flat       = stack.reshape(ny * nx, n_bins).astype(np.float32)
        ph_counts  = flat.sum(axis=1)
        valid_idx  = np.where(ph_counts >= min_photons)[0]
        t0 = time.time()
        for row_i in range(ny):
            if progress_callback is not None:
                progress_callback(row_i, ny)
            xi_range = range(nx)
            for xi in xi_range:
                flat_idx = row_i * nx + xi
                if ph_counts[flat_idx] < min_photons:
                    continue
                d  = flat[flat_idx].astype(np.float64)
                bg = estimate_bg(d, int(np.argmax(d)))
                dc = np.maximum(d - bg, 0.0)
                bd     = basis.astype(np.float64) @ dc
                amps_g = np.maximum(bd / bb_grid.astype(np.float64), 0.0)
                costs  = (dc ** 2).sum() - np.maximum(bd, 0.0) ** 2 / bb_grid.astype(np.float64)
                best   = int(np.argmin(costs))
                tau_c_px = float(param_pairs[best, 0])
                w_px     = float(param_pairs[best, 1])
                amp_px   = float(amps_g[best])
                tau_amp_ns = tau_c_px * 1e9
                tau_int_ns = (tau_c_px + (w_px ** 2) / max(tau_c_px, 1e-15)) * 1e9
                model_px = amp_px * basis[best].astype(np.float64) + bg
                resid_px = d - model_px
                chi2_px  = float(np.sum(resid_px ** 2 / np.maximum(model_px, 1.0)))
                maps['tau_center_1'][row_i, xi] = tau_c_px * 1e9
                maps['width_1'][row_i, xi]      = w_px * 1e9
                maps['alpha_1'][row_i, xi]      = amp_px
                maps['frac_1'][row_i, xi]       = 1.0
                maps['tau_mean_amp'][row_i, xi] = tau_amp_ns
                maps['tau_mean_int'][row_i, xi] = tau_int_ns
                maps['chi2_r'][row_i, xi]       = chi2_px / max(n_bins - 3, 1)
    else:
        from ..FLIM.fit_tools import estimate_bg as _ebg
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        flat      = stack.reshape(ny * nx, n_bins).astype(np.float32)
        ph_counts = flat.sum(axis=1)
        tau_lo_s = max(tau_centers_g.min() / 5.0, 1e-12)
        tau_hi_s = tau_centers_g.max() * 5.0
        w_lo_s   = widths_g.min() / 5.0
        w_hi_s   = widths_g.max() * 5.0
        amp_hi   = float(stack.max()) * 10.0
        lo_px = np.array([tau_lo_s] * n_components + [w_lo_s] * n_components + [0.0] * n_components)
        hi_px = np.array([tau_hi_s] * n_components + [w_hi_s] * n_components + [amp_hi] * n_components)
        p0_px = np.concatenate([tau_centers_g, widths_g,
                                 np.full(n_components, float(stack.max()) / n_components)])
        
        def _fit_pixel_dist(flat_idx):
            d      = flat[flat_idx].astype(np.float64)
            bg     = _ebg(d, int(np.argmax(d)))
            dc     = np.maximum(d - bg, 0.0)
            wt     = np.sqrt(np.maximum(d, 1.0))

            def _resid(p):
                full_p = np.concatenate([p, [shift]])
                if fit_sigma:
                    full_p = np.concatenate([full_p, [sigma]])
                if fit_bg:
                    full_p = np.concatenate([full_p, [bg]])
                m = dist_reconvolution_model(
                    full_p, tcspc_res, n_bins, irf_prompt,
                    n_components, dist_type, bg, False, False)
                return (m - d) / wt
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
            yi, xi      = divmod(flat_idx, nx)
            tau_cs      = sol[:n_components]
            ws          = sol[n_components:2 * n_components]
            amp_s       = sol[2 * n_components:]
            amp_sum     = max(amp_s.sum(), 1e-30)
            fracs       = amp_s / amp_sum
            tau_amp_ns  = float(np.dot(fracs, tau_cs)) * 1e9
            tau_int_ns  = float(np.dot(amp_s, tau_cs ** 2) / max(np.dot(amp_s, tau_cs), 1e-30)) * 1e9
            maps['tau_mean_amp'][yi, xi] = tau_amp_ns
            maps['tau_mean_int'][yi, xi] = tau_int_ns
            for i in range(n_components):
                maps[f"tau_center_{i+1}"][yi, xi] = tau_cs[i] * 1e9
                maps[f"width_{i+1}"][yi, xi]      = ws[i] * 1e9
                maps[f"alpha_{i+1}"][yi, xi]       = amp_s[i]
                maps[f"frac_{i+1}"][yi, xi]        = fracs[i]
    return maps