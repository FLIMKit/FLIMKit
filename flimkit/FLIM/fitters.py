import time
import os
import numpy as np
from tqdm import tqdm

# Disable tqdm globally – all progress is shown via progress windows instead
tqdm.disable = True

from scipy.optimize import least_squares, differential_evolution, nnls
from scipy.stats.distributions import chi2 as chi2_dist
from ..FLIM.irf_tools import build_full_irf
from ..FLIM.fit_tools import estimate_bg, find_fit_end, _build_bounds, _pack_p0, coates_pileup_correction
from ..FLIM.models import (reconvolution_model, _DECost, _DECostLogTau,
                           _DECostPoisson, _DECostPoissonLogTau)
from ..configs import MIN_PHOTONS_PERPIX

# Module-level GPU backend cache — resolved once on first call.
# Sentinel _GPU_BACKEND_UNSET means "not yet checked".
_GPU_BACKEND_UNSET = object()
_gpu_backend_cache = _GPU_BACKEND_UNSET

def fit_summed(decay, tcspc_res, n_bins, irf_prompt,
               has_tail, fit_bg, fit_sigma,
               n_exp, tau_min_ns, tau_max_ns,
               optimizer="de", n_restarts=8,
               de_popsize=15, de_maxiter=1000,
               workers=-1, polish=True,
               cost_function="poisson",
               sigma_max=3.0,
               irf_shift_bins=2) -> tuple[np.ndarray, dict]:
    """Fit summed FLIM decay via reconvolution.

    Parameters
    ----------
    cost_function : str, optional
        ``'poisson'`` — Poisson deviance / C-statistic (recommended, default).
        ``'chi2'``    — Neyman chi-squared (legacy: weighted least-squares on
                        normalised decay).
    sigma_max : float, optional
        Upper bound for the IRF Gaussian broadening parameter σ (bins).
        Only used when ``fit_sigma=True``.  Default 3.0 (full).
        Set to 0.5 for the balanced “half-sigma” mode.
    """

    tau_min  = tau_min_ns * 1e-9
    tau_max  = tau_max_ns * 1e-9

    if cost_function not in ("chi2", "poisson"):
        raise ValueError(f"Unknown cost_function: {cost_function!r}")
    decay_work = decay.astype(float)    # raw counts (both paths)
    if decay_work.max() == 0:
        raise ValueError("Decay has zero maximum – cannot fit.")
    scale = 1.0

    peak_bin = int(np.argmax(decay_work))
    bg_init  = estimate_bg(decay_work, peak_bin)
    bg_fixed = bg_init if not fit_bg else 0.0

    fit_end   = find_fit_end(decay_work, peak_bin, tau_max, tcspc_res, n_bins)
    fit_start = 1    # match Leica: skip bin 0

    leica_fit_end = int(round(44.9455 / (tcspc_res * 1e9)))
    fit_end = min(fit_end, leica_fit_end)

    bg_upper = max(bg_init * 2.0, bg_init + 10.0)

    print(f"  Cost function: {cost_function}")
    print(f"  bg initial guess = {bg_init:.3f} cts/bin"
          f", upper bound = {bg_upper:.3f} "
          f"({'free param' if fit_bg else 'fixed'})")
    print(f"  σ broadening: {'free param (σ≤' + f'{sigma_max:.1f})' if fit_sigma else 'fixed at 0'}")
    print(f"  Fit window: bins {fit_start}–{fit_end} "
          f"({fit_start*tcspc_res*1e9:.2f}–{fit_end*tcspc_res*1e9:.2f} ns), "
          f"{fit_end-fit_start} bins")

    lo, hi  = _build_bounds(n_exp, tau_min, tau_max, decay_work.max(),
                             has_tail, fit_bg, fit_sigma,
                             bg_init=bg_init, bg_upper=bg_upper,
                             sigma_max=sigma_max, irf_shift_bins=irf_shift_bins)
    bounds  = list(zip(lo, hi))

    # Define residual / cost functions
    if cost_function == "chi2":
        weights = np.sqrt(np.maximum(decay_work[fit_start:fit_end], 1.0))

        def residuals(params):
            model_vals = reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
            return (model_vals[fit_start:fit_end]
                    - decay_work[fit_start:fit_end]) / weights

    else:  # poisson
        def residuals(params):
            """Signed Poisson deviance residuals for LM."""
            model_vals = reconvolution_model(
                params, tcspc_res, n_bins, irf_prompt,
                n_exp, bg_fixed, has_tail, fit_bg, fit_sigma)
            n = decay_work[fit_start:fit_end]
            m = np.maximum(model_vals[fit_start:fit_end], 1e-10)
            dev = m - n
            pos = n > 0
            dev[pos] += n[pos] * np.log(n[pos] / m[pos])
            dev = np.maximum(dev, 0.0)       # numerical guard
            r = np.sqrt(2.0 * dev)
            r[m < n] *= -1                   # sign = data > model
            return r

    if optimizer == "lm_multistart":
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
                res = least_squares(residuals, p0, bounds=(lo, hi), method="trf",
                                    max_nfev=50000,
                                    ftol=1e-13, xtol=1e-13, gtol=1e-13)
            except Exception as exc:
                print(f"    Restart {i:2d}: failed ({exc})")
                continue
            tag = "log-spaced" if i == 0 else "random    "
            if res.cost < best_cost:
                best_cost = res.cost
                best_res  = res
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}  ← best")
            else:
                print(f"    Restart {i:2d} ({tag}): cost={res.cost:.4e}")

        if best_res is None:
            raise RuntimeError("All restarts failed.")
        popt_work = best_res.x
        message   = best_res.message

    elif optimizer == "de":
        print(f"  Differential evolution: popsize={de_popsize}, "
              f"maxiter={de_maxiter}, workers={workers}")

        bounds_log = list(bounds)
        for i in range(n_exp):
            lo_tau, hi_tau = bounds[i]
            bounds_log[i] = (np.log10(lo_tau), np.log10(hi_tau))

        if cost_function == "poisson":
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

        # Convert log₁₀(τ) → τ in the result
        popt_work = de_res.x.copy()
        popt_work[:n_exp] = 10.0 ** popt_work[:n_exp]
        message = f"DE success={de_res.success}, fun={de_res.fun:.4e}"

        if polish:
            print("  Running final LM polish...")
            # Clip strictly inside bounds — DE can land exactly on a bound
            # which causes least_squares to raise "Initial guess outside bounds"
            eps = 1e-10
            popt_work = np.clip(popt_work, np.asarray(lo) + eps, np.asarray(hi) - eps)
            try:
                pol = least_squares(residuals, popt_work, bounds=(lo, hi),
                                    method="trf", max_nfev=5000,
                                    ftol=1e-13, xtol=1e-13, gtol=1e-13)
                popt_work = pol.x
                message  += f"; polished cost={pol.cost:.4e}"
            except ValueError as e:
                print(f"  Warning: LM polish failed ({e}) — using DE result")
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
    """Unpack params in the same order as reconvolution_model."""

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

    # Neyman chi-squared: sum((d - m)² / max(d, 1))
    sigma_w = np.sqrt(np.maximum(d_win, 1.0))
    chi2    = float(np.sum(((d_win - m_win) / sigma_w)**2))
    dof     = max((fit_end - fit_start) - len(popt), 1)
    rchi2   = chi2 / dof
    p_val   = float(1 - chi2_dist.cdf(chi2, df=dof))
    resid   = (decay - model) / np.sqrt(np.maximum(model, 1.0))

    # Pearson (Leica convention) chi-squared: sum((d - m)² / max(m, 1))
    sigma_p  = np.sqrt(np.maximum(m_win, 1.0))
    chi2_p   = float(np.sum(((d_win - m_win) / sigma_p)**2))
    rchi2_p  = chi2_p / dof

    # Tail-only chi2_r: exclude rising edge
    peak_bin_loc = int(np.argmax(decay[fit_start:fit_end])) + fit_start
    tail_start   = peak_bin_loc + max(1, int(0.05 * (fit_end - peak_bin_loc)))
    d_tail  = decay[tail_start:fit_end].astype(float)
    m_tail  = model[tail_start:fit_end]
    sw_tail = np.sqrt(np.maximum(d_tail, 1.0))
    chi2_tail  = float(np.sum(((d_tail - m_tail) / sw_tail)**2))
    dof_tail   = max((fit_end - tail_start) - len(popt), 1)
    rchi2_tail = chi2_tail / dof_tail

    # Pearson tail
    sp_tail       = np.sqrt(np.maximum(m_tail, 1.0))
    chi2_tail_p   = float(np.sum(((d_tail - m_tail) / sp_tail)**2))
    rchi2_tail_p  = chi2_tail_p / dof_tail

    # Compute amplitude fractions and weighted means using the sorted arrays
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
                  use_gpu="auto",
                  gpu_backend=None) -> dict:
    ny, nx, _ = stack.shape
    # Per-pixel sync count: distribute total sync pulses evenly across pixels
    _n_sync_px = int(n_sync / max(ny * nx, 1)) if correct_pileup and n_sync > 0 else 0

    # Extract fixed IRF parameters from global fit using same unpacking order
    idx   = 2 * n_exp
    shift = global_popt[idx]; idx += 1
    sigma = global_popt[idx] if fit_sigma else 0.0
    if fit_sigma: idx += 1
    # skip bg — re-estimated per pixel
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
    ])  # (n_exp, n_bins)
    A = conv_basis.T   # (n_bins, n_exp)

    # GPU fast path — batches all pixels in one or two matrix operations.
    # For fixed-tau modes: one matmul (NNLS with pre-built basis).
    # For free-tau mode (n_exp > 1): batched Adam optimizer on GPU.
    # use_gpu="auto"  → use GPU if one is detected (default)
    # use_gpu=True    → same as "auto" but raises if GPU is explicitly wanted
    # use_gpu=False   → always use CPU path
    global _gpu_backend_cache
    if use_gpu is not False:
        _backend = gpu_backend
        if _backend is None:
            if _gpu_backend_cache is _GPU_BACKEND_UNSET:
                try:
                    from flimkit.GPU import get_backend
                    _gpu_backend_cache = get_backend()
                except Exception:
                    _gpu_backend_cache = None
            _backend = _gpu_backend_cache

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
                # free_tau and n_exp > 1: batched Adam
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
        # Alias for save_weighted_tau_images compatibility
        maps[f"a{i+1}"]      = maps[f"alpha_{i+1}"]

    fitted = skipped = 0
    t0 = time.time()

    if n_exp == 1:
        # Single-exponential: fit τ per pixel via log-τ grid scan
        # With only one component, NNLS amplitude-only fitting stamps the
        # global τ onto every pixel (fracs_px = [1.0] → tau_mean = tau_global).
        # Instead, scan a fine log-τ grid using the globally-fixed IRF and
        # pick the τ that minimises the scalar NNLS residual for each pixel.

        _lo = (tau_min_ns if tau_min_ns is not None
               else max(taus_fixed[0] * 1e9 / 20.0, 0.05)) * 1e-9
        _hi = (tau_max_ns if tau_max_ns is not None
               else min(taus_fixed[0] * 1e9 * 20.0, 45.0)) * 1e-9

        _N_GRID = 200
        tau_grid = np.logspace(np.log10(_lo), np.log10(_hi), _N_GRID)

        # Pre-compute IRF-convolved exponentials for every grid τ
        _irf_fft_g = np.fft.fft(irf_fixed)
        basis_grid = np.array([
            np.real(np.fft.ifft(
                np.fft.fft(np.exp(-t_axis / max(tau, 1e-15))) * _irf_fft_g))
            for tau in tau_grid
        ])  # (N_GRID, n_bins)

        # Precompute ||basis_grid[j]||² for the fast scalar NNLS formula
        bb_grid = np.maximum((basis_grid ** 2).sum(axis=1), 1e-20)  # (N_GRID,)

        for yi in tqdm(range(ny), desc='  Per-pixel rows', disable=True):
            if progress_callback is not None:
                progress_callback(yi, ny)

            decay_row = stack[yi, :, :].astype(float)  # (nx, n_bins)
            ph_counts = decay_row.sum(axis=1)           # (nx,)
            valid_xi  = np.where(ph_counts >= min_photons)[0]
            skipped  += nx - len(valid_xi)

            if len(valid_xi) == 0:
                continue

            dv = decay_row[valid_xi]  # (n_valid, n_bins)

            # Per-pixel background estimates (vectorised bg estimation)
            peak_b_v = np.argmax(dv, axis=1)  # (n_valid,)
            bg_v = np.array([estimate_bg(dv[k], int(peak_b_v[k]))
                             for k in range(len(valid_xi))])
            dc_v = np.maximum(dv - bg_v[:, np.newaxis], 0.0)  # (n_valid, n_bins)

            # Apply Coates pile-up correction per pixel if requested
            if correct_pileup and _n_sync_px > 0:
                dc_v = np.array([
                    coates_pileup_correction(dc_v[k], _n_sync_px)
                    for k in range(len(valid_xi))
                ])

            # Vectorised grid scan across all valid pixels in this row
            # bd[i,j] = basis_grid[j,:] · dc_v[i,:]
            bd     = dc_v @ basis_grid.T              # (n_valid, N_GRID)
            amps_g = np.maximum(bd / bb_grid, 0.0)   # (n_valid, N_GRID)
            dc_sq  = (dc_v ** 2).sum(axis=1)         # (n_valid,)
            # cost[i,j] = ||dc_v[i]||² - max(bd[i,j],0)² / ||basis[j]||²
            # Minimising cost ↔ maximising the projection bd²/||b||²
            costs  = dc_sq[:, np.newaxis] - np.maximum(bd, 0.0) ** 2 / bb_grid
            best_g = np.argmin(costs, axis=1)                          # (n_valid,)
            tau_v  = tau_grid[best_g]                                  # (n_valid,) s
            amp_v  = amps_g[np.arange(len(valid_xi)), best_g]         # (n_valid,)

            good = amp_v > 0
            skipped += int((~good).sum())

            for k, xi in enumerate(valid_xi):
                if not good[k]:
                    continue
                tau_ns = float(tau_v[k] * 1e9)
                maps["tau_1"][yi, xi]        = tau_ns
                maps["tau_mean_amp"][yi, xi] = tau_ns
                maps["tau_mean_int"][yi, xi] = tau_ns
                maps["alpha_1"][yi, xi]      = float(amp_v[k])
                maps["frac_1"][yi, xi]       = 1.0
                # Reduced chi² using the nearest grid basis vector
                best_b   = basis_grid[best_g[k]]
                model_px = float(amp_v[k]) * best_b + bg_v[k]
                resid    = dv[k] - model_px
                chi2_px  = float(np.sum(resid ** 2 / np.maximum(model_px, 1.0)))
                maps["chi2_r"][yi, xi] = chi2_px / max(n_bins - 2, 1)
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

                maps["tau_mean_int"][yi, xi] = tau_int
                maps["tau_mean_amp"][yi, xi] = tau_amp
                maps["chi2_r"][yi, xi]       = chi2_px / dof_px
                for i in range(n_exp):
                    maps[f"alpha_{i+1}"][yi, xi] = amps_px[i]
                    maps[f"frac_{i+1}"][yi, xi]  = fracs_px[i]
                fitted += 1

    else:  # free_tau and n_exp > 1
        # Free-τ per-pixel: LM fit with τ as free parameters.
        # IRF shift, sigma, tail fixed from global; bg estimated per pixel (not free).
        # Slower than NNLS (~30-100× per pixel) but recovers spatially resolved τ maps.

        tau_min_s = (tau_min_ns if tau_min_ns is not None
                     else taus_fixed.min() * 1e9 * 0.1) * 1e-9
        tau_max_s = (tau_max_ns if tau_max_ns is not None
                     else taus_fixed.max() * 1e9 * 10.0) * 1e-9
        amp_hi    = float(stack.max()) * 10.0
        # p = [τ1...τn, α1...αn]  (bg fixed per pixel, shift/sigma/tail from global)
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

                # Build full parameter vector: inject fixed shift, sigma, tail
                def _make_full(p_px):
                    taus_p = p_px[:n_exp]
                    amps_p = p_px[n_exp:2 * n_exp]
                    full   = list(taus_p) + list(amps_p) + [shift]
                    if fit_sigma:
                        full.append(sigma)
                    # bg: always fixed to per-pixel estimate (not a free param)
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

                # Sort by ascending τ (consistent with global convention)
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

                maps["tau_mean_int"][yi, xi] = tau_int
                maps["tau_mean_amp"][yi, xi] = tau_amp
                maps["chi2_r"][yi, xi]       = chi2_px / dof_px
                for i in range(n_exp):
                    maps[f"tau_{i+1}"][yi, xi]   = taus_ns[i]
                    maps[f"alpha_{i+1}"][yi, xi] = amps_sol[i]
                    maps[f"frac_{i+1}"][yi, xi]  = fracs_px[i]
                fitted += 1

    elapsed = time.time() - t0
    return maps