import numpy as np
from ..FLIM.irf_tools import build_full_irf

_N_QUAD = 200

def _exponential_kernel(tcspc_res, n_bins, taus, amps, bg, rep_period_s=None):
    t = np.arange(n_bins, dtype=float) * tcspc_res
    if rep_period_s:
        return sum(a * np.exp(-t / max(tau, 1e-15)) / -np.expm1(-rep_period_s / max(tau, 1e-15))
                   for a, tau in zip(amps, taus)) + bg
    return sum(a * np.exp(-t / max(tau, 1e-15))
               for a, tau in zip(amps, taus)) + bg

def period_grid_bins(rep_period_s, tcspc_res, n_bins):
    if not rep_period_s:
        return n_bins
    n_period = int(round(rep_period_s / tcspc_res))
    if n_period < n_bins and n_bins * tcspc_res - rep_period_s < tcspc_res:
        return n_bins
    if n_period < n_bins:
        raise ValueError(f'The pulse period ({rep_period_s * 1e9:.3f} ns) is shorter than the histogram '
                         f'({n_bins * tcspc_res * 1e9:.3f} ns), so more than one pulse lands in the window. '
                         f'The periodic model needs a period at least as long as the histogram.')
    return n_period

def apply_pileup(model, n_sync):
    n_s = float(n_sync)
    lam = np.maximum(np.asarray(model, dtype=float), 0.0) / n_s
    cum = np.concatenate([[0.0], np.cumsum(lam[:-1])])
    return n_s * np.exp(-cum) * (1.0 - np.exp(-lam))

class _DECost:
    def __init__(self, tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                 has_tail, fit_bg, fit_sigma,
                 fit_idx, decay, weights,
                 tvb_profile=None, fit_tvb=False, n_sync=None, rep_period_s=None):
        self.rep_period_s = rep_period_s
        self.tcspc_res = tcspc_res
        self.n_bins = n_bins
        self.irf_prompt = irf_prompt
        self.n_exp = n_exp
        self.bg_fixed = bg_fixed
        self.has_tail = has_tail
        self.fit_bg = fit_bg
        self.fit_sigma = fit_sigma
        self.fit_idx = fit_idx
        self.decay = decay
        self.weights = weights
        self.tvb_profile = tvb_profile
        self.fit_tvb = fit_tvb
        self.n_sync = n_sync

    def __call__(self, params):
        model = reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_exp, self.bg_fixed, self.has_tail,
            self.fit_bg, self.fit_sigma,
            tvb_profile=self.tvb_profile, fit_tvb=self.fit_tvb,
            n_sync=self.n_sync, rep_period_s=self.rep_period_s)
        res = ((model[self.fit_idx]
                - self.decay[self.fit_idx])
               / self.weights)
        return np.sum(res**2)

class _DECostLogTau(_DECost):
    def __call__(self, params):
        params_lin = np.array(params, dtype=float)
        params_lin[:self.n_exp] = 10.0 ** params_lin[:self.n_exp]
        return super().__call__(params_lin)

class _DECostPoisson:
    def __init__(self, tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                 has_tail, fit_bg, fit_sigma,
                 fit_idx, decay,
                 tvb_profile=None, fit_tvb=False, n_sync=None, rep_period_s=None):
        self.rep_period_s = rep_period_s
        self.tcspc_res = tcspc_res
        self.n_bins = n_bins
        self.irf_prompt = irf_prompt
        self.n_exp = n_exp
        self.bg_fixed = bg_fixed
        self.has_tail = has_tail
        self.fit_bg = fit_bg
        self.fit_sigma = fit_sigma
        self.fit_idx = fit_idx
        self.decay = decay
        self.tvb_profile = tvb_profile
        self.fit_tvb = fit_tvb
        self.n_sync = n_sync

    def __call__(self, params):
        model = reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_exp, self.bg_fixed, self.has_tail,
            self.fit_bg, self.fit_sigma,
            tvb_profile=self.tvb_profile, fit_tvb=self.fit_tvb,
            n_sync=self.n_sync, rep_period_s=self.rep_period_s)
        n = self.decay[self.fit_idx]
        m = np.maximum(model[self.fit_idx], 1e-10)
        dev = m - n
        pos = n > 0
        dev[pos] += n[pos] * np.log(n[pos] / m[pos])
        return 2.0 * np.sum(dev)

class _DECostPoissonLogTau(_DECostPoisson):

    def __call__(self, params):
        params_lin = np.array(params, dtype=float)
        params_lin[:self.n_exp] = 10.0 ** params_lin[:self.n_exp]
        return super().__call__(params_lin)

def unpack_tail_params(params, n_exp, fit_t0, fit_bg, fit_tvb,
                       t0_fixed=0.0, bg_fixed=0.0, tvb_fixed=0.0):
    taus = np.clip(np.asarray(params[:n_exp], dtype=float), 1e-14, None)
    amps = np.asarray(params[n_exp:2 * n_exp], dtype=float)
    idx = 2 * n_exp
    if fit_t0:
        t0 = float(params[idx]); idx += 1
    else:
        t0 = float(t0_fixed)
    if fit_bg:
        bg = float(params[idx]); idx += 1
    else:
        bg = float(bg_fixed)
    if fit_tvb:
        tvb_scale = float(params[idx]); idx += 1
    else:
        tvb_scale = float(tvb_fixed)
    return taus, amps, t0, bg, tvb_scale

def periodic_gain(taus, rep_period_s):
    taus = np.maximum(np.asarray(taus, dtype=float), 1e-15)
    if not rep_period_s:
        return np.ones_like(taus)
    return 1.0 / -np.expm1(-rep_period_s / taus)

def tail_basis(tcspc_res, n_bins, taus, t0, rep_period_s=None):
    t = np.arange(n_bins, dtype=float) * tcspc_res - t0
    live = t >= 0.0
    t_pos = np.maximum(t, 0.0)
    gain = periodic_gain(taus, rep_period_s)
    return np.array([g * np.where(live, np.exp(-t_pos / max(tau, 1e-15)), 0.0)
                     for tau, g in zip(taus, gain)])

def tail_model(params, tcspc_res, n_bins, n_exp, bg_fixed, fit_bg,
               fit_t0=False, t0_fixed=0.0,
               tvb_profile=None, fit_tvb=False, tvb_fixed=0.0,
               n_sync=None, rep_period_s=None):
    taus, amps, t0, bg, tvb_scale = unpack_tail_params(
        params, n_exp, fit_t0, fit_bg, fit_tvb,
        t0_fixed=t0_fixed, bg_fixed=bg_fixed, tvb_fixed=tvb_fixed)
    order = np.argsort(-taus)
    taus = taus[order]
    amps = amps[order]
    model = amps @ tail_basis(tcspc_res, n_bins, taus, t0, rep_period_s) + bg
    if tvb_profile is not None:
        model = model + tvb_scale * tvb_profile
    if n_sync:
        model = apply_pileup(model, n_sync)
    return model

class _DECostTail:
    def __init__(self, tcspc_res, n_bins, n_exp, bg_fixed, fit_bg,
                 fit_idx, decay, weights,
                 fit_t0=False, t0_fixed=0.0,
                 tvb_profile=None, fit_tvb=False, n_sync=None, rep_period_s=None):
        self.rep_period_s = rep_period_s
        self.tcspc_res = tcspc_res
        self.n_bins = n_bins
        self.n_exp = n_exp
        self.bg_fixed = bg_fixed
        self.fit_bg = fit_bg
        self.fit_idx = fit_idx
        self.decay = decay
        self.weights = weights
        self.fit_t0 = fit_t0
        self.t0_fixed = t0_fixed
        self.tvb_profile = tvb_profile
        self.fit_tvb = fit_tvb
        self.n_sync = n_sync

    def _model(self, params):
        return tail_model(
            params, self.tcspc_res, self.n_bins, self.n_exp,
            self.bg_fixed, self.fit_bg,
            fit_t0=self.fit_t0, t0_fixed=self.t0_fixed,
            tvb_profile=self.tvb_profile, fit_tvb=self.fit_tvb,
            n_sync=self.n_sync, rep_period_s=self.rep_period_s)

    def __call__(self, params):
        model = self._model(params)
        res = (model[self.fit_idx] - self.decay[self.fit_idx]) / self.weights
        return np.sum(res ** 2)

class _DECostTailLogTau(_DECostTail):
    def __call__(self, params):
        params_lin = np.array(params, dtype=float)
        params_lin[:self.n_exp] = 10.0 ** params_lin[:self.n_exp]
        return super().__call__(params_lin)

class _DECostTailPoisson(_DECostTail):
    def __init__(self, tcspc_res, n_bins, n_exp, bg_fixed, fit_bg,
                 fit_idx, decay,
                 fit_t0=False, t0_fixed=0.0,
                 tvb_profile=None, fit_tvb=False, n_sync=None, rep_period_s=None):
        super().__init__(tcspc_res, n_bins, n_exp, bg_fixed, fit_bg,
                         fit_idx, decay, None,
                         fit_t0=fit_t0, t0_fixed=t0_fixed,
                         tvb_profile=tvb_profile, fit_tvb=fit_tvb, n_sync=n_sync,
                         rep_period_s=rep_period_s)

    def __call__(self, params):
        model = self._model(params)
        n = self.decay[self.fit_idx]
        m = np.maximum(model[self.fit_idx], 1e-10)
        dev = m - n
        pos = n > 0
        dev[pos] += n[pos] * np.log(n[pos] / m[pos])
        return 2.0 * np.sum(dev)

class _DECostTailPoissonLogTau(_DECostTailPoisson):
    def __call__(self, params):
        params_lin = np.array(params, dtype=float)
        params_lin[:self.n_exp] = 10.0 ** params_lin[:self.n_exp]
        return super().__call__(params_lin)

def _alpha_gaussian(tau_grid, tau_center, sigma_tau):
    return np.exp(-0.5 * ((tau_grid - tau_center) / max(sigma_tau, 1e-15)) ** 2)

def _alpha_lorentzian(tau_grid, tau_center, gamma):
    half_g = max(gamma / 2.0, 1e-15)
    return half_g ** 2 / ((tau_grid - tau_center) ** 2 + half_g ** 2)

def _dist_kernel(tcspc_res, n_bins, tau_center, width, amp, dist_type, n_quad=_N_QUAD, rep_period_s=None):
    spread = 4.0 * width if dist_type == 'gaussian' else 8.0 * max(width / 2.0, 1e-15)
    tau_lo = max(tau_center - spread, 1e-12)
    tau_hi = max(tau_center + spread, tau_lo + 1e-12)
    tau_grid = np.linspace(tau_lo, tau_hi, n_quad)
    alpha = (_alpha_gaussian(tau_grid, tau_center, width) if dist_type == 'gaussian'
             else _alpha_lorentzian(tau_grid, tau_center, width))
    alpha_sum = alpha.sum()
    if alpha_sum > 0:
        alpha = alpha / alpha_sum
    t = np.arange(n_bins, dtype=float) * tcspc_res
    exp_mat = np.exp(-t[None, :] / np.maximum(tau_grid[:, None], 1e-15))
    return amp * ((alpha * periodic_gain(tau_grid, rep_period_s)) @ exp_mat)

def dist_reconvolution_model(params, tcspc_res, n_bins, irf_prompt,
                              n_components, dist_type, bg_fixed, fit_bg, fit_sigma,
                              tvb_profile=None, fit_tvb=False, tvb_fixed=0.0,
                              n_sync=None, rep_period_s=None):
    tau_centers = np.clip(params[:n_components], 1e-14, None)
    widths = np.clip(params[n_components:2 * n_components], 1e-14, None)
    amps = params[2 * n_components:3 * n_components]
    idx = 3 * n_components
    shift = params[idx]; idx += 1
    sigma = params[idx] if fit_sigma else 0.0
    if fit_sigma:
        idx += 1
    if fit_bg:
        bg = params[idx]; idx += 1
    else:
        bg = bg_fixed
    if fit_tvb:
        tvb_scale = params[idx]; idx += 1
    else:
        tvb_scale = tvb_fixed
    n_grid = period_grid_bins(rep_period_s, tcspc_res, n_bins)
    kernel = np.zeros(n_grid, dtype=float)
    for i in range(n_components):
        kernel += _dist_kernel(tcspc_res, n_grid, tau_centers[i], widths[i], amps[i], dist_type,
                               rep_period_s=rep_period_s)
    kernel += bg
    irf_full = np.pad(build_full_irf(irf_prompt, shift, sigma, 0.0, 1.0, n_bins), (0, n_grid - n_bins))
    model = np.real(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(irf_full)))[:n_bins]
    if tvb_profile is not None:
        model = model + tvb_scale * tvb_profile
    if n_sync:
        model = apply_pileup(model, n_sync)
    return model

def build_dist_basis_grid(tcspc_res, n_bins, irf_fixed,
                           tau_grid, width_grid, dist_type, n_quad=_N_QUAD, rep_period_s=None):
    n_grid = period_grid_bins(rep_period_s, tcspc_res, n_bins)
    irf_fft = np.fft.fft(np.pad(np.asarray(irf_fixed, dtype=float), (0, n_grid - n_bins)))
    n_total = len(tau_grid) * len(width_grid)
    basis = np.empty((n_total, n_bins), dtype=np.float32)
    param_pairs = np.empty((n_total, 2), dtype=np.float32)
    idx = 0
    for tau_c in tau_grid:
        for w in width_grid:
            kernel = _dist_kernel(tcspc_res, n_grid, tau_c, w, 1.0, dist_type, n_quad, rep_period_s)
            basis[idx] = np.real(np.fft.ifft(np.fft.fft(kernel) * irf_fft))[:n_bins].astype(np.float32)
            param_pairs[idx] = [tau_c, w]
            idx += 1
    return basis, param_pairs

class _DECostDist:
    def __init__(self, tcspc_res, n_bins, irf_prompt, n_components, dist_type,
                 bg_fixed, fit_bg, fit_sigma, fit_idx, decay, weights,
                 tvb_profile=None, fit_tvb=False, n_sync=None, rep_period_s=None):
        self.rep_period_s = rep_period_s
        self.tcspc_res = tcspc_res
        self.n_bins = n_bins
        self.irf_prompt = irf_prompt
        self.n_components = n_components
        self.dist_type = dist_type
        self.bg_fixed = bg_fixed
        self.fit_bg = fit_bg
        self.fit_sigma = fit_sigma
        self.fit_idx = fit_idx
        self.decay = decay
        self.weights = weights
        self.tvb_profile = tvb_profile
        self.fit_tvb = fit_tvb
        self.n_sync = n_sync

    def __call__(self, params):
        model = dist_reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_components, self.dist_type,
            self.bg_fixed, self.fit_bg, self.fit_sigma,
            tvb_profile=self.tvb_profile, fit_tvb=self.fit_tvb,
            n_sync=self.n_sync, rep_period_s=self.rep_period_s)
        res = ((model[self.fit_idx]
                - self.decay[self.fit_idx])
               / self.weights)
        return np.sum(res ** 2)

class _DECostDistLogParam(_DECostDist):
    def __call__(self, params):
        p = np.array(params, dtype=float)
        n = self.n_components
        p[:n] = 10.0 ** p[:n]
        p[n:2*n] = 10.0 ** p[n:2*n]
        return super().__call__(p)

class _DECostDistPoisson:
    def __init__(self, tcspc_res, n_bins, irf_prompt, n_components, dist_type,
                 bg_fixed, fit_bg, fit_sigma, fit_idx, decay,
                 tvb_profile=None, fit_tvb=False, n_sync=None, rep_period_s=None):
        self.rep_period_s = rep_period_s
        self.tcspc_res = tcspc_res
        self.n_bins = n_bins
        self.irf_prompt = irf_prompt
        self.n_components = n_components
        self.dist_type = dist_type
        self.bg_fixed = bg_fixed
        self.fit_bg = fit_bg
        self.fit_sigma = fit_sigma
        self.fit_idx = fit_idx
        self.decay = decay
        self.tvb_profile = tvb_profile
        self.fit_tvb = fit_tvb
        self.n_sync = n_sync

    def __call__(self, params):
        model = dist_reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_components, self.dist_type,
            self.bg_fixed, self.fit_bg, self.fit_sigma,
            tvb_profile=self.tvb_profile, fit_tvb=self.fit_tvb,
            n_sync=self.n_sync, rep_period_s=self.rep_period_s)
        n = self.decay[self.fit_idx]
        m = np.maximum(model[self.fit_idx], 1e-10)
        dev = m - n
        pos = n > 0
        dev[pos] += n[pos] * np.log(n[pos] / m[pos])
        return 2.0 * np.sum(dev)

class _DECostDistPoissonLogParam(_DECostDistPoisson):
    def __call__(self, params):
        p = np.array(params, dtype=float)
        n = self.n_components
        p[:n] = 10.0 ** p[:n]
        p[n:2*n] = 10.0 ** p[n:2*n]
        return super().__call__(p)

def reconvolution_model(params, tcspc_res, n_bins, irf_prompt,
                        n_exp, bg_fixed, has_tail, fit_bg, fit_sigma,
                        tvb_profile=None, fit_tvb=False, tvb_fixed=0.0,
                        n_sync=None, rep_period_s=None):
    taus = np.clip(params[:n_exp], 1e-14, None)
    amps = params[n_exp:2*n_exp]
    order = np.argsort(-taus)               
    taus = taus[order]
    amps = amps[order]
    idx = 2 * n_exp
    shift = params[idx]; idx += 1
    if fit_sigma:
        sigma = params[idx]; idx += 1
    else:
        sigma = 0.0
    if fit_bg:
        bg = params[idx]; idx += 1
    else:
        bg = bg_fixed
    if fit_tvb:
        tvb_scale = params[idx]; idx += 1
    else:
        tvb_scale = tvb_fixed
    if has_tail:
        tail_amp = params[idx]
        tail_tau = params[idx + 1]
    else:
        tail_amp, tail_tau = 0.0, 1.0
    irf_full = build_full_irf(irf_prompt, shift, sigma, tail_amp, tail_tau, n_bins)
    n_grid = period_grid_bins(rep_period_s, tcspc_res, n_bins)
    if n_grid > n_bins:
        irf_full = np.pad(irf_full, (0, n_grid - n_bins))
    kernel = _exponential_kernel(tcspc_res, n_grid, taus, amps, bg, rep_period_s)
    model = np.real(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(irf_full)))[:n_bins]
    if tvb_profile is not None:
        model = model + tvb_scale * tvb_profile
    if n_sync:
        model = apply_pileup(model, n_sync)
    return model