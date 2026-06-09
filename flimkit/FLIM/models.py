import numpy as np
from ..FLIM.irf_tools import build_full_irf

_N_QUAD = 200

def _exponential_kernel(tcspc_res, n_bins, taus, amps, bg):
    t = np.arange(n_bins, dtype=float) * tcspc_res
    return sum(a * np.exp(-t / max(tau, 1e-15))
               for a, tau in zip(amps, taus)) + bg

class _DECost:
    def __init__(self, tcspc_res, n_bins, irf_prompt, n_exp, bg_fixed,
                 has_tail, fit_bg, fit_sigma,
                 fit_start, fit_end, decay, weights):
        self.tcspc_res  = tcspc_res
        self.n_bins     = n_bins
        self.irf_prompt = irf_prompt
        self.n_exp      = n_exp
        self.bg_fixed   = bg_fixed
        self.has_tail   = has_tail
        self.fit_bg     = fit_bg
        self.fit_sigma  = fit_sigma
        self.fit_start  = fit_start
        self.fit_end    = fit_end
        self.decay      = decay
        self.weights    = weights

    def __call__(self, params):
        model = reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_exp, self.bg_fixed, self.has_tail,
            self.fit_bg, self.fit_sigma)
        res = ((model[self.fit_start:self.fit_end]
                - self.decay[self.fit_start:self.fit_end])
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
                 fit_start, fit_end, decay):
        self.tcspc_res  = tcspc_res
        self.n_bins     = n_bins
        self.irf_prompt = irf_prompt
        self.n_exp      = n_exp
        self.bg_fixed   = bg_fixed
        self.has_tail   = has_tail
        self.fit_bg     = fit_bg
        self.fit_sigma  = fit_sigma
        self.fit_start  = fit_start
        self.fit_end    = fit_end
        self.decay      = decay          # raw counts (not normalised)

    def __call__(self, params):
        model = reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_exp, self.bg_fixed, self.has_tail,
            self.fit_bg, self.fit_sigma)
        n = self.decay[self.fit_start:self.fit_end]
        m = np.maximum(model[self.fit_start:self.fit_end], 1e-10)
        # Poisson deviance (C-statistic)
        dev = m - n
        pos = n > 0
        dev[pos] += n[pos] * np.log(n[pos] / m[pos])
        return 2.0 * np.sum(dev)


class _DECostPoissonLogTau(_DECostPoisson):

    def __call__(self, params):
        params_lin = np.array(params, dtype=float)
        params_lin[:self.n_exp] = 10.0 ** params_lin[:self.n_exp]
        return super().__call__(params_lin)


def _alpha_gaussian(tau_grid, tau_center, sigma_tau):
    return np.exp(-0.5 * ((tau_grid - tau_center) / max(sigma_tau, 1e-15)) ** 2)


def _alpha_lorentzian(tau_grid, tau_center, gamma):
    half_g = max(gamma / 2.0, 1e-15)
    return half_g ** 2 / ((tau_grid - tau_center) ** 2 + half_g ** 2)


def _dist_kernel(tcspc_res, n_bins, tau_center, width, amp, dist_type, n_quad=_N_QUAD):
    spread   = 4.0 * width if dist_type == 'gaussian' else 8.0 * max(width / 2.0, 1e-15)
    tau_lo   = max(tau_center - spread, 1e-12)
    tau_hi   = max(tau_center + spread, tau_lo + 1e-12)
    tau_grid = np.linspace(tau_lo, tau_hi, n_quad)

    alpha     = (_alpha_gaussian(tau_grid, tau_center, width) if dist_type == 'gaussian'
                 else _alpha_lorentzian(tau_grid, tau_center, width))
    alpha_sum = alpha.sum()
    if alpha_sum > 0:
        alpha = alpha / alpha_sum

    t       = np.arange(n_bins, dtype=float) * tcspc_res
    exp_mat = np.exp(-t[None, :] / np.maximum(tau_grid[:, None], 1e-15))
    return amp * (alpha @ exp_mat)


def dist_reconvolution_model(params, tcspc_res, n_bins, irf_prompt,
                              n_components, dist_type, bg_fixed, fit_bg, fit_sigma):
    # params layout: [tau_c×N, width×N, amp×N, shift, (sigma), (bg)]
    tau_centers = np.clip(params[:n_components], 1e-14, None)
    widths      = np.clip(params[n_components:2 * n_components], 1e-14, None)
    amps        = params[2 * n_components:3 * n_components]

    idx   = 3 * n_components
    shift = params[idx]; idx += 1

    sigma = params[idx] if fit_sigma else 0.0
    if fit_sigma:
        idx += 1

    bg = params[idx] if fit_bg else bg_fixed

    kernel = np.zeros(n_bins, dtype=float)
    for i in range(n_components):
        kernel += _dist_kernel(tcspc_res, n_bins, tau_centers[i], widths[i], amps[i], dist_type)
    kernel += bg

    irf_full = build_full_irf(irf_prompt, shift, sigma, 0.0, 1.0, n_bins)
    return np.real(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(irf_full)))


def build_dist_basis_grid(tcspc_res, n_bins, irf_fixed,
                           tau_grid, width_grid, dist_type, n_quad=_N_QUAD):
    irf_fft     = np.fft.fft(irf_fixed)
    n_total     = len(tau_grid) * len(width_grid)
    basis       = np.empty((n_total, n_bins), dtype=np.float32)
    param_pairs = np.empty((n_total, 2),      dtype=np.float32)

    idx = 0
    for tau_c in tau_grid:
        for w in width_grid:
            kernel           = _dist_kernel(tcspc_res, n_bins, tau_c, w, 1.0, dist_type, n_quad)
            basis[idx]       = np.real(np.fft.ifft(np.fft.fft(kernel) * irf_fft)).astype(np.float32)
            param_pairs[idx] = [tau_c, w]
            idx += 1

    return basis, param_pairs


class _DECostDist:
    def __init__(self, tcspc_res, n_bins, irf_prompt, n_components, dist_type,
                 bg_fixed, fit_bg, fit_sigma, fit_start, fit_end, decay, weights):
        self.tcspc_res    = tcspc_res
        self.n_bins       = n_bins
        self.irf_prompt   = irf_prompt
        self.n_components = n_components
        self.dist_type    = dist_type
        self.bg_fixed     = bg_fixed
        self.fit_bg       = fit_bg
        self.fit_sigma    = fit_sigma
        self.fit_start    = fit_start
        self.fit_end      = fit_end
        self.decay        = decay
        self.weights      = weights

    def __call__(self, params):
        model = dist_reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_components, self.dist_type,
            self.bg_fixed, self.fit_bg, self.fit_sigma)
        res = ((model[self.fit_start:self.fit_end]
                - self.decay[self.fit_start:self.fit_end])
               / self.weights)
        return np.sum(res ** 2)


class _DECostDistLogParam(_DECostDist):
    def __call__(self, params):
        p        = np.array(params, dtype=float)
        n        = self.n_components
        p[:n]    = 10.0 ** p[:n]
        p[n:2*n] = 10.0 ** p[n:2*n]
        return super().__call__(p)


class _DECostDistPoisson:
    def __init__(self, tcspc_res, n_bins, irf_prompt, n_components, dist_type,
                 bg_fixed, fit_bg, fit_sigma, fit_start, fit_end, decay):
        self.tcspc_res    = tcspc_res
        self.n_bins       = n_bins
        self.irf_prompt   = irf_prompt
        self.n_components = n_components
        self.dist_type    = dist_type
        self.bg_fixed     = bg_fixed
        self.fit_bg       = fit_bg
        self.fit_sigma    = fit_sigma
        self.fit_start    = fit_start
        self.fit_end      = fit_end
        self.decay        = decay

    def __call__(self, params):
        model = dist_reconvolution_model(
            params, self.tcspc_res, self.n_bins, self.irf_prompt,
            self.n_components, self.dist_type,
            self.bg_fixed, self.fit_bg, self.fit_sigma)
        n   = self.decay[self.fit_start:self.fit_end]
        m   = np.maximum(model[self.fit_start:self.fit_end], 1e-10)
        dev = m - n
        pos = n > 0
        dev[pos] += n[pos] * np.log(n[pos] / m[pos])
        return 2.0 * np.sum(dev)


class _DECostDistPoissonLogParam(_DECostDistPoisson):
    def __call__(self, params):
        p        = np.array(params, dtype=float)
        n        = self.n_components
        p[:n]    = 10.0 ** p[:n]
        p[n:2*n] = 10.0 ** p[n:2*n]
        return super().__call__(p)


def reconvolution_model(params, tcspc_res, n_bins, irf_prompt,
                        n_exp, bg_fixed, has_tail, fit_bg, fit_sigma):
    taus  = np.clip(params[:n_exp], 1e-14, None)
    amps  = params[n_exp:2*n_exp]

    # Enforce τ₁ > τ₂ > τ₃ by sorting descending 
    order = np.argsort(-taus)               
    taus = taus[order]
    amps = amps[order]

    idx   = 2 * n_exp
    shift = params[idx]; idx += 1

    if fit_sigma:
        sigma = params[idx]; idx += 1
    else:
        sigma = 0.0

    if fit_bg:
        bg = params[idx]; idx += 1
    else:
        bg = bg_fixed

    if has_tail:
        tail_amp = params[idx]
        tail_tau = params[idx + 1]
    else:
        tail_amp, tail_tau = 0.0, 1.0

    irf_full = build_full_irf(irf_prompt, shift, sigma, tail_amp, tail_tau, n_bins)
    kernel   = _exponential_kernel(tcspc_res, n_bins, taus, amps, bg)
    return np.real(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(irf_full)))