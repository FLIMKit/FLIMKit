import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


def calibrated_chi2(data, model, axis=None):
    """Return the expected-contribution-normalized model chi-square."""
    data_arr, model_arr = np.broadcast_arrays(
        np.asarray(data, dtype=float), np.asarray(model, dtype=float))
    valid = np.all(
        np.isfinite(data_arr) & (data_arr >= 0.0)
        & np.isfinite(model_arr) & (model_arr >= 0.0),
        axis=axis)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        numerator = np.sum(
            (data_arr - model_arr) ** 2 / np.maximum(model_arr, 1.0), axis=axis)
        expected = np.sum(np.minimum(model_arr, 1.0), axis=axis)
    if np.ndim(expected) == 0:
        return float(numerator / expected) if valid and expected > 0 else np.nan
    result = np.full(np.shape(expected), np.nan, dtype=float)
    return np.divide(
        numerator, expected, out=result,
        where=valid & (expected > 0) & np.isfinite(numerator))


def coates_pileup_correction(decay: np.ndarray, n_sync: int):
    m = np.asarray(decay, dtype=float)
    n_s = float(n_sync or 0.0)
    total = float(m.sum())
    if n_s <= 0.0 or n_s <= total:
        raise ValueError(
            f'coates_pileup_correction needs the excitation-pulse count N_sync; got '
            f'n_sync={n_s:,.0f} for {total:,.0f} photons. N_sync must exceed the photon count.')
    cumulative = np.concatenate([[0.0], np.cumsum(m[:-1])])
    denom = n_s - cumulative
    # Argument of log: 1 - M(t)/(N_s - C(t))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(denom > 0, m / denom, 0.0)
        arg = 1.0 - ratio
        safe = arg > 0.0
        corrected = np.where(safe, -n_s * np.log(arg), 0.0)
    return corrected

def bins_from_ns(value_ns, tcspc_res):
    return int(round(float(value_ns) * 1e-9 / tcspc_res))

def build_fit_idx(fit_start, fit_end, n_bins, exclude_bins=None):
    # explicit bin set the fit runs over: a contiguous window minus any excluded
    # bands (eg a reflection peak mid-decay). Dropping bins keeps the fit linear
    # in the amplitudes, so the per-pixel projection is unaffected.
    idx = np.arange(max(0, int(fit_start)), min(int(n_bins), int(fit_end)), dtype=int)
    if exclude_bins:
        keep = np.ones(idx.shape, dtype=bool)
        for lo, hi in exclude_bins:
            keep &= ~((idx >= int(lo)) & (idx < int(hi)))
        idx = idx[keep]
    if idx.size == 0:
        raise ValueError('fit window is empty (check the window and any exclusion bands)')
    return idx

def find_irf_peak_bin(decay: np.ndarray, smooth_sigma: float = 1.5):
    smoothed = gaussian_filter1d(decay.astype(float), sigma=smooth_sigma)
    deriv = np.gradient(smoothed)
    # Only search in the first half of the histogram (rising edge region)
    half = len(decay) // 2
    peak_bin = int(np.argmax(deriv[:half]))
    return peak_bin

def estimate_bg(decay: np.ndarray, peak_bin: int, pre_gap: int = 5):
    end = max(0, peak_bin - pre_gap)
    region = decay[:end]
    if len(region) >= 5:
        return max(float(np.median(region)), 0.0)
    return max(float(np.median(decay[-30:])), 0.0)

def estimate_bg_from_histogram(hist, pre_bins=20):
    if hist.ndim == 3:
        bg_region = hist[..., :pre_bins].mean()
    else:
        bg_region = hist[:pre_bins].mean()
    return float(bg_region)

def find_fit_start(decay: np.ndarray, irf_prompt: np.ndarray,
                   tcspc_res: float, pre_bins: int = 5):
    threshold = irf_prompt.max() * 1e-3
    onset_bins = np.where(irf_prompt >= threshold)[0]
    if len(onset_bins) == 0:
        return 0
    onset = int(onset_bins[0])
    fit_start = max(0, onset - pre_bins)
    return fit_start


def find_tail_fit_start(decay, peak_bin, n_bins, frac=0.9):
    d = np.asarray(decay, dtype=float)
    peak_val = float(d[peak_bin]) if 0 <= peak_bin < n_bins else 0.0
    if peak_val <= 0:
        return min(peak_bin + 1, n_bins - 1)
    below = np.where(d[peak_bin:] < frac * peak_val)[0]
    if len(below) == 0:
        return min(peak_bin + 1, n_bins - 1)
    return int(min(peak_bin + below[0], n_bins - 1))

def _build_bounds_tail(n_exp, tau_min, tau_max, decay_peak, fit_bg,
                       bg_init=0.0, bg_upper=None,
                       fit_t0=False, t0_init=0.0, t0_range=0.0,
                       fit_tvb=False, tvb_init=0.0, tvb_upper=None):
    lo = [tau_min] * n_exp + [0.0] * n_exp
    hi = [tau_max] * n_exp + [10 * decay_peak] * n_exp
    if fit_t0:
        lo += [t0_init - t0_range];  hi += [t0_init + t0_range]
    if fit_bg:
        _bg_hi = bg_upper if bg_upper is not None else bg_init * 1.5 + 10
        lo += [0.0];  hi += [_bg_hi]
    if fit_tvb:
        _tvb_hi = tvb_upper if tvb_upper is not None else tvb_init * 1.5 + 10
        lo += [0.0];  hi += [_tvb_hi]
    return lo, hi

def _pack_p0_tail(n_exp, tau_min, tau_max, decay_peak, fit_bg, bg_init,
                  tau_override=None, fit_t0=False, t0_init=0.0,
                  fit_tvb=False, tvb_init=0.0):
    if tau_override is not None:
        taus0 = np.asarray(tau_override)
    else:
        tmin = max(tau_min, 1e-14) * 1.001
        tmax = tau_max * 0.999
        taus0 = np.logspace(np.log10(tmin), np.log10(tmax), n_exp)
    amps0 = np.full(n_exp, decay_peak / n_exp)
    base = np.concatenate([taus0, amps0])
    if fit_t0:
        base = np.concatenate([base, [t0_init]])
    if fit_bg:
        base = np.concatenate([base, [bg_init]])
    if fit_tvb:
        base = np.concatenate([base, [tvb_init]])
    return base

def find_fit_end(decay, peak_bin, tau_max_s, tcspc_res, n_bins):
    candidate = min(n_bins, peak_bin + int(6.0 * tau_max_s / tcspc_res))
    search_start = int(0.82 * n_bins)
    tail = gaussian_filter1d(decay[search_start:].astype(float), sigma=2)
    deriv = np.gradient(tail)
    thresh = 3.0 * np.std(deriv[:max(1, len(deriv)//2)])
    spikes = np.where(deriv > thresh)[0]
    if len(spikes) > 0:
        spike_abs = search_start + spikes[0]
        if spike_abs < candidate:
            print(f"  Next-period artefact at bin {spike_abs} "
                  f"({spike_abs*tcspc_res*1e9:.2f} ns). Truncating fit window.")
            candidate = spike_abs
    return candidate

def _build_bounds(n_exp, tau_min, tau_max, decay_peak,
                  has_tail, fit_bg, fit_sigma, bg_init=0.0, bg_upper=None,
                  sigma_max=3.0, irf_shift_bins=5,
                  fit_tvb=False, tvb_init=0.0, tvb_upper=None):
    lo = [tau_min] * n_exp + [0.0] * n_exp + [-float(irf_shift_bins)]
    hi = [tau_max] * n_exp + [10 * decay_peak] * n_exp + [float(irf_shift_bins)]
    if fit_sigma:
        lo += [0.0];  hi += [sigma_max]
    if fit_bg:
        _bg_hi = bg_upper if bg_upper is not None else bg_init * 1.5 + 10
        lo += [0.0];  hi += [_bg_hi]
    if fit_tvb:
        _tvb_hi = tvb_upper if tvb_upper is not None else tvb_init * 1.5 + 10
        lo += [0.0];  hi += [_tvb_hi]
    if has_tail:
        lo += [0.0,   1.0]
        hi += [5.0, 200.0]
    return lo, hi

def _build_bounds_dist(n_components, tau_min, tau_max, decay_peak,
                       fit_bg, fit_sigma, bg_init=0.0, bg_upper=None,
                       sigma_max=3.0, irf_shift_bins=5,
                       fit_tvb=False, tvb_init=0.0, tvb_upper=None):
    width_max = (tau_max - tau_min) / 2.0
    width_min = tau_min / 20.0
    lo = [tau_min] * n_components + [width_min] * n_components + [0.0] * n_components + [-float(irf_shift_bins)]
    hi = [tau_max] * n_components + [width_max] * n_components + [10 * decay_peak] * n_components + [float(irf_shift_bins)]
    if fit_sigma:
        lo += [0.0];  hi += [sigma_max]
    if fit_bg:
        _bg_hi = bg_upper if bg_upper is not None else bg_init * 1.5 + 10
        lo += [0.0];  hi += [_bg_hi]
    if fit_tvb:
        _tvb_hi = tvb_upper if tvb_upper is not None else tvb_init * 1.5 + 10
        lo += [0.0];  hi += [_tvb_hi]
    return lo, hi


def _pack_p0_dist(n_components, tau_min, tau_max, decay_peak,
                  fit_bg, fit_sigma, bg_init, fit_tvb=False, tvb_init=0.0):
    tmin = max(tau_min, 1e-14) * 1.001
    tmax = tau_max * 0.999
    tau0 = np.logspace(np.log10(tmin), np.log10(tmax), n_components)
    width0 = tau0 * 0.2
    amp0 = np.full(n_components, decay_peak / n_components)
    base = np.concatenate([tau0, width0, amp0, [0.0]])
    if fit_sigma:
        base = np.concatenate([base, [0.3]])
    if fit_bg:
        base = np.concatenate([base, [bg_init]])
    if fit_tvb:
        base = np.concatenate([base, [tvb_init]])
    return base


def _pack_p0(n_exp, tau_min, tau_max, decay_peak,
             has_tail, fit_bg, fit_sigma, bg_init,
             tau_override=None, fit_tvb=False, tvb_init=0.0):
    if tau_override is not None:
        taus0 = np.asarray(tau_override)
    else:
        tmin = max(tau_min, 1e-14) * 1.001
        tmax = tau_max * 0.999
        taus0 = np.logspace(np.log10(tmin), np.log10(tmax), n_exp)
    amps0 = np.full(n_exp, decay_peak / n_exp)
    base = np.concatenate([taus0, amps0, [0.0]])
    if fit_sigma:
        base = np.concatenate([base, [0.3]])
    if fit_bg:
        base = np.concatenate([base, [bg_init]])
    if fit_tvb:
        base = np.concatenate([base, [tvb_init]])
    if has_tail:
        base = np.concatenate([base, [0.5, 20.0]])
    return base