import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


def coates_pileup_correction(decay: np.ndarray, n_sync: int) -> np.ndarray:
    """Apply Coates pile-up correction to a TCSPC histogram.

    In TCSPC only the *first* photon arriving after each laser pulse is
    recorded.  This causes systematic underrepresentation of late time bins
    (long-lifetime components appear shorter than they are).  The Coates
    formula recovers the true histogram from the measured one.

    Algorithm
    ---------
    Let M(t)  = measured histogram (counts per bin, this function's input).
    Let N_s   = total number of sync pulses (laser shots) = ``n_sync``.
    Let C(t)  = cumulative sum of M up to but NOT including bin t:
                C(0) = 0, C(t) = sum_{k=0}^{t-1} M(k).

    The corrected histogram is:

        D(t) = -N_s * ln(1 - M(t) / (N_s - C(t)))

    This is exact for a Poisson photon-arrival process.  At low pile-up
    (M/N_s << 1) it reduces to M(t), as expected.

    Numerical stability: bins where N_s - C(t) <= M(t) (i.e. the argument
    of ln would be <= 0) are set to 0.  In practice this only occurs in the
    deep pre-IRF region where counts are zero anyway.

    References
    ----------
    Coates, P. B. (1968). "The correction for photon 'pile-up' in the
        measurement of radiative lifetimes." J. Phys. E: Sci. Instrum. 1,
        878–879.  https://doi.org/10.1088/0022-3735/1/8/437

    Becker, W. (2005). Advanced Time-Correlated Single Photon Counting
        Techniques. Springer.  §4.3 "Pile-up", pp. 82–84.

    Walker, J. G. (2002). "Iterative correction for 'pile-up' in single-
        photon lifetime measurement." Opt. Commun. 201, 271–277.
        https://doi.org/10.1016/S0030-4018(01)01663-7

    Parameters
    ----------
    decay  : 1-D array of measured histogram counts (integer or float).
    n_sync : Total number of laser sync pulses during acquisition.
             Computed as ``PTUFile.n_records`` (every record = one sync pulse
             in T3 format, including overflow markers, which is a conservative
             overestimate; in practice ~1–2 % error, negligible).

    Returns
    -------
    corrected : 1-D float array, same length as ``decay``, with pile-up
                distortion removed.  Sum is >= sum(decay).
    """
    m = np.asarray(decay, dtype=float)
    n_s = float(n_sync)

    cumulative = np.concatenate([[0.0], np.cumsum(m[:-1])])  # C(t), length = len(m)
    denom = n_s - cumulative                                  # N_s - C(t)

    # Argument of log: 1 - M(t)/(N_s - C(t))
    # Use np.errstate to suppress divide-by-zero/invalid warnings from np.where
    # evaluating both branches eagerly — guarded values are replaced by 0.0 anyway.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(denom > 0, m / denom, 0.0)
        arg   = 1.0 - ratio
        safe  = arg > 0.0
        corrected = np.where(safe, -n_s * np.log(arg), 0.0)
    return corrected

def find_irf_peak_bin(decay: np.ndarray, smooth_sigma: float = 1.5) -> int:
    """
    Estimate the IRF peak position from the maximum of the first derivative
    of the smoothed decay (steepest point of the rising edge).

    np.argmax(decay) gives the fluorescence convolution peak, which is shifted
    right of the true IRF peak by 1-2 bins depending on the shortest lifetime.
    The IRF peak is where the measured signal rises fastest — i.e. where the
    derivative is maximum.

    Parameters
    ----------
    smooth_sigma : Gaussian smoothing width (bins) before differentiating.
                   1.5 bins is sufficient to suppress Poisson noise on the
                   rising edge without significantly shifting the peak estimate.
    """
    smoothed  = gaussian_filter1d(decay.astype(float), sigma=smooth_sigma)
    deriv     = np.gradient(smoothed)
    # Only search in the first half of the histogram (rising edge region)
    half      = len(decay) // 2
    peak_bin  = int(np.argmax(deriv[:half]))
    return peak_bin


def estimate_bg(decay: np.ndarray, peak_bin: int, pre_gap: int = 5) -> float:
    """Pre-IRF median — used as initial guess for bg when fit_bg=True."""
    end    = max(0, peak_bin - pre_gap)
    region = decay[:end]
    if len(region) >= 5:
        return max(float(np.median(region)), 0.0)
    return max(float(np.median(decay[-30:])), 0.0)

def estimate_bg_from_histogram(hist, pre_bins=20):
    """
    Estimate background from the first `pre_bins` time bins.

    Parameters
    ----------
    hist : np.ndarray
        3D array of shape (Y, X, H) – photon counts.
    pre_bins : int, optional
        Number of initial bins to use.

    Returns
    -------
    bg : float
        Estimated background level (counts per bin).
    """
    # Average over all pixels and the first pre_bins bins
    if hist.ndim == 3:
        bg_region = hist[..., :pre_bins].mean()
    else:
        bg_region = hist[:pre_bins].mean()
    return float(bg_region)

def find_fit_start(decay: np.ndarray, irf_prompt: np.ndarray,
                   tcspc_res: float, pre_bins: int = 5) -> int:
    """
    Start the fit window at the IRF onset minus pre_bins, not at bin 0.

    Bins before the IRF onset are flat background — including them adds
    DOF without constraining the model and inflates χ²_r because the
    weighted residuals in the rising edge dominate.

    pre_bins: how many bins before the first non-negligible IRF value
              to include (captures any pre-IRF photons and rounding).
    """
    threshold    = irf_prompt.max() * 1e-3
    onset_bins   = np.where(irf_prompt >= threshold)[0]
    if len(onset_bins) == 0:
        return 0
    onset        = int(onset_bins[0])
    fit_start    = max(0, onset - pre_bins)
    return fit_start


def find_fit_end(decay, peak_bin, tau_max_s, tcspc_res, n_bins) -> int:
    candidate    = min(n_bins, peak_bin + int(6.0 * tau_max_s / tcspc_res))
    search_start = int(0.82 * n_bins)
    tail         = gaussian_filter1d(decay[search_start:].astype(float), sigma=2)
    deriv        = np.gradient(tail)
    thresh       = 3.0 * np.std(deriv[:max(1, len(deriv)//2)])
    spikes       = np.where(deriv > thresh)[0]
    if len(spikes) > 0:
        spike_abs = search_start + spikes[0]
        if spike_abs < candidate:
            print(f"  Next-period artefact at bin {spike_abs} "
                  f"({spike_abs*tcspc_res*1e9:.2f} ns). Truncating fit window.")
            candidate = spike_abs
    return candidate


def _build_bounds(n_exp, tau_min, tau_max, decay_peak,
                  has_tail, fit_bg, fit_sigma, bg_init=0.0, bg_upper=None,
                  sigma_max=3.0, irf_shift_bins=5):
    """
    Build lower/upper bound lists matching the parameter vector layout
    in reconvolution_model.
    """
    lo = [tau_min] * n_exp + [0.0] * n_exp + [-float(irf_shift_bins)]   # τ, α, shift
    hi = [tau_max] * n_exp + [10 * decay_peak] * n_exp + [float(irf_shift_bins)]

    if fit_sigma:
        lo += [0.0];  hi += [sigma_max]

    if fit_bg:
        _bg_hi = bg_upper if bg_upper is not None else bg_init * 1.5 + 10
        lo += [0.0];  hi += [_bg_hi]

    if has_tail:
        lo += [0.0,   1.0]
        hi += [5.0, 200.0]

    return lo, hi


def _pack_p0(n_exp, tau_min, tau_max, decay_peak,
             has_tail, fit_bg, fit_sigma, bg_init,
             tau_override=None):
    """Build initial parameter vector matching the layout in reconvolution_model."""
    if tau_override is not None:
        taus0 = np.asarray(tau_override)
    else:
        tmin  = max(tau_min, 1e-14) * 1.001
        tmax  = tau_max * 0.999
        taus0 = np.logspace(np.log10(tmin), np.log10(tmax), n_exp)

    amps0 = np.full(n_exp, decay_peak / n_exp)
    base  = np.concatenate([taus0, amps0, [0.0]])   # τ, α, shift

    if fit_sigma:
        base = np.concatenate([base, [0.3]])

    if fit_bg:
        base = np.concatenate([base, [bg_init]])

    if has_tail:
        base = np.concatenate([base, [0.5, 20.0]])

    return base