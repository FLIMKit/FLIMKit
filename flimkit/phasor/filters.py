import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def _nan_aware_filter(arr, filter_fn):
    """Apply *filter_fn* to *arr* while honouring NaN pixels.

    Uses the weighted-convolution trick: filter both the data and a binary
    validity mask then divide, so NaN pixels do not bleed into neighbours.

    Parameters
    ----------
    arr : ndarray
        2-D array that may contain NaN values.
    filter_fn : callable
        ``filter_fn(x) -> ndarray`` — applies a spatial filter to a
        fully-finite 2-D float array.

    Returns
    -------
    ndarray
        Filtered array with NaN preserved at the original NaN positions.
    """
    nan_mask = np.isnan(arr)
    filled   = np.where(nan_mask, 0.0, arr)
    weight   = np.where(nan_mask, 0.0, 1.0)

    filtered_data   = filter_fn(filled)
    filtered_weight = filter_fn(weight)

    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(filtered_weight > 0, filtered_data / filtered_weight, np.nan)

    out[nan_mask] = np.nan
    return out


def phasor_filter(
    real,
    imag,
    method,
    *,
    mean=None,
    sigma=1.0,
    size=3,
    wavelet='db4',
    level=1,
    threshold_mode='soft',
):
    """Apply a spatial filter to phasor G/S coordinate arrays.

    Parameters
    ----------
    real, imag : ndarray
        Calibrated phasor G and S arrays.  May contain NaN for masked /
        low-photon pixels.
    method : {'gaussian', 'median', 'wavelet', None}
        Filter to apply.  ``None`` or ``'none'`` returns the arrays unchanged.
    mean : ndarray, optional
        Mean-intensity image (same shape as real/imag).  Required to use the
        phasorpy native implementations; when omitted the scipy fallbacks are
        used instead.
    sigma : float
        Gaussian σ in pixels (default 1.0).
    size : int
        Median filter kernel size in pixels (default 3).
    wavelet : str
        Wavelet family for wavelet denoising (default ``'db4'``).
    level : int
        Wavelet decomposition level (default 1).
    threshold_mode : str
        ``'soft'`` or ``'hard'`` thresholding for wavelet denoising
        (default ``'soft'``).

    Returns
    -------
    real_f, imag_f : ndarray
        Filtered phasor coordinates with the same shape and NaN mask as
        the inputs.
    """
    if method is None or str(method).lower() in ('none', ''):
        return real, imag

    method = str(method).lower()

    if method == 'gaussian':
        return _gaussian(real, imag, mean=mean, sigma=sigma)
    if method == 'median':
        return _median(real, imag, mean=mean, size=size)
    if method == 'wavelet':
        return _wavelet(real, imag, wavelet=wavelet, level=level,
                        threshold_mode=threshold_mode)

    raise ValueError(
        f"Unknown phasor filter method {method!r}.  "
        "Choose from: 'gaussian', 'median', 'wavelet', None."
    )


def _gaussian(real, imag, *, mean, sigma):
    """Gaussian phasor filter."""
    # phasorpy >= 0.10 ships a NaN-aware C implementation — prefer it
    # (requires mean as the first positional argument)
    if mean is not None:
        try:
            from phasorpy.filter import phasor_filter_gaussian
            _, real_f, imag_f = phasor_filter_gaussian(
                mean, real, imag, sigma=sigma)
            return np.asarray(real_f, dtype=float), np.asarray(imag_f, dtype=float)
        except ImportError:
            pass

    # Fallback: scipy with NaN-aware normalisation
    def _fn(x):
        return gaussian_filter(x, sigma=sigma)

    return (_nan_aware_filter(real.astype(float), _fn),
            _nan_aware_filter(imag.astype(float), _fn))


def _median(real, imag, *, mean, size):
    """Median phasor filter."""
    # phasorpy >= 0.10 NaN-aware median — prefer it
    if mean is not None:
        try:
            from phasorpy.filter import phasor_filter_median
            _, real_f, imag_f = phasor_filter_median(
                mean, real, imag, size=size)
            return np.asarray(real_f, dtype=float), np.asarray(imag_f, dtype=float)
        except ImportError:
            pass

    # Fallback: scipy (NaN-aware via weighted convolution trick)
    def _fn(x):
        return median_filter(x, size=size, mode='reflect')

    return (_nan_aware_filter(real.astype(float), _fn),
            _nan_aware_filter(imag.astype(float), _fn))


def _wavelet(real, imag, *, wavelet, level, threshold_mode):
    """Wavelet soft/hard-threshold phasor denoising."""
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "PyWavelets is required for wavelet phasor filtering.  "
            "Run  python install.py  to reinstall core requirements."
        ) from exc

    def _denoise(arr):
        nan_mask = np.isnan(arr)
        filled   = np.where(nan_mask, 0.0, arr)
        coeffs   = pywt.wavedec2(filled, wavelet=wavelet, level=level)

        # Estimate noise σ from the finest-scale detail subband (MAD estimator)
        detail_finest = coeffs[-1]
        sigma_est     = np.median(np.abs(detail_finest)) / 0.6745
        threshold     = sigma_est * np.sqrt(2 * np.log(filled.size))

        new_coeffs = [coeffs[0]]
        for detail_tuple in coeffs[1:]:
            new_coeffs.append(
                tuple(pywt.threshold(d, threshold, mode=threshold_mode)
                      for d in detail_tuple)
            )

        out = pywt.waverec2(new_coeffs, wavelet=wavelet)
        # waverec2 may pad by 1 pixel — trim back to original shape
        out = out[: arr.shape[0], : arr.shape[1]]
        out[nan_mask] = np.nan
        return out

    return _denoise(real.astype(float)), _denoise(imag.astype(float))
