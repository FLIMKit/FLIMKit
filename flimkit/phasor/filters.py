import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def _nan_aware_filter(arr, filter_fn):
    nan_mask = np.isnan(arr)
    filled = np.where(nan_mask, 0.0, arr)
    weight = np.where(nan_mask, 0.0, 1.0)

    filtered_data = filter_fn(filled)
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

    from flimkit.plugins import registry
    registered = registry.get_phasor_filter(method)
    if registered is not None:
        return registered(real, imag, mean=mean, sigma=sigma, size=size,
                          wavelet=wavelet, level=level,
                          threshold_mode=threshold_mode)

    raise ValueError(
        f"Unknown phasor filter method {method!r}.  "
        f"Choose from: {', '.join(repr(m) for m in phasor_filter_methods())}, None."
    )


def phasor_filter_methods():
    from flimkit.plugins import registry
    return ['gaussian', 'median', 'wavelet'] + [f.id for f in registry.phasor_filters()]


def _gaussian(real, imag, *, mean, sigma):
    if mean is not None:
        try:
            from phasorpy.filter import phasor_filter_gaussian
            _, real_f, imag_f = phasor_filter_gaussian(
                mean, real, imag, sigma=sigma)
            return np.asarray(real_f, dtype=float), np.asarray(imag_f, dtype=float)
        except ImportError:
            pass

    def _fn(x):
        return gaussian_filter(x, sigma=sigma)

    return (_nan_aware_filter(real.astype(float), _fn),
            _nan_aware_filter(imag.astype(float), _fn))


def _median(real, imag, *, mean, size):
    if mean is not None:
        try:
            from phasorpy.filter import phasor_filter_median
            _, real_f, imag_f = phasor_filter_median(
                mean, real, imag, size=size)
            return np.asarray(real_f, dtype=float), np.asarray(imag_f, dtype=float)
        except ImportError:
            pass

    def _fn(x):
        return median_filter(x, size=size, mode='reflect')

    return (_nan_aware_filter(real.astype(float), _fn),
            _nan_aware_filter(imag.astype(float), _fn))


def _wavelet(real, imag, *, wavelet, level, threshold_mode):
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            'PyWavelets is required for wavelet phasor filtering.  '
            'Run  python install.py  to reinstall core requirements.'
        ) from exc

    def _denoise(arr):
        nan_mask = np.isnan(arr)
        filled = np.where(nan_mask, 0.0, arr)
        coeffs = pywt.wavedec2(filled, wavelet=wavelet, level=level)

        detail_finest = coeffs[-1]
        sigma_est = np.median(np.abs(detail_finest)) / 0.6745
        threshold = sigma_est * np.sqrt(2 * np.log(filled.size))

        new_coeffs = [coeffs[0]]
        for detail_tuple in coeffs[1:]:
            new_coeffs.append(
                tuple(pywt.threshold(d, threshold, mode=threshold_mode)
                      for d in detail_tuple)
            )

        out = pywt.waverec2(new_coeffs, wavelet=wavelet)
        out = out[: arr.shape[0], : arr.shape[1]]
        out[nan_mask] = np.nan
        return out

    return _denoise(real.astype(float)), _denoise(imag.astype(float))
