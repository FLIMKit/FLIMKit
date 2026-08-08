from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.optimize import minimize


@dataclass
class AnisotropyResult:
    time_ns: np.ndarray
    parallel_decay: np.ndarray
    perpendicular_decay: np.ndarray
    anisotropy_decay: np.ndarray
    anisotropy_cube: np.ndarray
    anisotropy_map: np.ndarray
    parallel_intensity: np.ndarray
    perpendicular_intensity: np.ndarray
    parallel_background: float
    perpendicular_background: float
    spatial_window: int
    stride: int
    perpendicular_shift: tuple
    valid_mask: np.ndarray
    map_valid_mask: np.ndarray
    total_counts: np.ndarray
    window_origins: np.ndarray
    g_factor: float
    parallel_exposure: float
    perpendicular_exposure: float
    metadata: dict = field(default_factory=dict)


def apply_translation(data, shift_yx):
    data = np.asarray(data, dtype=float)
    if data.ndim < 2:
        raise ValueError('Spatial data must have at least two dimensions')
    if len(shift_yx) != 2 or not np.all(np.isfinite(shift_yx)):
        raise ValueError('shift_yx must contain two finite values')
    shift_vector = tuple(float(value) for value in shift_yx)
    shift_vector += (0.0,) * (data.ndim - 2)
    return nd_shift(data, shift_vector, order=1, mode='constant',
                    cval=0.0, prefilter=False)


def estimate_translation(reference, moving, max_shift=3.0):
    reference = np.asarray(reference, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if reference.ndim != 2 or moving.ndim != 2:
        raise ValueError('Registration images must be two-dimensional')
    if reference.shape != moving.shape:
        raise ValueError('Registration images must have the same shape')
    if not np.isfinite(max_shift) or max_shift <= 0:
        raise ValueError('max_shift must be positive and finite')
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(moving)):
        raise ValueError('Registration images must contain only finite values')

    reference = np.log1p(np.maximum(reference, 0.0))
    moving = np.log1p(np.maximum(moving, 0.0))
    if np.std(reference) <= np.finfo(float).eps or np.std(moving) <= np.finfo(float).eps:
        raise ValueError('Registration images must contain spatial variation')
    margin = min(12, max(0, min(reference.shape) // 8))

    def correlation(first, second):
        if margin:
            first = first[margin:-margin, margin:-margin]
            second = second[margin:-margin, margin:-margin]
        first = first.ravel()
        second = second.ravel()
        if np.std(first) == 0 or np.std(second) == 0:
            return -1.0
        return float(np.corrcoef(first, second)[0, 1])

    def objective(offset):
        shifted = nd_shift(moving, offset, order=1, mode='nearest',
                           prefilter=False)
        return -correlation(reference, shifted)

    fitted = minimize(
        objective, np.zeros(2), method='Powell',
        bounds=[(-max_shift, max_shift), (-max_shift, max_shift)],
        options={'xtol': 1e-4, 'ftol': 1e-10, 'maxiter': 100})
    if not fitted.success:
        raise RuntimeError(f'Image registration failed: {fitted.message}')
    boundary_tolerance = max(1e-3, max_shift * 1e-3)
    if np.any(np.abs(fitted.x) >= max_shift - boundary_tolerance):
        raise RuntimeError(
            'Image registration reached the search boundary; increase max_shift '
            'or disable automatic registration')
    if not np.isfinite(fitted.fun) or -float(fitted.fun) < 0.1:
        raise RuntimeError('Image registration confidence is too low')
    return tuple(float(value) for value in fitted.x)


def spatial_window_sum(data, window_size=1, stride=1):
    data = np.asarray(data, dtype=float)
    if data.ndim < 2:
        raise ValueError('Spatial data must have at least two dimensions')
    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError('window_size must be a positive integer')
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError('stride must be a positive integer')
    if window_size > data.shape[0] or window_size > data.shape[1]:
        raise ValueError('window_size cannot exceed the image dimensions')

    cumulative = data.cumsum(axis=0).cumsum(axis=1)
    pad = [(1, 0), (1, 0)] + [(0, 0)] * (data.ndim - 2)
    cumulative = np.pad(cumulative, pad, mode='constant')
    windowed = (cumulative[window_size:, window_size:]
                - cumulative[:-window_size, window_size:]
                - cumulative[window_size:, :-window_size]
                + cumulative[:-window_size, :-window_size])
    return windowed[::stride, ::stride]


def _select_time_bins(data, selector, name):
    if np.isscalar(selector):
        raise ValueError(f'{name} must preserve the TCSPC axis')
    try:
        selected = data[..., selector]
    except (IndexError, TypeError) as exc:
        raise ValueError(f'{name} is not a valid time-bin selector') from exc
    if selected.ndim != data.ndim or selected.shape[-1] == 0:
        raise ValueError(f'{name} must select at least one time bin')
    return selected


def subtract_background(data, background_bins):
    data = np.asarray(data, dtype=float)
    selected = _select_time_bins(data, background_bins, 'background_bins')
    background = np.median(selected, axis=-1)
    corrected = data - background[..., None]
    return corrected, background


def calculate_anisotropy(parallel, perpendicular, g_factor=1.0,
                          min_denominator=0.0, parallel_exposure=1.0,
                          perpendicular_exposure=1.0):
    parallel = np.asarray(parallel, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)
    if parallel.shape != perpendicular.shape:
        raise ValueError('Parallel and perpendicular data must have the same shape')
    if not np.isfinite(g_factor) or g_factor <= 0:
        raise ValueError('g_factor must be positive and finite')
    if (not np.isfinite(parallel_exposure) or parallel_exposure <= 0
            or not np.isfinite(perpendicular_exposure)
            or perpendicular_exposure <= 0):
        raise ValueError('Exposure values must be positive and finite')

    parallel_rate = parallel / parallel_exposure
    scaled_perpendicular = g_factor * perpendicular / perpendicular_exposure
    denominator = parallel_rate + 2.0 * scaled_perpendicular
    valid = np.isfinite(denominator) & (denominator > min_denominator)
    result = np.full(parallel.shape, np.nan, dtype=float)
    np.divide(parallel_rate - scaled_perpendicular, denominator,
              out=result, where=valid)
    return result


def analyze_anisotropy(parallel, perpendicular, time_ns, background_bins,
                       analysis_bins=None, g_factor=1.0, spatial_window=1,
                       stride=1, min_bin_photons=0.0,
                       min_map_photons=0.0,
                       perpendicular_shift=(0.0, 0.0),
                       parallel_exposure=1.0,
                       perpendicular_exposure=1.0):
    parallel = np.asarray(parallel, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)
    time_ns = np.asarray(time_ns, dtype=float)
    if parallel.ndim != 3 or perpendicular.ndim != 3:
        raise ValueError('Polarization data must have shape (Y, X, H)')
    if parallel.shape != perpendicular.shape:
        raise ValueError('Parallel and perpendicular stacks must have the same shape')
    if time_ns.ndim != 1 or time_ns.size != parallel.shape[-1]:
        raise ValueError('time_ns must match the TCSPC axis')
    if (not np.isfinite(min_bin_photons) or min_bin_photons < 0
            or not np.isfinite(min_map_photons) or min_map_photons < 0):
        raise ValueError('Photon thresholds must be finite and non-negative')
    if analysis_bins is None:
        analysis_bins = slice(None)

    parallel_decay_raw = parallel.sum(axis=(0, 1))
    perpendicular_decay_raw = perpendicular.sum(axis=(0, 1))
    parallel_decay, parallel_background = subtract_background(
        parallel_decay_raw, background_bins)
    perpendicular_decay, perpendicular_background = subtract_background(
        perpendicular_decay_raw, background_bins)
    anisotropy_decay = calculate_anisotropy(
        parallel_decay, perpendicular_decay, g_factor=g_factor,
        parallel_exposure=parallel_exposure,
        perpendicular_exposure=perpendicular_exposure)
    decay_observed = parallel_decay_raw + perpendicular_decay_raw
    anisotropy_decay[decay_observed < min_bin_photons] = np.nan

    parallel_pooled = spatial_window_sum(
        parallel, window_size=spatial_window, stride=stride)
    perpendicular_registered = apply_translation(
        perpendicular, perpendicular_shift)
    perpendicular_pooled = spatial_window_sum(
        perpendicular_registered, window_size=spatial_window, stride=stride)
    perpendicular_support = apply_translation(
        np.ones(perpendicular.shape[:2]), perpendicular_shift)
    support_pooled = spatial_window_sum(
        perpendicular_support[..., None], window_size=spatial_window,
        stride=stride)[..., 0]
    full_registration_support = np.isclose(
        support_pooled, spatial_window ** 2, rtol=0.0, atol=1e-6)
    parallel_corrected, _ = subtract_background(
        parallel_pooled, background_bins)
    perpendicular_corrected, _ = subtract_background(
        perpendicular_pooled, background_bins)
    anisotropy_cube = calculate_anisotropy(
        parallel_corrected, perpendicular_corrected, g_factor=g_factor,
        parallel_exposure=parallel_exposure,
        perpendicular_exposure=perpendicular_exposure)
    observed_cube = parallel_pooled + perpendicular_pooled
    bin_valid = ((observed_cube >= min_bin_photons)
                 & np.isfinite(anisotropy_cube)
                 & full_registration_support[..., None])
    anisotropy_cube[~bin_valid] = np.nan

    parallel_selected = _select_time_bins(
        parallel_corrected, analysis_bins, 'analysis_bins')
    perpendicular_selected = _select_time_bins(
        perpendicular_corrected, analysis_bins, 'analysis_bins')
    observed_selected = _select_time_bins(
        observed_cube, analysis_bins, 'analysis_bins')
    parallel_map = parallel_selected.sum(axis=-1)
    perpendicular_map = perpendicular_selected.sum(axis=-1)
    anisotropy_map = calculate_anisotropy(
        parallel_map, perpendicular_map, g_factor=g_factor,
        parallel_exposure=parallel_exposure,
        perpendicular_exposure=perpendicular_exposure)
    total_counts = observed_selected.sum(axis=-1)
    map_valid = ((total_counts >= min_map_photons)
                 & np.isfinite(anisotropy_map)
                 & full_registration_support)
    anisotropy_map[~map_valid] = np.nan
    origins_y = np.arange(anisotropy_cube.shape[0]) * stride
    origins_x = np.arange(anisotropy_cube.shape[1]) * stride
    grid_y, grid_x = np.meshgrid(origins_y, origins_x, indexing='ij')
    window_origins = np.stack([grid_y, grid_x], axis=-1)

    return AnisotropyResult(
        time_ns=time_ns,
        parallel_decay=parallel_decay,
        perpendicular_decay=perpendicular_decay,
        anisotropy_decay=anisotropy_decay,
        anisotropy_cube=anisotropy_cube,
        anisotropy_map=anisotropy_map,
        parallel_intensity=parallel_map,
        perpendicular_intensity=perpendicular_map,
        parallel_background=float(parallel_background),
        perpendicular_background=float(perpendicular_background),
        spatial_window=spatial_window,
        stride=stride,
        perpendicular_shift=tuple(float(value)
                                  for value in perpendicular_shift),
        valid_mask=bin_valid,
        map_valid_mask=map_valid,
        total_counts=total_counts,
        window_origins=window_origins,
        g_factor=float(g_factor),
        parallel_exposure=float(parallel_exposure),
        perpendicular_exposure=float(perpendicular_exposure),
    )


def analyze_ptu_pair(parallel_path, perpendicular_path, background_bins,
                     analysis_bins=None, g_factor=1.0, spatial_window=1,
                     stride=1, min_bin_photons=0.0,
                     min_map_photons=0.0, auto_register=False,
                     perpendicular_shift=(0.0, 0.0), reader_class=None,
                     parallel_channel=None, perpendicular_channel=None,
                     parallel_exposure=1.0, perpendicular_exposure=1.0):
    if parallel_channel is None or perpendicular_channel is None:
        raise ValueError('Photon channels must be selected explicitly')
    channels = (parallel_channel, perpendicular_channel)
    if any(isinstance(channel, (bool, np.bool_))
           or not isinstance(channel, (int, np.integer))
           or channel < 0 for channel in channels):
        raise ValueError('Photon channels must be non-negative integers')
    if reader_class is None:
        from flimkit.formats.PTU.reader import PTUFile
        reader_class = PTUFile

    with reader_class(parallel_path, verbose=False) as parallel_file:
        parallel = parallel_file.pixel_stack(channel=parallel_channel)
        parallel_time = np.asarray(parallel_file.time_ns, dtype=float)
    with reader_class(perpendicular_path, verbose=False) as perpendicular_file:
        perpendicular = perpendicular_file.pixel_stack(
            channel=perpendicular_channel)
        perpendicular_time = np.asarray(perpendicular_file.time_ns, dtype=float)

    if parallel.shape != perpendicular.shape:
        raise ValueError('Polarization PTU stacks must have the same shape')
    if (parallel_time.shape != perpendicular_time.shape
            or not np.allclose(parallel_time, perpendicular_time)):
        raise ValueError('Polarization PTUs must have matching time axes')
    if auto_register:
        perpendicular_shift = estimate_translation(
            parallel.sum(axis=-1), perpendicular.sum(axis=-1))

    result = analyze_anisotropy(
        parallel, perpendicular, parallel_time,
        background_bins=background_bins, analysis_bins=analysis_bins,
        g_factor=g_factor, spatial_window=spatial_window, stride=stride,
        min_bin_photons=min_bin_photons,
        min_map_photons=min_map_photons,
        perpendicular_shift=perpendicular_shift,
        parallel_exposure=parallel_exposure,
        perpendicular_exposure=perpendicular_exposure)
    result.metadata.update({
        'parallel_file': Path(parallel_path).name,
        'perpendicular_file': Path(perpendicular_path).name,
        'parallel_role': 'parallel',
        'perpendicular_role': 'perpendicular',
        'parallel_channel': int(parallel_channel),
        'perpendicular_channel': int(perpendicular_channel),
        'min_bin_photons': float(min_bin_photons),
        'min_map_photons': float(min_map_photons),
        'auto_registration': bool(auto_register),
    })
    if isinstance(background_bins, slice):
        result.metadata.update({
            'background_start_bin': background_bins.start,
            'background_stop_bin': background_bins.stop,
            'background_step': background_bins.step,
        })
    if isinstance(analysis_bins, slice):
        result.metadata.update({
            'analysis_start_bin': analysis_bins.start,
            'analysis_stop_bin': analysis_bins.stop,
            'analysis_step': analysis_bins.step,
        })
    return result


def save_anisotropy_npz(result, path):
    payload = {
        'time_ns': result.time_ns,
        'parallel_decay': result.parallel_decay,
        'perpendicular_decay': result.perpendicular_decay,
        'anisotropy_decay': result.anisotropy_decay,
        'anisotropy_cube': result.anisotropy_cube,
        'anisotropy_map': result.anisotropy_map,
        'parallel_intensity': result.parallel_intensity,
        'perpendicular_intensity': result.perpendicular_intensity,
        'parallel_background': result.parallel_background,
        'perpendicular_background': result.perpendicular_background,
        'valid_mask': result.valid_mask,
        'map_valid_mask': result.map_valid_mask,
        'total_counts': result.total_counts,
        'window_origins': result.window_origins,
        'perpendicular_shift': np.asarray(result.perpendicular_shift),
        'g_factor': result.g_factor,
        'parallel_exposure': result.parallel_exposure,
        'perpendicular_exposure': result.perpendicular_exposure,
        'spatial_window': result.spatial_window,
        'stride': result.stride,
    }
    reserved_keys = set(payload)
    for key, value in result.metadata.items():
        if not isinstance(key, str) or key in reserved_keys:
            continue
        if key.endswith('_file'):
            value = Path(value).name
        if isinstance(value, (str, int, float, bool, np.number)):
            output_key = f'metadata_{key}' if key in {'file', 'allow_pickle'} else key
            if output_key not in payload:
                payload[output_key] = value
    np.savez_compressed(path, **payload)
