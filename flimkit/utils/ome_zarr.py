from pathlib import Path
import numpy as np

NGFF_VERSION = '0.4'


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _open_group(path):
    import zarr
    try:
        return zarr.open_group(store=str(path), mode='w', zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode='w')


def _compressor():
    try:
        from numcodecs import Blosc
    except ImportError:
        return None
    return Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)


def _write_array(group, name, data, chunks):
    codec = _compressor()
    if hasattr(group, 'create_array'):
        kwargs = {'compressors': codec} if codec is not None else {}
        arr = group.create_array(name=name, shape=data.shape, dtype=data.dtype,
                                 chunks=chunks, **kwargs)
        arr[:] = data
        return arr
    kwargs = {'compressor': codec} if codec is not None else {}
    return group.create_dataset(name, data=data, chunks=chunks, **kwargs)


def _channel_window(plane):
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def write_ome_zarr(path, channels, pixel_size_um=None, metadata=None, name='flimkit'):
    if not channels:
        raise ValueError('write_ome_zarr needs at least one channel')
    labels = list(channels)
    shapes = {np.asarray(channels[k]).shape[:2] for k in labels}
    if len(shapes) != 1:
        raise ValueError(f'all channels must share one shape, got {sorted(shapes)}')
    ny, nx = shapes.pop()
    stack = np.empty((len(labels), ny, nx), dtype=np.float32)
    windows = []
    for idx, label in enumerate(labels):
        plane = np.asarray(channels[label], dtype=np.float32)
        if plane.ndim == 3:
            plane = plane.mean(axis=2)
        stack[idx] = plane
        windows.append(_channel_window(plane))
    path = Path(path)
    group = _open_group(path)
    chunks = (1, min(ny, 512), min(nx, 512))
    _write_array(group, '0', stack, chunks)
    scale = [1.0, 1.0, 1.0]
    axes = [{'name': 'c', 'type': 'channel'},
            {'name': 'y', 'type': 'space'},
            {'name': 'x', 'type': 'space'}]
    if pixel_size_um and pixel_size_um > 0:
        scale = [1.0, float(pixel_size_um), float(pixel_size_um)]
        axes[1]['unit'] = 'micrometer'
        axes[2]['unit'] = 'micrometer'
    attrs = {
        'multiscales': [{
            'version': NGFF_VERSION,
            'name': name,
            'axes': axes,
            'datasets': [{'path': '0',
                          'coordinateTransformations': [{'type': 'scale', 'scale': scale}]}],
        }],
        'omero': {
            'name': name,
            'version': NGFF_VERSION,
            'rdefs': {'model': 'greyscale'},
            'channels': [{'label': label,
                          'color': 'FFFFFF',
                          'active': True,
                          'window': {'start': lo, 'end': hi, 'min': lo, 'max': hi}}
                         for label, (lo, hi) in zip(labels, windows)],
        },
    }
    if metadata:
        attrs['flimkit'] = _to_jsonable(metadata)
    group.attrs.update(attrs)
    return path


def fit_metadata(fit_result, pixel_size_um=None, channel_units=None):
    meta = {}
    summary = (fit_result or {}).get('global_summary')
    if isinstance(summary, dict):
        meta['global_summary'] = summary
    for key in ('n_exp', 'model', 'suggested_binning', 'tau_weighting', 'irf_source'):
        if fit_result and key in fit_result and not isinstance(fit_result[key], np.ndarray):
            meta[key] = fit_result[key]
    if pixel_size_um and pixel_size_um > 0:
        meta['pixel_size_um'] = float(pixel_size_um)
    if channel_units:
        meta['channel_units'] = channel_units
    return meta
