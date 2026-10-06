import os
import sys
import warnings

BUILTIN = ('mlx', 'cuda', 'mps', 'rocm')


def __getattr__(name):
    # imported on first use so `import flimkit.GPU` stays light
    if name == 'PluginBackend':
        from flimkit.GPU._base import PluginBackend
        return PluginBackend
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def get_backend(prefer='auto'):
    # FLIMKIT_FIT_BACKEND overrides 'auto', so an add-on backend can be bypassed
    # (or picked by name) without uninstalling anything
    if prefer == 'auto':
        prefer = os.environ.get('FLIMKIT_FIT_BACKEND', '').strip().lower() or 'auto'
    if prefer == 'cpu':
        return None
    if prefer == 'auto':
        builtin = None
        for name in BUILTIN:
            builtin = _try_backend(name)
            if builtin is not None:
                break
        for entry in _plugin_backends():
            b = _try_plugin(entry)
            if b is not None:
                # fits the add-on does not cover go to the GPU backend it displaced
                _set_fallback(b, builtin)
                return b
        return builtin
    return _try_backend(prefer)


def _set_fallback(backend, fallback):
    try:
        backend.fallback_backend = fallback
    except AttributeError:
        pass


def _plugin_backends():
    try:
        from flimkit import plugins
        plugins.ensure_loaded()
        return plugins.fit_backends()
    except Exception as exc:
        print(f'[GPU] add-on fit backends unavailable ({type(exc).__name__}: {exc})')
        return []


def _try_plugin(entry):
    try:
        backend = entry.create()
    except Exception as exc:
        print(f'[GPU] add-on fit backend {entry.id!r} from {entry.source} failed to '
              f'start ({type(exc).__name__}: {exc}), skipping it')
        return None
    if backend is not None:
        print(f'[GPU] using add-on fit backend {entry.id!r} ({entry.label})')
    return backend


def _try_backend(name):
    if name == 'mlx':
        return _try_mlx()
    if name in ('cuda', 'mps', 'rocm'):
        return _try_torch(name)
    found = [e for e in _plugin_backends() if e.id == str(name).lower()]
    if found:
        return _try_plugin(found[0])
    known = ', '.join(repr(n) for n in ('auto', 'cpu') + BUILTIN
                      + tuple(e.id for e in _plugin_backends()))
    raise ValueError(f"Unknown backend {name!r}. Choose from: {known}.")


def _try_mlx():
    if sys.platform != 'darwin':
        return None
    try:
        import mlx.core as mx  # noqa: F401
        gpu = mx.Device(mx.gpu)
        with mx.stream(gpu):
            mx.eval(mx.array([1.0]) + 1)  
    except Exception:
        return None
    from flimkit.GPU.mlx_backend import MLXBackend
    return MLXBackend()

def _cuda_available():
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_device_name():
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return torch.cuda.get_device_name(0).lower()
    except Exception:
        return ''


def _try_torch(name):
    try:
        import torch
    except ImportError:
        return None

    if name == 'cuda':
        if not _cuda_available():
            return None
        device = 'cuda'
    elif name == 'mps':
        if not (torch.backends.mps.is_available() and
                torch.backends.mps.is_built()):
            return None
        device = 'mps'
    elif name == 'rocm':
        if not _cuda_available():
            return None
        name_str = _cuda_device_name()
        if not any(k in name_str for k in ('amd', 'radeon', 'vega', 'navi', 'gfx')):
            return None
        device = 'cuda'
    else:
        return None

    from flimkit.GPU.torch_backend import TorchBackend
    return TorchBackend(device=device)

