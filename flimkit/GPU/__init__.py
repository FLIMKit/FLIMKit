import sys

def get_backend(prefer='auto'):
    if prefer == 'auto':
        for name in ('mlx', 'cuda', 'mps', 'rocm'):
            b = _try_backend(name)
            if b is not None:
                return b
        return None
    return _try_backend(prefer)


def _try_backend(name):
    if name == 'mlx':
        return _try_mlx()
    if name in ('cuda', 'mps', 'rocm'):
        return _try_torch(name)
    raise ValueError(
        f"Unknown backend {name!r}. "
        "Choose from: 'auto', 'mlx', 'cuda', 'mps', 'rocm'."
    )


def _try_mlx():
    if sys.platform != 'darwin':
        return None
    try:
        import mlx.core as mx  # noqa: F401
        gpu = mx.Device(mx.gpu)
        mx.eval(mx.array([1.0], device=gpu))  # confirm Metal is actually usable
    except Exception:
        return None
    from flimkit.GPU.mlx_backend import MLXBackend
    return MLXBackend()


def _try_torch(name):
    try:
        import torch
    except ImportError:
        return None

    if name == 'cuda':
        if not torch.cuda.is_available():
            return None
        device = 'cuda'
    elif name == 'mps':
        if not (torch.backends.mps.is_available() and
                torch.backends.mps.is_built()):
            return None
        device = 'mps'
    elif name == 'rocm':
        # ROCm uses the same CUDA API in PyTorch; tell them apart by device name
        if not torch.cuda.is_available():
            return None
        name_str = torch.cuda.get_device_name(0).lower()
        if not any(k in name_str for k in ('amd', 'radeon', 'vega', 'navi', 'gfx')):
            return None
        device = 'cuda'
    else:
        return None

    from flimkit.GPU.torch_backend import TorchBackend
    return TorchBackend(device=device)

