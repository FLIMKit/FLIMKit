from flimkit.GPU.torch_backend import TorchBackend


class CUDABackend(TorchBackend):
    """TorchBackend pinned to CUDA — use this when you know there's an NVIDIA GPU."""

    def __init__(self):
        super().__init__(device="cuda")

    def __repr__(self):
        return "CUDABackend(device='cuda')"