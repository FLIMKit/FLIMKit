from flimkit.GPU.torch_backend import TorchBackend


class ROCmBackend(TorchBackend):
    """TorchBackend for AMD GPUs — ROCm uses the same CUDA API string under PyTorch."""

    def __init__(self):
        super().__init__(device="cuda")  # ROCm uses device="cuda" in PyTorch

    def __repr__(self):
        return "ROCmBackend(device='cuda/rocm')"