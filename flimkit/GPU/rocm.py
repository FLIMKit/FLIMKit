from flimkit.GPU.torch_backend import TorchBackend


class ROCmBackend(TorchBackend):

    def __init__(self):
        super().__init__(device='cuda')  # ROCm uses device="cuda" in PyTorch

    def __repr__(self):
        return "ROCmBackend(device='cuda/rocm')"