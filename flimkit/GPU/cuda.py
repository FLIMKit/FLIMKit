from flimkit.GPU.torch_backend import TorchBackend


class CUDABackend(TorchBackend):

    def __init__(self):
        super().__init__(device='cuda')

    def __repr__(self):
        return "CUDABackend(device='cuda')"