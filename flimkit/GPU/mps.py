from flimkit.GPU.torch_backend import TorchBackend


class MPSBackend(TorchBackend):

    def __init__(self):
        super().__init__(device='mps')

    def __repr__(self):
        return "MPSBackend(device='mps')"