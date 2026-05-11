from flimkit.GPU.torch_backend import TorchBackend


class MPSBackend(TorchBackend):
    """TorchBackend pinned to Metal Performance Shaders — Apple Silicon fallback."""

    def __init__(self):
        super().__init__(device="mps")

    def __repr__(self):
        return "MPSBackend(device='mps')"