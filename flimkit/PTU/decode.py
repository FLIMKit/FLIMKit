import numpy as np
from pathlib import Path
from .reader import normalise_flim, PTUFile

def create_time_axis(n_bins, tcspc_resolution):
    return np.arange(n_bins) * tcspc_resolution * 1e9

def get_flim_histogram_from_ptufile(
    ptu_path,
    rotate_cw=True,
    binning=1,
    channel=None
):
    # Attempt with custom PTUFile

    from .reader import PTUFile
    ptu = PTUFile(str(ptu_path), verbose=False)
    # Use raw_pixel_stack if available; else pixel_stack (but we prefer raw)
    if hasattr(ptu, 'raw_pixel_stack'):
        stack = ptu.raw_pixel_stack(channel=channel, binning=binning)
    else:
        stack = ptu.pixel_stack(channel=channel, binning=binning)
        # If pixel_stack returns normalized floats, treat as failure
        if stack.max() <= 1.0 and stack.sum() > 0:
            raise ValueError('Custom class returned normalized data')

    # Check if stack has any photons
    if stack.sum() == 0:
        raise ValueError('Custom class returned zero photons')

    # Success: rotate if needed and build metadata
    if rotate_cw:
        stack = np.rot90(stack, k=-1, axes=(0, 1))
    metadata = {
        'tcspc_resolution': ptu.tcspc_res,
        'n_time_bins': ptu.n_bins,
        'tile_shape': (ptu.n_y // binning, ptu.n_x // binning),
        'frequency': ptu.sync_rate,
        'binning': binning,
        'channel': channel,
    }
    return stack, metadata

def get_raw_flim_histogram(ptu_path, rotate_cw=True):
    ptu = PTUFile(str(ptu_path), verbose=False)
    stack = ptu.raw_pixel_stack(channel=None, binning=1)
    if rotate_cw:
        stack = np.rot90(stack, k=-1, axes=(0, 1))
    metadata = {
        'tcspc_resolution': ptu.tcspc_res,
        'n_time_bins': ptu.n_bins,
        'tile_shape': (ptu.n_y, ptu.n_x),
        'frequency': ptu.sync_rate,
    }
    return stack, metadata

def get_raw_flim_histogram2(ptu_path, rotate_cw=True):
    import ptufile
    ptu = ptufile.PtuFile(str(ptu_path))
    data = ptu[:].squeeze()
    if data.ndim != 3:
        # If there are extra dims (e.g., T, C), sum over them
        # For simplicity, assume H is last and others are singletons or to be summed
        # This matches typical FLIM microscope data: (T, Y, X, C, H) with T=1, C=1
        data = data.reshape((data.shape[0], data.shape[1], -1))
    if rotate_cw:
        data = np.rot90(data, k=-1, axes=(0, 1))
    metadata = {
        'tcspc_resolution': ptu.tcspc_resolution,
        'n_time_bins': data.shape[2],
        'tile_shape': (data.shape[0], data.shape[1]),
        'frequency': ptu.frequency,
    }
    return data.astype(np.uint32), metadata