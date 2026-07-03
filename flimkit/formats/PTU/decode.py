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
    from .reader import PTUFile
    ptu = PTUFile(str(ptu_path), verbose=False)
    if hasattr(ptu, 'raw_pixel_stack'):
        stack = ptu.raw_pixel_stack(channel=channel, binning=binning)
    else:
        stack = ptu.pixel_stack(channel=channel, binning=binning)
        if stack.max() <= 1.0 and stack.sum() > 0:
            raise ValueError('Custom class returned normalized data')
    if stack.sum() == 0:
        raise ValueError('Custom class returned zero photons')
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
