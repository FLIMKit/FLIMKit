import numpy as np
from .reader import PTUFile

def signal_from_PTUFile(
    filename,
    /,
    *,
    dtype=None,
    frame=None,
    channel=0,
    dtime=0,
    binning=1,
    keepdims=False,
):
    from xarray import DataArray

    ptu = PTUFile(str(filename), verbose=False)
    is_image = ptu.n_x > 0 and ptu.n_y > 0
    _active_channels = np.asarray(ptu._active)

    if is_image:
        cubes = [ptu.raw_pixel_stack(channel=int(c), binning=binning)
                 for c in _active_channels]
        stack = np.stack(cubes, axis=0)
        data = np.transpose(stack, (1, 2, 0, 3))[np.newaxis, ...]
    else:
        hists = [ptu.summed_decay(channel=int(c)) for c in _active_channels]
        data = np.stack(hists, axis=0)[np.newaxis, np.newaxis, np.newaxis, :, :]

    if dtype is None:
        dtype = np.uint16
    data = data.astype(dtype)

    n_bins_full = data.shape[-1]

    if dtime is not None:
        if dtime < 0:
            data = data.sum(axis=-1, keepdims=keepdims)
        elif dtime > 0:
            data = data[..., :dtime]

    if channel is not None:
        if channel < 0:
            data = data.sum(axis=3, keepdims=keepdims)
        else:
            ch_positions = np.where(_active_channels == channel)[0]
            if len(ch_positions) == 0:
                raise ValueError(
                    f"Channel {channel} not found in PTU file. "
                    f"Available hardware channels: {_active_channels.tolist()}"
                )
            channel_idx = int(ch_positions[0])
            if keepdims:
                data = data[:, :, :, channel_idx : channel_idx + 1]
            else:
                data = data[:, :, :, channel_idx]

    if frame is not None:
        if frame < 0:
            data = data.sum(axis=0, keepdims=keepdims)
        else:
            if keepdims:
                data = data[frame : frame + 1]
            else:
                data = data[frame]

    all_dims = ['T', 'Y', 'X', 'C', 'H']
    removed = []
    if dtime is not None and dtime < 0 and not keepdims:
        removed.append('H')
    if channel is not None and channel >= 0 and not keepdims:
        removed.append('C')
    if frame is not None and not keepdims:
        removed.append('T')
    dims = [d for d in all_dims if d not in removed]

    coords = {}
    h_bins = dtime if (dtime is not None and dtime > 0) else n_bins_full
    time_ns = ptu.time_ns[:h_bins]
    if 'H' in dims:
        coords['H'] = ('H', time_ns)

    frequency_mhz = ptu.sync_rate * 1e-6

    da = DataArray(
        data,
        dims=dims,
        coords=coords,
        attrs={
            'frequency': frequency_mhz,
            'ptu_tags': ptu.tags,
        },
    )
    return da
