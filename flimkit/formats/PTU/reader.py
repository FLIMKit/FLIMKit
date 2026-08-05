import math
import numpy as np
from pathlib import Path

def _load_ptufile():
    try:
        import ptufile
    except ImportError:
        raise ImportError('ptufile is required to read PicoQuant .ptu files '
                          '(pip install ptufile)')
    return ptufile

def _norm_channel(channel):
    if not isinstance(channel, str):
        return channel
    channel = channel.strip()
    if channel == '':
        return None
    if channel.isdigit():
        return int(channel)
    return None

def _fit_bins(arr, n_bins):
    h = arr.shape[-1]
    if h == n_bins:
        return arr
    if h > n_bins:
        return arr[..., :n_bins]
    pad = [(0, 0)] * (arr.ndim - 1) + [(0, n_bins - h)]
    return np.pad(arr, pad)

def _reduce_yxh(arr, dims):
    d = list(dims)
    if arr.ndim != len(d):
        d = d[-arr.ndim:]
    drop = tuple(i for i, x in enumerate(d) if x not in ('Y', 'X', 'H'))
    if drop:
        arr = arr.sum(axis=drop)
    d = [x for x in d if x in ('Y', 'X', 'H')]
    return np.transpose(arr, [d.index(x) for x in ('Y', 'X', 'H') if x in d])

def _bin_cube(cube, binning):
    if binning <= 1:
        return cube
    ny, nx, nh = cube.shape
    ny2, nx2 = (ny // binning) * binning, (nx // binning) * binning
    cube = cube[:ny2, :nx2, :]
    return cube.reshape(ny2 // binning, binning, nx2 // binning, binning, nh).sum(axis=(1, 3))

def read_pck(path):
    ptufile = _load_ptufile()
    try:
        with ptufile.PqFile(str(path)) as f:
            tags = dict(f.tags)
    except Exception as exc:
        raise ValueError(f'Not a PicoQuant Check (.pck) file: {path!r}') from exc
    hist = tags.get('ChkHistogram')
    if hist is None:
        raise ValueError(f'No ChkHistogram block in {path!r} - not a Check/IRF .pck?')
    hist = np.frombuffer(np.asarray(hist).tobytes(), dtype='<u4')
    n_chan = max(1, int(tags.get('ChkChannels', 1)))
    if hist.size % n_chan == 0:
        hist = hist.reshape(n_chan, hist.size // n_chan)
    else:
        hist = hist.reshape(1, hist.size)
    return hist, tags

class PTUFile:
    def __init__(self, path, verbose=True):
        self.path = str(path)
        self.verbose = verbose
        ptufile = _load_ptufile()
        self._ptu = ptufile.PtuFile(self.path)
        self._parse_meta()

    def _parse_meta(self):
        p = self._ptu
        dims = dict(zip(p.dims, p.shape))
        self.tcspc_res = float(p.tcspc_resolution)
        gr = float(p.global_resolution) if p.global_resolution else 0.0
        self.global_resolution = gr
        self.sync_rate = (1.0 / gr) if gr else 0.0
        self.period_ns = gr * 1e9
        self.n_bins_stored = int(dims.get('H', 0))
        if self.tcspc_res > 0 and gr > 0:
            self.n_bins = int(math.ceil(gr / self.tcspc_res))
        else:
            self.n_bins = self.n_bins_stored
        self.n_x = int(dims.get('X', 0))
        self.n_y = int(dims.get('Y', 0))
        self.is_image = bool(getattr(p, 'is_image', self.n_x > 0 and self.n_y > 0))
        self._active = [int(c) for c in p.active_channels]
        self.n_channels = len(self._active)
        self.n_records = int(getattr(p, 'number_records', 0) or 0)
        self.n_photons = int(getattr(p, 'number_photons', 0) or 0)
        self.n_frames = int(getattr(p, 'number_images', 0) or 0) or 1
        self.n_sync = int(getattr(p, 'global_acquisition_time', 0) or 0) or None
        self.acq_time_s = float(getattr(p, 'acquisition_time', 0.0) or 0.0) or None
        self.time_ns = (np.arange(self.n_bins) + 0.5) * self.tcspc_res * 1e9
        self.photon_channel = None
        self.tags = dict(p.tags)
        self.rec_type = int(self.tags.get('TTResultFormat_TTTRRecType', 0))
        self._total_photons = None
        if self.verbose:
            print(f"  PTU      : {Path(self.path).name}")
            print(f"  RecType  : 0x{self.rec_type:08X}")
            print(f"  TCSPC    : {self.n_bins} bins x {self.tcspc_res*1e12:.2f} ps")
            if self.n_bins_stored > self.n_bins:
                print(f"             ({self.n_bins_stored} stored, using {self.n_bins} within one laser period)")
            print(f"  Laser    : {self.sync_rate/1e6:.3f} MHz  ({self.period_ns:.3f} ns)")
            if self.is_image:
                print(f"  Image    : {self.n_x} x {self.n_y} px, {self.n_channels} channel(s)")
            else:
                print(f"  Point    : {self.n_channels} channel(s)")
            print(f"  Records  : {self.n_records:,}  ({self.n_photons:,} photons)")
            if self.n_sync:
                print(f"  Sync     : {self.n_sync:,} pulses over {self.acq_time_s or 0.0:.1f} s"
                      f"  ({self.n_photons / self.n_sync:.4f} photons/pulse)")
            print(' ')

    def close(self):
        try:
            self._ptu.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def photons_per_pulse(self):
        if self._total_photons is None or not self.n_sync:
            return None
        return self._total_photons / self.n_sync

    def _ch_pos(self, channel):
        channel = int(channel)
        if channel in self._active:
            return self._active.index(channel)
        if 0 <= channel < self.n_channels:
            return channel
        if self.n_channels == 1:
            return 0
        raise ValueError(f'Channel {channel} not in PTU active channels {self._active}')

    def summed_decay(self, channel=None):
        channel = _norm_channel(channel)
        h = np.asarray(self._ptu.decode_histogram(asxarray=False), dtype=float)
        if h.ndim == 1:
            h = h[None, :]
        if channel is None:
            pos = int(np.argmax(h.sum(axis=1)))
            self.photon_channel = self._active[pos] if self._active else 0
        else:
            pos = self._ch_pos(channel)
        decay = _fit_bins(h[pos], self.n_bins).astype(float)
        self._total_photons = int(decay.sum())
        return decay

    def _decode_cube(self, channel):
        channel = _norm_channel(channel)
        if channel is None:
            if self.photon_channel is None:
                self.summed_decay(channel=None)
            channel = self.photon_channel
        pos = self._ch_pos(channel)
        p = self._ptu
        dims = dict(zip(p.dims, p.shape))
        ny, nx = int(dims.get('Y', 0)), int(dims.get('X', 0))
        if ny == 0 or nx == 0:
            raise RuntimeError('PTU has no scan image (point-mode); use summed_decay')
        arr = np.asarray(p.decode_image(channel=pos, frame=-1, asxarray=False))
        cube = _reduce_yxh(arr, p.dims)
        return _fit_bins(cube, self.n_bins).astype(np.uint32)

    def raw_pixel_stack(self, channel=None, binning=1):
        cube = self._decode_cube(channel)
        if binning > 1:
            cube = _bin_cube(cube, binning)
        self._total_photons = int(cube.sum())
        return cube.astype(np.uint32)

    def pixel_stack(self, channel=None, binning=1):
        return self.raw_pixel_stack(channel=channel, binning=binning).astype(float)

    def intensity_image(self, channel=None, binning=1):
        return self.raw_pixel_stack(channel=channel, binning=binning).sum(axis=2)

def _metadata(ptu, data):
    return {
        'frequency': ptu.sync_rate,
        'tcspc_resolution': ptu.tcspc_res,
        'shape': data.shape,
        'dims': ('Y', 'X', 'H'),
        'tags': ptu.tags,
        'x_pixel_size': ptu.tags.get('ImgHdr_PixResol', ptu.tags.get('ImgHdr_PixRes', 0)),
        'y_pixel_size': ptu.tags.get('ImgHdr_PixResol', ptu.tags.get('ImgHdr_PixRes', 0)),
        'n_bins': ptu.n_bins,
        'time_ns': ptu.time_ns,
        'photon_channel': ptu.photon_channel,
    }

def read_ptu(path, binning=1, channel=None, verbose=False):
    ptu = PTUFile(path, verbose=verbose)
    data = ptu.pixel_stack(channel=channel, binning=binning)
    metadata = _metadata(ptu, data)
    return data, metadata

def get_intensity_image(path, binning=1, channel=None):
    data, metadata = read_ptu(path, binning=binning, channel=channel, verbose=False)
    img = data.sum(axis=2)
    return img, metadata

def get_flim_data(path, binning=1, channel=None):
    return read_ptu(path, binning=binning, channel=channel, verbose=False)

def normalise_flim(flim):
    if flim is None:
        return None
    if flim.ndim == 5:
        return flim[0, :, :, 0, :]
    if flim.ndim == 4:
        if flim.shape[0] == 1:
            return flim[0]
        else:
            return flim[:, :, 0, :]
    if flim.ndim == 3:
        return flim
    return None

def create_time_axis(n_bins, tcspc_resolution):
    return np.arange(n_bins) * tcspc_resolution * 1e9
