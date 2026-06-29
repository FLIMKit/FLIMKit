import struct
import numpy as np
from pathlib import Path

_FRAME_CH = 5
_LINE_CH = 6
_PIXEL_CH = 7
_CLOCK_CH = 8

_TAG_EXTS = {
    'time': '.TAGTIME',
    'channel': '.TAGCHANNEL',
    'decay': '.TAGDECAY',
}

def _find_with_ext(stem, ext):
    direct = Path(str(stem) + ext)
    if direct.exists():
        return direct
    parent = stem.parent if str(stem.parent) else Path('.')
    target = (stem.name + ext).lower()
    if parent.exists():
        for f in parent.iterdir():
            if f.name.lower() == target:
                return f
    return None

def _resolve_triplet(path):
    p = Path(path)
    stem = p.with_suffix('') if p.suffix.upper() in (e.upper() for e in _TAG_EXTS.values()) else p
    out = {}
    for key, ext in _TAG_EXTS.items():
        cand = _find_with_ext(stem, ext)
        if cand is None:
            raise FileNotFoundError(f"Missing ISS TD-FLIM file '{stem.name}{ext}' next to {p}")
        out[key] = cand
    return out

def _read_tag_file(path, byteorder, value_bytes):
    dtype = np.dtype(f'{byteorder}i{value_bytes}')
    with open(path, 'rb') as fh:
        period = struct.unpack(f'{byteorder}i', fh.read(4))[0]
        data = np.frombuffer(fh.read(), dtype=dtype)
    return period, data

class ISSFile:
    def __init__(self, path, verbose=True, byteorder='<', n_bins=None):
        self.path = str(path)
        self.verbose = verbose
        self.byteorder = byteorder
        self.paths = _resolve_triplet(path)
        self._total_photons = None
        self._load()
        self._parse_meta(n_bins=n_bins)
    def _load(self):
        p_t, self.arrival_ps = _read_tag_file(self.paths['time'], self.byteorder, 8)
        p_c, self.channel = _read_tag_file(self.paths['channel'], self.byteorder, 4)
        p_d, self.decay = _read_tag_file(self.paths['decay'], self.byteorder, 4)
        periods = [p for p in (p_t, p_c, p_d) if p > 0]
        self.period_ps = periods[0] if periods else 0
        if len(set(periods)) > 1 and self.verbose:
            print(f'  WARNING: excitation period differs across files: {p_t}, {p_c}, {p_d} ps')
        n = min(self.arrival_ps.shape[0], self.channel.shape[0], self.decay.shape[0])
        if not (self.arrival_ps.shape[0] == self.channel.shape[0] == self.decay.shape[0]):
            if self.verbose:
                print(f'  WARNING: event counts differ '
                      f'(time={self.arrival_ps.shape[0]}, channel={self.channel.shape[0]}, '
                      f'decay={self.decay.shape[0]}); truncating to {n}')
            self.arrival_ps = self.arrival_ps[:n]
            self.channel = self.channel[:n]
            self.decay = self.decay[:n]
    def _parse_meta(self, n_bins=None):
        ach = np.abs(self.channel)
        is_det = (ach >= 1) & (ach <= 4)
        if n_bins is not None:
            self.n_bins = int(n_bins)
        else:
            dec_ph = self.decay[is_det]
            self.n_bins = int(dec_ph.max()) + 1 if dec_ph.size else 0
        period_s = self.period_ps * 1e-12
        self.frequency = (1.0 / period_s) if period_s > 0 else 0.0
        self.period_ns = period_s * 1e9
        self.tcspc_res = (period_s / self.n_bins) if self.n_bins > 0 else 0.0
        self.time_ns = (np.arange(self.n_bins) + 0.5) * self.tcspc_res * 1e9
        det = ach[is_det]
        self.photon_channel = int(np.bincount(det, minlength=5)[1:5].argmax() + 1) if det.size else None
        self.detector_channels = [int(c) for c in np.unique(det)] if det.size else []
        self.sync_rate = self.frequency
        self.n_records = int(self.channel.shape[0])
        self.rec_type = None
        is_frame = ach == _FRAME_CH
        is_line = ach == _LINE_CH
        is_pixel = ach == _PIXEL_CH
        self.n_y = int(self._counts_since(is_frame, is_line).max()) if is_line.any() else 0
        self.n_x = int(self._counts_since(is_line, is_pixel).max()) if is_pixel.any() else 0
        self.tags = {
            'ISS_ExcitationPeriod_ps': int(self.period_ps),
            'ISS_NumEvents': int(self.channel.shape[0]),
            'ISS_NumPhotons': int(is_det.sum()),
            'ISS_DecayResolution': int(self.n_bins),
            'ISS_DetectorChannels': self.detector_channels,
            'ISS_ImageShape': (self.n_y, self.n_x),
            'ISS_SourceFiles': {k: str(v) for k, v in self.paths.items()},
        }
        if self.verbose:
            print(f"  ISS TD-FLIM : {self.paths['time'].stem}")
            print(f'  Period   : {self.period_ns:.3f} ns  ({self.frequency/1e6:.3f} MHz)')
            print(f'  TCSPC    : {self.n_bins} bins x {self.tcspc_res*1e12:.2f} ps')
            print(f'  Events   : {self.channel.shape[0]:,}  (photons {int(is_det.sum()):,})')
            print(f'  Channels : {self.detector_channels}')
            print(' ')
    def _counts_since(self, reset_mask, count_mask):
        # counter that resets to 0 at every reset_mask event
        n = reset_mask.shape[0]
        cum = np.cumsum(count_mask.astype(np.int64))
        reset_pos = np.where(reset_mask, np.arange(n), -1)
        last = np.maximum.accumulate(reset_pos)
        base = np.zeros(n, dtype=np.int64)
        seen = last >= 0
        base[seen] = cum[last[seen]]
        return cum - base
    def summed_decay(self, channel=None):
        ach = np.abs(self.channel)
        is_det = (ach >= 1) & (ach <= 4)
        if channel is None:
            channel = self.photon_channel
        else:
            channel = int(channel)
        if channel is None:
            return np.zeros(self.n_bins, dtype=float)
        ph = is_det & (ach == channel)
        dec = self.decay[ph]
        dec = dec[(dec >= 0) & (dec < self.n_bins)].astype(np.int64)
        decay = np.bincount(dec, minlength=self.n_bins).astype(float)
        self._total_photons = int(decay.sum())
        return decay[:self.n_bins]
    @property
    def pileup_fraction(self):
        if self._total_photons is None or self.n_records == 0:
            return None
        return self._total_photons / self.n_records
    def pixel_stack(self, channel=None, binning=1, n_x=None, n_y=None):
        ach = np.abs(self.channel)
        is_frame = ach == _FRAME_CH
        is_line = ach == _LINE_CH
        is_pixel = ach == _PIXEL_CH
        is_det = (ach >= 1) & (ach <= 4)
        if not is_line.any():
            raise RuntimeError('No line markers (ch6) found; cannot build an image cube.')
        # y/x are 0-based: line/pixel markers seen since the parent reset, minus one
        y = self._counts_since(is_frame, is_line) - 1
        x = self._counts_since(is_line, is_pixel) - 1
        if channel is not None:
            ph = is_det & (ach == int(channel))
        else:
            ph = is_det
        dec = self.decay
        valid = ph & (y >= 0) & (x >= 0) & (dec >= 0) & (dec < self.n_bins)
        yv = y[valid].astype(np.int64)
        xv = x[valid].astype(np.int64)
        dv = dec[valid].astype(np.int64)
        ny = int(n_y) if n_y is not None else (self.n_y or (int(yv.max()) + 1 if yv.size else 0))
        nx = int(n_x) if n_x is not None else (self.n_x or (int(xv.max()) + 1 if xv.size else 0))
        keep = (yv < ny) & (xv < nx)
        yv, xv, dv = yv[keep], xv[keep], dv[keep]
        if binning > 1:
            yv = yv // binning
            xv = xv // binning
            ny = (ny + binning - 1) // binning
            nx = (nx + binning - 1) // binning
        if ny == 0 or nx == 0 or self.n_bins == 0:
            return np.zeros((ny, nx, self.n_bins), dtype=np.uint32)
        flat = (yv * nx + xv) * self.n_bins + dv
        cube = np.bincount(flat, minlength=ny * nx * self.n_bins)
        cube = cube.reshape(ny, nx, self.n_bins).astype(np.uint32)
        self._total_photons = int(cube.sum())
        if self.verbose:
            print(f'  Built cube {cube.shape}  ({self._total_photons:,} photons)')
        return cube
    def raw_pixel_stack(self, channel=None, binning=1):
        return self.pixel_stack(channel=channel, binning=binning)

def read_iss(path, binning=1, channel=None, verbose=False):
    iss = ISSFile(path, verbose=verbose)
    data = iss.pixel_stack(channel=channel, binning=binning)
    metadata = {
        'frequency': iss.frequency,
        'tcspc_resolution': iss.tcspc_res,
        'shape': data.shape,
        'dims': ('Y', 'X', 'H'),
        'tags': iss.tags,
        'x_pixel_size': 0,
        'y_pixel_size': 0,
        'n_bins': iss.n_bins,
        'time_ns': iss.time_ns,
        'photon_channel': iss.photon_channel,
    }
    return data, metadata

def get_intensity_image(path, binning=1, channel=None):
    data, metadata = read_iss(path, binning=binning, channel=channel, verbose=False)
    img = data.sum(axis=2)
    return img, metadata

def get_flim_data(path, binning=1, channel=None):
    return read_iss(path, binning=binning, channel=channel, verbose=False)

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