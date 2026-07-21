import numpy as np
from pathlib import Path

def _load_ptufile():
    try:
        import ptufile
    except ImportError:
        raise ImportError('ptufile is required to read PicoQuant .phu files '
                          '(pip install ptufile)')
    return ptufile

class PHUFile:
    modality = 'time'

    def __init__(self, path, verbose=True, channel=None, **kwargs):
        self.path = str(path)
        self.verbose = verbose
        self.channel = channel
        ptufile = _load_ptufile()
        with ptufile.PhuFile(self.path) as f:
            self._hist = [np.asarray(h, dtype=np.uint32) for h in f.histograms()]
            self.tcspc_res = float(f.tcspc_resolution)
            self._resolutions = [float(r) for r in f.histogram_resolutions]
            self.tags = dict(f.tags)
        self.n_channels = len(self._hist)
        self.n_bins = int(max((h.size for h in self._hist), default=0))
        self.n_x = 0
        self.n_y = 0
        self.is_image = False
        self.time_ns = (np.arange(self.n_bins) + 0.5) * self.tcspc_res * 1e9
        self.sync_rate = 0.0
        self.period_ns = 0.0
        self.photon_channel = None
        self._total_photons = None
        if self.verbose:
            print(f'  PicoQuant PHU : {Path(self.path).name}')
            print(f'  TCSPC    : {self.n_bins} bins x {self.tcspc_res*1e12:.2f} ps')
            print(f'  Point    : no image (histogram mode), {self.n_channels} histogram(s)')
            print(' ')

    def _ch_index(self, channel):
        if channel is None:
            totals = [int(h.sum()) for h in self._hist]
            idx = int(np.argmax(totals)) if totals else 0
            self.photon_channel = idx + 1
            return idx
        c = int(channel)
        idx = c - 1 if c >= 1 else c
        return idx if 0 <= idx < self.n_channels else 0

    def summed_decay(self, channel=None):
        if not self._hist:
            return np.zeros(self.n_bins, dtype=float)
        idx = self._ch_index(self.channel if channel is None else channel)
        decay = np.zeros(self.n_bins, dtype=float)
        h = self._hist[idx]
        decay[:h.size] = h
        self._total_photons = int(decay.sum())
        return decay

    def pixel_stack(self, *args, **kwargs):
        raise RuntimeError('PicoQuant .phu is histogram mode and has no scan image; '
                           'use summed_decay')

    def raw_pixel_stack(self, *args, **kwargs):
        return self.pixel_stack()

    @property
    def photons_per_pulse(self):
        return None

def read_phu(path, channel=None, verbose=False):
    phu = PHUFile(path, verbose=verbose)
    decay = phu.summed_decay(channel=channel)
    metadata = {
        'frequency': phu.sync_rate,
        'tcspc_resolution': phu.tcspc_res,
        'shape': decay.shape,
        'dims': ('H',),
        'tags': phu.tags,
        'n_bins': phu.n_bins,
        'time_ns': phu.time_ns,
        'photon_channel': phu.photon_channel,
    }
    return decay, metadata
