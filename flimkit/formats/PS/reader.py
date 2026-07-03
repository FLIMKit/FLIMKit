import numpy as np
from pathlib import Path
from . import decode as pd

class PSFile:
    def __init__(self, path, verbose=True, channel=None, pixels=512,
                 n_bins=256, period_ns=None):
        self.path = str(path)
        self.verbose = verbose
        self.header = pd.read_header(self.path)
        self.attrs = pd.read_attributes(self.path)
        self._names = [d['name'] for d in self.header['datasets']]
        self.dual_tdc = pd.has_dual_tdc(self.path)
        self.pos_range = 1 << int(self.attrs.get('/photons/PositionBits', 12))
        self.tac_range = 1 << int(self.attrs.get('/photons/TacBits', 12))
        self.pixels = int(pixels)
        self.n_bins = int(n_bins)
        self.n_x = self.pixels
        self.n_y = self.pixels
        self._period_ns = period_ns
        self._streams = None
        self._total_photons = None
        self.n_records = 0
        self.n_channels = 1
        self.photon_channel = 1
        self.detector_channels = [1]
        self._parse_meta(period_ns)

    def _parse_meta(self, period_ns, announce=True):
        # TacChannel = picoseconds per raw dt unit; dual-TDC dt is already in ps
        tac_channel = self.attrs.get('/photons/TacChannel')
        if period_ns and float(period_ns) > 0:
            period_s = float(period_ns) * 1e-9
            self.calib_source = 'user'
        elif self.dual_tdc:
            period_s = self.tac_range * 1e-12
            self.calib_source = 'dual_tdc'
        elif tac_channel:
            period_s = self.tac_range * float(tac_channel) * 1e-12
            self.calib_source = 'TacChannel'
        else:
            period_s = 0.0
            self.calib_source = 'unknown'
        if period_s > 0:
            self.frequency = 1.0 / period_s
            self.period_ns = period_s * 1e9
            self.tcspc_res = period_s / self.n_bins
        else:
            self.frequency = 0.0
            self.period_ns = 0.0
            self.tcspc_res = 0.0
        self.sync_rate = self.frequency
        self.time_ns = (np.arange(self.n_bins) + 0.5) * self.tcspc_res * 1e9
        self.dt_unit = 'ps' if self.dual_tdc else 'tac_bin'
        self.tags = {
            'PS_Magic': self.header['magic'],
            'PS_Version': self.header['version'],
            'PS_Datasets': self._names,
            'PS_PositionRange': self.pos_range,
            'PS_TacRange': self.tac_range,
            'PS_TacBias': self.attrs.get('/photons/TacBias'),
            'PS_TacChannel': self.attrs.get('/photons/TacChannel'),
            'PS_TimerFrequency': self.attrs.get('/photons/TimerFrequency'),
            'PS_DetectorGuid': self.attrs.get('/photons/DetectorGuid'),
            'PS_Created': self.attrs.get('Created'),
            'PS_Pixels': self.pixels,
            'PS_DecayResolution': self.n_bins,
            'PS_CalibrationSource': self.calib_source,
            'PS_DetectorChannels': self.detector_channels,
            'PS_SourceFile': self.path,
        }
        if self.verbose and announce:
            print(f'  Photonscore : {Path(self.path).name}')
            print(f'  Container   : {self.header["magic"]} v{self.header["version"]}')
            print(f'  Image       : {self.pixels} x {self.pixels} px  '
                  f'(from {self.pos_range} raw positions)')
            print(f'  TCSPC       : {self.n_bins} bins over {self.tac_range} dt units  '
                  f'({self.tcspc_res*1e12:.2f} ps/bin, {self.calib_source})')
            if self.dual_tdc:
                print('  Timing      : dual-TDC, dt = stop - start in ps '
                      '(range set from data)')
            if self.calib_source == 'unknown':
                print('  NOTE        : dt->time calibration unknown; '
                      'pass period_ns=<laser period ns> for a real time axis')
            print(' ')

    @property
    def pileup_fraction(self):
        return None

    def _ensure_streams(self):
        if self._streams is None:
            self._streams = pd.read_photons(self.path)
            if self.dual_tdc and 'dt' in self._streams:
                dt = np.asarray(self._streams['dt'])
                dt = dt[dt >= 0]
                if dt.size:
                    self.tac_range = int(dt.max()) + 1
                    self._parse_meta(self._period_ns, announce=False)
        return self._streams

    def _bin_positions(self, binning):
        s = self._ensure_streams()
        x = np.asarray(s['x']).astype(np.int64)
        y = np.asarray(s['y']).astype(np.int64)
        dt = np.asarray(s['dt']).astype(np.int64)
        n = min(x.shape[0], y.shape[0], dt.shape[0])
        x, y, dt = x[:n], y[:n], dt[:n]
        valid = ((x >= 0) & (x < self.pos_range) & (y >= 0) & (y < self.pos_range)
                 & (dt >= 0) & (dt < self.tac_range))
        x, y, dt = x[valid], y[valid], dt[valid]
        p = self.pixels
        xi = (x * p) // self.pos_range
        yi = (y * p) // self.pos_range
        di = (dt * self.n_bins) // self.tac_range
        if binning > 1:
            xi = xi // binning
            yi = yi // binning
            p = (p + binning - 1) // binning
        return xi, yi, di, p

    def pixel_stack(self, channel=None, binning=1):
        xi, yi, di, p = self._bin_positions(binning)
        b = self.n_bins
        if p == 0 or b == 0:
            return np.zeros((p, p, b), dtype=np.uint32)
        flat = (yi * p + xi) * b + di
        cube = np.bincount(flat, minlength=p * p * b).reshape(p, p, b).astype(np.uint32)
        self.n_y, self.n_x = cube.shape[0], cube.shape[1]
        self._total_photons = int(cube.sum())
        self.n_records = self._total_photons
        if self.verbose:
            print(f'  Built cube {cube.shape}  ({self._total_photons:,} photons)')
        return cube

    def raw_pixel_stack(self, channel=None, binning=1):
        return self.pixel_stack(channel=channel, binning=binning)

    def summed_decay(self, channel=None):
        s = self._ensure_streams()
        dt = np.asarray(s['dt']).astype(np.int64)
        dt = dt[(dt >= 0) & (dt < self.tac_range)]
        di = (dt * self.n_bins) // self.tac_range
        decay = np.bincount(di, minlength=self.n_bins).astype(float)
        self._total_photons = int(decay.sum())
        return decay[:self.n_bins]

    def intensity_image(self, channel=None, binning=1):
        xi, yi, _, p = self._bin_positions(binning)
        if p == 0:
            return np.zeros((0, 0), dtype=np.uint64)
        flat = yi * p + xi
        img = np.bincount(flat, minlength=p * p).reshape(p, p).astype(np.uint64)
        return img

def _metadata(ps, data):
    return {
        'frequency': ps.frequency,
        'tcspc_resolution': ps.tcspc_res,
        'shape': data.shape,
        'dims': ('Y', 'X', 'H'),
        'tags': ps.tags,
        'x_pixel_size': 0,
        'y_pixel_size': 0,
        'n_bins': ps.n_bins,
        'time_ns': ps.time_ns,
        'photon_channel': ps.photon_channel,
    }

def read_ps(path, binning=1, channel=None, verbose=False, pixels=512,
            n_bins=256, period_ns=None):
    ps = PSFile(path, verbose=verbose, pixels=pixels, n_bins=n_bins, period_ns=period_ns)
    data = ps.pixel_stack(channel=channel, binning=binning)
    metadata = _metadata(ps, data)
    return data, metadata

def get_flim_data(path, binning=1, channel=None, pixels=512, n_bins=256, period_ns=None):
    return read_ps(path, binning=binning, channel=channel, verbose=False,
                   pixels=pixels, n_bins=n_bins, period_ns=period_ns)

def get_intensity_image(path, binning=1, channel=None, pixels=512, n_bins=256):
    ps = PSFile(path, verbose=False, pixels=pixels, n_bins=n_bins)
    img = ps.intensity_image(channel=channel, binning=binning)
    metadata = _metadata(ps, np.empty(0))
    metadata['shape'] = (ps.n_y, ps.n_x, ps.n_bins)
    return img, metadata

def normalise_flim(flim):
    if flim is None:
        return None
    if flim.ndim == 5:
        return flim[0, :, :, 0, :]
    if flim.ndim == 4:
        return flim[0] if flim.shape[0] == 1 else flim[:, :, 0, :]
    if flim.ndim == 3:
        return flim
    return None

def create_time_axis(n_bins, tcspc_resolution):
    return np.arange(n_bins) * tcspc_resolution * 1e9
