import numpy as np
from lfdfiles import VistaIfli

def _to_hz(f):
    return float(f) if f > 1e5 else float(f) * 1e6

class ISSFdFlim:
    modality = 'frequency'

    def __init__(self, path, verbose=True, channel=None, **kwargs):
        self.path = str(path)
        self.verbose = verbose
        self.channel = channel
        with VistaIfli(self.path) as f:
            self.header = dict(f.header)
            self.axes = f.axes
            data = f.asarray()
        sizer, sizee, sizet, sizec, sizez, sizey, sizex, sizef, _ = data.shape
        self.n_x = int(sizex)
        self.n_y = int(sizey)
        self.n_z = max(int(sizez), 1)
        self.n_channels = max(int(sizec), 1)
        self.n_time_series = max(int(sizet), 1)
        self.n_freq = max(int(sizef), 1)
        self.n_positions = max(int(sizer), 1)
        self.n_spectral = max(int(sizee), 1)
        self._data = data
        self._mod_freqs = np.asarray(self.header.get('ModFrequency', ()), dtype=np.float32)
        self._ref_lifetime_ns = np.asarray(self.header.get('RefLifetime', ()), dtype=np.float32)
        self._ref_phasor = np.asarray(self.header.get('RefDCPhasor'))
        if self.verbose:
            print(f'  ISS FD-FLIM : {self.path}')
            print(f'  Size     : {self.n_x} x {self.n_y} px, {self.n_channels} ch, '
                  f'{self.n_freq} freq, {self.n_time_series} time series')
            print('  Frequency-domain data: loads into phasor analysis, no lifetime fitting')

    def _channel_index(self, channel):
        if channel is None:
            return 0
        c = int(channel)
        idx = c - 1 if c >= 1 else c
        return idx if 0 <= idx < self.n_channels else 0

    def phasor(self, channel=None, harmonic=0, calibrate=True, time_series=0):
        ci = self._channel_index(self.channel if channel is None else channel)
        fi = int(harmonic) if 0 <= int(harmonic) < self.n_freq else 0
        ts = int(time_series) if 0 <= int(time_series) < self.n_time_series else 0
        block = self._data[0, 0, ts, ci].sum(axis=0)
        dc = block[:, :, fi, 0]
        gx = block[:, :, fi, 1]
        gy = block[:, :, fi, 2]
        freq_hz = _to_hz(self._mod_freqs[fi]) if self._mod_freqs.size else 0.0
        if calibrate:
            gx, gy = self._calibrate(gx, gy, ci, fi, freq_hz)
        return dc.astype(float), gx.astype(float), gy.astype(float), freq_hz / 1e6

    def _calibrate(self, gx, gy, ci, fi, freq_hz):
        tau_ref = float(self._ref_lifetime_ns[ci]) * 1e-9 if ci < self._ref_lifetime_ns.size else 0.0
        rx = ry = 0.0
        if self._ref_phasor is not None and self._ref_phasor.ndim == 3 \
                and ci < self._ref_phasor.shape[0] and fi < self._ref_phasor.shape[1]:
            rx = float(self._ref_phasor[ci, fi, 1])
            ry = float(self._ref_phasor[ci, fi, 2])
        if tau_ref <= 0 or (rx == 0 and ry == 0) or freq_hz <= 0:
            return gx, gy
        w = 2.0 * np.pi * freq_hz
        denom = 1.0 + (w * tau_ref) ** 2
        gt, st = 1.0 / denom, (w * tau_ref) / denom
        scale = np.hypot(gt, st) / np.hypot(rx, ry)
        dphi = np.arctan2(st, gt) - np.arctan2(ry, rx)
        cos, sin = np.cos(dphi), np.sin(dphi)
        return scale * (gx * cos - gy * sin), scale * (gx * sin + gy * cos)

    def pixel_stack(self, *args, **kwargs):
        raise ValueError('ISS .ifli is frequency-domain (FD-FLIM): there is no '
                         'time-domain decay to fit. Load it into phasor analysis instead.')

    def summed_decay(self, *args, **kwargs):
        return self.pixel_stack()

    def raw_pixel_stack(self, *args, **kwargs):
        return self.pixel_stack()

def phasor_from_ifli(path, channel=None, harmonic=0, calibrate=True, verbose=False):
    return ISSFdFlim(path, verbose=verbose).phasor(channel=channel, harmonic=harmonic,
                                                   calibrate=calibrate)
