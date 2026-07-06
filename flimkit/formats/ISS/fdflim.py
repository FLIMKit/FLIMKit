import struct
import numpy as np

_MAGIC = b'VistaFLImage'
_HEADER_BYTES = 1024

def _read_header(buf):
    h = {}
    h['signature'] = bytes(buf[0:12])
    h['file_version'] = buf[12]
    h['is_spectral_lifetime'] = bool(buf[13])
    h['is_spectral_phasor'] = bool(buf[14])
    h['is_spectral_intensity'] = bool(buf[15])
    h['histogram_resolution'] = struct.unpack_from('<i', buf, 16)[0]
    h['channel_bits'] = struct.unpack_from('<I', buf, 20)[0]
    h['compression'] = buf[24]
    dims = struct.unpack_from('<5H', buf, 25)
    h['n_x'], h['n_y'], h['n_z'], h['n_channels'], h['n_time_series'] = dims
    h['boundaries'] = struct.unpack_from('<6f', buf, 35)
    h['coord_type'] = buf[59]
    h['pixel_sampling_time'] = struct.unpack_from('<f', buf, 60)[0]
    h['pixel_interval'] = struct.unpack_from('<f', buf, 64)[0]
    h['line_interval'] = struct.unpack_from('<f', buf, 68)[0]
    h['frame_interval'] = struct.unpack_from('<f', buf, 72)[0]
    h['n_freq'] = struct.unpack_from('<i', buf, 76)[0]
    h['cross_cor_freq'] = struct.unpack_from('<f', buf, 80)[0]
    h['frame_repeat_count'] = struct.unpack_from('<H', buf, 84)[0]
    h['phasor_data_offset'] = struct.unpack_from('<Q', buf, 86)[0]
    h['time_tag_offset'] = struct.unpack_from('<Q', buf, 94)[0]
    h['mod_freq_list_offset'] = struct.unpack_from('<Q', buf, 102)[0]
    h['ref_lifetime_offset'] = struct.unpack_from('<Q', buf, 110)[0]
    h['ref_dc_phasor_offset'] = struct.unpack_from('<Q', buf, 118)[0]
    return h

def _to_hz(f):
    # modulation frequencies are stored as plain floats; accept MHz or Hz
    return float(f) if f > 1e5 else float(f) * 1e6

class ISSFdFlim:
    modality = 'frequency'

    def __init__(self, path, verbose=True, channel=None, **kwargs):
        self.path = str(path)
        self.verbose = verbose
        self.channel = channel
        with open(self.path, 'rb') as fh:
            self._raw = fh.read()
        if self._raw[:len(_MAGIC)] != _MAGIC:
            raise ValueError(f'Not an ISS .ifli file (magic {_MAGIC!r} not found): {self.path}')
        self.header = _read_header(self._raw[:_HEADER_BYTES])
        h = self.header
        self.n_x = int(h['n_x'])
        self.n_y = int(h['n_y'])
        self.n_z = max(int(h['n_z']), 1)
        self.n_channels = max(int(h['n_channels']), 1)
        self.n_time_series = max(int(h['n_time_series']), 1)
        self.n_freq = max(int(h['n_freq']), 1)
        self._mod_freqs = self._read_floats(h['mod_freq_list_offset'], self.n_freq)
        self._ref_lifetime_ns = self._read_floats(h['ref_lifetime_offset'], 1)
        self._ref_phasor = self._read_floats(h['ref_dc_phasor_offset'], self.n_freq * 3)
        data_off = int(h['phasor_data_offset']) or _HEADER_BYTES
        n = self.n_time_series * self.n_channels * self.n_z * self.n_y * self.n_x * self.n_freq * 3
        flat = np.frombuffer(self._raw, dtype='<f4', count=n, offset=data_off)
        self._dc = flat.reshape(self.n_time_series, self.n_channels, self.n_z,
                                self.n_y, self.n_x, self.n_freq, 3)
        if self.verbose:
            print(f'  ISS FD-FLIM : {self.path}')
            print(f'  Size     : {self.n_x} x {self.n_y} px, {self.n_channels} ch, '
                  f'{self.n_freq} freq, {self.n_time_series} time series')
            print('  Frequency-domain data: loads into phasor analysis, no lifetime fitting')

    def _read_floats(self, offset, count):
        if not offset or offset + count * 4 > len(self._raw):
            return np.zeros(count, dtype=np.float32)
        return np.frombuffer(self._raw, dtype='<f4', count=count, offset=int(offset)).copy()

    def _channel_index(self, channel):
        if channel is None:
            return 0
        c = int(channel)
        idx = c - 1 if c >= 1 else c
        return idx if 0 <= idx < self.n_channels else 0

    def phasor(self, channel=None, harmonic=0, calibrate=True, time_series=0):
        ci = self._channel_index(self.channel if channel is None else channel)
        fi = int(harmonic) if 0 <= int(harmonic) < self.n_freq else 0
        block = self._dc[int(time_series), ci].sum(axis=0) 
        dc = block[:, :, fi, 0]
        gx = block[:, :, fi, 1]
        gy = block[:, :, fi, 2]
        freq_hz = _to_hz(self._mod_freqs[fi])
        if calibrate:
            gx, gy = self._calibrate(gx, gy, fi, freq_hz)
        return dc.astype(float), gx.astype(float), gy.astype(float), freq_hz / 1e6

    def _calibrate(self, gx, gy, fi, freq_hz):
        tau_ref = float(self._ref_lifetime_ns[0]) * 1e-9
        rx, ry = float(self._ref_phasor[fi * 3 + 1]), float(self._ref_phasor[fi * 3 + 2])
        if tau_ref <= 0 or (rx == 0 and ry == 0):
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
