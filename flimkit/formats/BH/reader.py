import numpy as np
from pathlib import Path

_LASER_RATES_HZ = (20e6, 50e6, 80e6)
_DECAY_CONTENT = (0x10, 0x60)
_INT_CONTENT = 0xa0

def _load_sdtfile():
    try:
        import sdtfile
    except ImportError:
        raise ImportError('sdtfile is required to read Becker & Hickl .sdt files '
                          '(pip install sdtfile)')
    return sdtfile

def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length() if n > 1 else 1

def _reshape_cube(arr, image_x, image_y, adc_re):
    flat = np.ascontiguousarray(arr).reshape(-1)
    per_pixel = adc_re if adc_re > 0 else 1
    n_pixels = flat.size // per_pixel
    if n_pixels == image_x * image_y:
        out = flat[:image_y * image_x * per_pixel].reshape(image_y, image_x, per_pixel)
    else:
        xpad, ypad = _next_pow2(image_x), _next_pow2(image_y)
        out = flat[:ypad * xpad * per_pixel].reshape(ypad, xpad, per_pixel)
        out = out[:image_y, :image_x, :]
    return np.ascontiguousarray(out)

def _reshape_intensity(arr, image_x, image_y):
    flat = np.ascontiguousarray(arr).reshape(-1)
    if flat.size == image_x * image_y:
        out = flat.reshape(image_y, image_x)
    else:
        xpad, ypad = _next_pow2(image_x), _next_pow2(image_y)
        out = flat[:ypad * xpad].reshape(ypad, xpad)[:image_y, :image_x]
    return np.ascontiguousarray(out)

def _decode_str(value):
    if isinstance(value, bytes):
        return value.split(b'\x00', 1)[0].decode('ascii', 'replace').strip()
    return str(value).strip().strip('\x04').strip()

class BHFile:
    def __init__(self, path, verbose=True, channel=None, sync_rate=None):
        self.path = str(path)
        self.verbose = verbose
        self._sync_override = float(sync_rate) if sync_rate else None
        sdtfile = _load_sdtfile()
        self._sdt = sdtfile.SdtFile(self.path)
        headers = self._sdt.block_headers
        self._decay_idx = [i for i, b in enumerate(headers)
                           if (int(b['block_type']) & 0x00f0) in _DECAY_CONTENT]
        self._int_idx = [i for i, b in enumerate(headers)
                         if (int(b['block_type']) & 0x00f0) == _INT_CONTENT]
        if not self._decay_idx:
            raise ValueError(f'No FLIM image/decay blocks found in {self.path}')
        self._cube_cache = {}
        self.n_x = None
        self.n_y = None
        self._total_photons = None
        self._parse_meta(channel)

    def _measure_info(self, data_idx):
        mi = self._sdt.measure_info
        no = int(self._sdt.block_headers[data_idx]['meas_desc_block_no'])
        return mi[no] if no < len(mi) else mi[0]

    def close(self):
        try:
            self._sdt.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _parse_meta(self, channel):
        self.n_channels = len(self._decay_idx)
        if channel is not None:
            self.photon_channel = int(channel)
        else:
            self.photon_channel = 1 if self.n_channels == 1 else None
        mi = self._measure_info(self._decay_idx[0])
        self.module_type = _decode_str(mi['mod_type'])
        adc_re = int(mi['adc_re'])
        self.n_bins = adc_re if adc_re != 0 else 65536
        tac_r = float(mi['tac_r'])
        tac_g = int(mi['tac_g'])
        if tac_g and self.n_bins:
            self.tcspc_res = tac_r / (tac_g * self.n_bins)
        else:
            self.tcspc_res = 0.0
        self.time_ns = (np.arange(self.n_bins) + 0.5) * self.tcspc_res * 1e9
        stop = mi['StopInfo']
        self.min_sync_rate = float(stop['min_sync_rate'])
        self.max_sync_rate = float(stop['max_sync_rate'])
        if self._sync_override and self._sync_override > 0:
            self.sync_rate = self._sync_override
            self.sync_source = 'user'
        else:
            sync = self.max_sync_rate if self.max_sync_rate > 0 else self.min_sync_rate
            self.sync_rate = float(sync) if sync and sync > 0 else 0.0
            self.sync_source = 'measured'
        self.period_ns = (1e9 / self.sync_rate) if self.sync_rate > 0 else 0.0
        self.n_records = 0
        info = self._sdt.info
        self.tags = {
            'BH_ID': _decode_str(getattr(info, 'id', '')),
            'BH_Title': _decode_str(getattr(info, 'title', '')),
            'BH_Date': _decode_str(getattr(info, 'date', '')),
            'BH_Time': _decode_str(getattr(info, 'time', '')),
            'BH_ModuleType': self.module_type,
            'BH_MeasMode': int(mi['meas_mode']),
            'BH_ADCResolution': self.n_bins,
            'BH_TACRange_s': tac_r,
            'BH_TACGain': tac_g,
            'BH_ImageX': int(mi['image_x']),
            'BH_ImageY': int(mi['image_y']),
            'BH_CollectionTime_s': float(mi['col_t']),
            'BH_MinSyncRate_Hz': self.min_sync_rate,
            'BH_MaxSyncRate_Hz': self.max_sync_rate,
            'BH_SyncRate_Hz': self.sync_rate,
            'BH_SyncRateSource': self.sync_source,
            'BH_Channels': self.n_channels,
        }
        if self.verbose:
            print(f"  B&H SPC  : {Path(self.path).name}")
            print(f"  Module   : {self.module_type}  ({self.n_channels} channel(s))")
            print(f"  TCSPC    : {self.n_bins} bins x {self.tcspc_res*1e12:.2f} ps")
            sync_mhz = self.sync_rate / 1e6
            print(f"  Sync     : {sync_mhz:.2f} MHz  (period {self.period_ns:.3f} ns, {self.sync_source})")
            if (self.sync_source == 'measured' and self.min_sync_rate > 0
                    and self.max_sync_rate > 0
                    and abs(self.max_sync_rate - self.min_sync_rate) > 0.01 * self.max_sync_rate):
                print(f"  WARNING: min/max sync differ "
                      f"({self.min_sync_rate/1e6:.3f} vs {self.max_sync_rate/1e6:.3f} MHz), using max")
            if self.sync_rate > 0:
                nearest = min(_LASER_RATES_HZ, key=lambda r: abs(r - self.sync_rate))
                if abs(nearest - self.sync_rate) > 0.02 * nearest:
                    print(f"  NOTE     : {sync_mhz:.2f} MHz is non-standard "
                          f"(period {self.period_ns:.3f} ns from measured sync); "
                          f"pass sync_rate=<Hz> to override if the laser differs")
            print(' ')

    @property
    def pileup_fraction(self):
        return None

    def _norm_channel(self, channel):
        if channel is None or int(channel) < 1:
            return self._ensure_photon_channel()
        return int(channel)

    def _ensure_photon_channel(self):
        if self.photon_channel is None:
            self.photon_channel = self._select_photon_channel() if self.n_channels > 1 else 1
        return self.photon_channel

    def _select_photon_channel(self):
        best_idx, best_sum, total_all = 1, -1, 0
        for idx in range(1, self.n_channels + 1):
            cube = self._decode_cube_for(idx)
            total = int(cube.sum())
            total_all += total
            if total > best_sum:
                self._cube_cache.pop(best_idx, None)
                best_idx, best_sum = idx, total
                self._cube_cache[idx] = cube
        if self.verbose:
            share = (100.0 * best_sum / total_all) if total_all > 0 else 0.0
            print(f"  Channel  : auto-selected {best_idx}/{self.n_channels} "
                  f"({best_sum:,} photons, {share:.1f}% of total)")
        return best_idx

    def _decode_cube_for(self, idx):
        data_idx = self._decay_idx[idx - 1]
        mi = self._measure_info(data_idx)
        return _reshape_cube(self._sdt.data[data_idx], int(mi['image_x']),
                             int(mi['image_y']), self.n_bins)

    def _decode_cube(self, channel):
        idx = self._norm_channel(channel)
        if idx in self._cube_cache:
            return self._cube_cache[idx]
        if idx < 1 or idx > self.n_channels:
            raise ValueError(f'Channel {channel} out of range 1..{self.n_channels}')
        cube = self._decode_cube_for(idx)
        self._cube_cache[idx] = cube
        return cube

    def pixel_stack(self, channel=None, binning=1):
        cube = self._decode_cube(channel)
        if binning > 1:
            cube = _bin_cube(cube, binning)
        self.n_y, self.n_x = cube.shape[0], cube.shape[1]
        self._total_photons = int(cube.sum())
        self.n_records = self._total_photons
        if self.verbose:
            print(f'  Built cube {cube.shape}  ({self._total_photons:,} photons)')
        return cube.astype(np.uint32)

    def raw_pixel_stack(self, channel=None, binning=1):
        return self.pixel_stack(channel=channel, binning=binning)

    def summed_decay(self, channel=None):
        cube = self._decode_cube(channel)
        return cube.sum(axis=(0, 1)).astype(float)

    def intensity_image(self, channel=None, binning=1):
        idx = self._norm_channel(channel)
        decay_idx = self._decay_idx[idx - 1]
        md = int(self._sdt.block_headers[decay_idx]['meas_desc_block_no'])
        match = [i for i in self._int_idx
                 if int(self._sdt.block_headers[i]['meas_desc_block_no']) == md]
        if match:
            mi = self._measure_info(match[0])
            img = _reshape_intensity(self._sdt.data[match[0]],
                                     int(mi['image_x']), int(mi['image_y'])).astype(np.uint64)
        else:
            return self.pixel_stack(channel=idx, binning=binning).sum(axis=2)
        if binning > 1:
            ny, nx = img.shape
            ny2, nx2 = (ny // binning) * binning, (nx // binning) * binning
            img = img[:ny2, :nx2].reshape(ny2 // binning, binning,
                                          nx2 // binning, binning).sum(axis=(1, 3))
        return img

def _bin_cube(cube, binning):
    if binning <= 1:
        return cube
    ny, nx, nh = cube.shape
    ny2, nx2 = (ny // binning) * binning, (nx // binning) * binning
    cube = cube[:ny2, :nx2, :]
    return cube.reshape(ny2 // binning, binning, nx2 // binning, binning, nh).sum(axis=(1, 3))

def _metadata(bh, data):
    return {
        'frequency': bh.sync_rate,
        'tcspc_resolution': bh.tcspc_res,
        'shape': data.shape,
        'dims': ('Y', 'X', 'H'),
        'tags': bh.tags,
        'x_pixel_size': 0,
        'y_pixel_size': 0,
        'n_bins': bh.n_bins,
        'time_ns': bh.time_ns,
        'photon_channel': bh.photon_channel,
    }

def read_bh(path, binning=1, channel=None, verbose=False, sync_rate=None):
    bh = BHFile(path, verbose=verbose, channel=channel, sync_rate=sync_rate)
    data = bh.pixel_stack(channel=channel, binning=binning)
    metadata = _metadata(bh, data)
    bh.close()
    return data, metadata

def get_flim_data(path, binning=1, channel=None, sync_rate=None):
    return read_bh(path, binning=binning, channel=channel, verbose=False, sync_rate=sync_rate)

def get_intensity_image(path, binning=1, channel=None, sync_rate=None):
    bh = BHFile(path, verbose=False, channel=channel, sync_rate=sync_rate)
    img = bh.intensity_image(channel=channel, binning=binning)
    cube_shape = (bh.n_y, bh.n_x, bh.n_bins)
    metadata = _metadata(bh, np.empty(0))
    metadata['shape'] = cube_shape
    bh.close()
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
