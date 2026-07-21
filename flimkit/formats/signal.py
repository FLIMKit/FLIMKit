import numpy as np
from pathlib import Path

def _import_io():
    try:
        import phasorpy.io as pio
    except ImportError:
        raise ImportError('phasorpy is required to read this format '
                          '(pip install phasorpy)')
    return pio

def _bin_cube(cube, binning):
    if binning <= 1:
        return cube
    ny, nx, nh = cube.shape
    ny2, nx2 = (ny // binning) * binning, (nx // binning) * binning
    cube = cube[:ny2, :nx2, :]
    return cube.reshape(ny2 // binning, binning, nx2 // binning, binning, nh).sum(axis=(1, 3))

class SignalReader:
    modality = 'time'

    def __init__(self, path, signal, label='signal', verbose=True, channel=None,
                 bin_width_ns=None, frequency_mhz=None):
        self.path = str(path)
        self.verbose = verbose
        self.label = label
        self.channel = channel
        dims = list(signal.dims)
        vals = np.asarray(signal.values)
        attrs = dict(getattr(signal, 'attrs', {}) or {})
        coords = getattr(signal, 'coords', {})
        self._dims = dims
        self._vals = vals
        self.tags = {f'phasorpy_{k}': v for k, v in attrs.items()}
        c_ax = dims.index('C') if 'C' in dims else None
        self.n_channels = int(vals.shape[c_ax]) if c_ax is not None else 1
        self._c_ax = c_ax
        h_ax = dims.index('H') if 'H' in dims else None
        if h_ax is None:
            raise ValueError(f'{label}: no TCSPC histogram (H) axis in {dims}')
        self.n_bins = int(vals.shape[h_ax])
        self.n_y = int(vals.shape[dims.index('Y')]) if 'Y' in dims else 0
        self.n_x = int(vals.shape[dims.index('X')]) if 'X' in dims else 0
        self.is_image = self.n_x > 0 and self.n_y > 0
        h_ns = None
        if 'H' in coords:
            h_ns = np.asarray(coords['H'].values, dtype=float)
        elif bin_width_ns is not None:
            h_ns = np.arange(self.n_bins) * float(bin_width_ns)
        if h_ns is not None and h_ns.size > 1:
            self.tcspc_res = float(np.median(np.diff(h_ns))) * 1e-9
            self.time_ns = h_ns
        else:
            self.tcspc_res = float(bin_width_ns) * 1e-9 if bin_width_ns else 0.0
            self.time_ns = np.arange(self.n_bins) * (self.tcspc_res * 1e9)
        freq_mhz = frequency_mhz if frequency_mhz is not None else attrs.get('frequency')
        self.sync_rate = float(freq_mhz) * 1e6 if freq_mhz else 0.0
        self.period_ns = (1e9 / self.sync_rate) if self.sync_rate else 0.0
        self.photon_channel = None
        self._total_photons = None
        self._cube_cache = {}
        if self.verbose:
            print(f'  {label} : {Path(self.path).name}')
            print(f'  TCSPC    : {self.n_bins} bins x {self.tcspc_res*1e12:.2f} ps')
            if self.sync_rate:
                print(f'  Laser    : {self.sync_rate/1e6:.3f} MHz  ({self.period_ns:.3f} ns)')
            if self.is_image:
                print(f'  Image    : {self.n_x} x {self.n_y} px, {self.n_channels} channel(s)')
            else:
                print(f'  Point    : no image (no Y/X axes), {self.n_channels} channel(s)')
            if self.tcspc_res <= 0:
                print(f'  WARNING  : {label} carries no time axis; pass bin_width_ns='
                      f'<ns> (and frequency_mhz=<MHz>) or fits will be meaningless')
            print(' ')

    def _ch_index(self, channel):
        if channel is None:
            return None
        c = int(channel)
        idx = c - 1 if c >= 1 else c
        return idx if 0 <= idx < self.n_channels else 0

    def _cube_for(self, channel):
        key = channel
        if key in self._cube_cache:
            return self._cube_cache[key]
        arr = self._vals
        dims = list(self._dims)
        if self._c_ax is not None:
            idx = self._ch_index(channel)
            if idx is None:
                sums = arr.sum(axis=tuple(i for i in range(arr.ndim) if i != self._c_ax))
                idx = int(np.argmax(sums))
                self.photon_channel = idx + 1
            arr = np.take(arr, idx, axis=self._c_ax)
            dims = [d for i, d in enumerate(dims) if i != self._c_ax]
        drop = tuple(i for i, d in enumerate(dims) if d not in ('Y', 'X', 'H'))
        if drop:
            arr = arr.sum(axis=drop)
            dims = [d for d in dims if d in ('Y', 'X', 'H')]
        order = [dims.index(d) for d in ('Y', 'X', 'H') if d in dims]
        cube = np.transpose(arr, order)
        self._cube_cache[key] = cube
        return cube

    def summed_decay(self, channel=None):
        cube = self._cube_for(self.channel if channel is None else channel)
        decay = cube.reshape(-1, self.n_bins).sum(axis=0).astype(float)
        self._total_photons = int(decay.sum())
        return decay

    def raw_pixel_stack(self, channel=None, binning=1):
        cube = self._cube_for(self.channel if channel is None else channel)
        if cube.ndim != 3:
            raise RuntimeError(f'{self.label} has no scan image; use summed_decay')
        if binning > 1:
            cube = _bin_cube(cube, binning)
        self._total_photons = int(cube.sum())
        return np.ascontiguousarray(cube).astype(np.uint32)

    def pixel_stack(self, channel=None, binning=1):
        return self.raw_pixel_stack(channel=channel, binning=binning).astype(float)

    def intensity_image(self, channel=None, binning=1):
        return self.raw_pixel_stack(channel=channel, binning=binning).sum(axis=2)

    @property
    def photons_per_pulse(self):
        return None

class _GohlkeSignalFile(SignalReader):
    _reader_name = None
    _label = None
    _reader_kwargs = {}

    def __init__(self, path, verbose=True, channel=None, **kwargs):
        pio = _import_io()
        fn = getattr(pio, self._reader_name, None)
        if fn is None:
            raise ImportError(f'{self._reader_name} is not available in this '
                              f'phasorpy version; upgrade phasorpy')
        signal = fn(str(path), **self._reader_kwargs)
        super().__init__(path, signal, label=self._label, verbose=verbose,
                         channel=channel, **kwargs)

class PQBinFile(_GohlkeSignalFile):
    _reader_name = 'signal_from_pqbin'
    _label = 'PicoQuant BIN'

class SimfcsBHFile(_GohlkeSignalFile):
    _reader_name = 'signal_from_bh'
    _label = 'SimFCS B&H'

class SimfcsBHZFile(_GohlkeSignalFile):
    _reader_name = 'signal_from_bhz'
    _label = 'SimFCS BHZ'

class ImspectorTIFFFile(_GohlkeSignalFile):
    _reader_name = 'signal_from_imspector_tiff'
    _label = 'ImSpector FLIM TIFF'

class VistaTdflimFile(_GohlkeSignalFile):
    _reader_name = 'signal_from_tdflim'
    _label = 'ISS Vista TDFLIM'
    _reader_kwargs = {'channel': None}

class FlimLabsSignalFile(_GohlkeSignalFile):
    _reader_name = 'signal_from_flimlabs_json'
    _label = 'FLIM LABS imaging'
