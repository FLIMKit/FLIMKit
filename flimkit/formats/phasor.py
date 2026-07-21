import numpy as np
from pathlib import Path

def _import_io():
    try:
        import phasorpy.io as pio
    except ImportError:
        raise ImportError('phasorpy is required to read this format '
                          '(pip install phasorpy)')
    return pio

class PhasorReader:
    modality = 'frequency'

    def __init__(self, path, mean, real, imag, attrs, label='phasor',
                 verbose=True, channel=None, frequency_mhz=None):
        self.path = str(path)
        self.verbose = verbose
        self.label = label
        self.channel = channel
        self._mean = np.asarray(mean)
        self._real = np.asarray(real)
        self._imag = np.asarray(imag)
        attrs = dict(attrs or {})
        self._dims = tuple(attrs.get('dims') or ())
        self._harmonics = attrs.get('harmonic')
        freq = frequency_mhz if frequency_mhz is not None else attrs.get('frequency')
        self.frequency_mhz = float(freq) if freq else 0.0
        self.tags = {f'phasorpy_{k}': v for k, v in attrs.items()}
        dims = self._dims
        self.n_channels = int(self._mean.shape[dims.index('C')]) if 'C' in dims else 1
        self.n_y = int(self._mean.shape[dims.index('Y')]) if 'Y' in dims else 0
        self.n_x = int(self._mean.shape[dims.index('X')]) if 'X' in dims else 0
        self.is_image = self.n_x > 0 and self.n_y > 0
        self.n_harmonics = len(self._harmonics) if isinstance(self._harmonics, (list, tuple)) else 1
        if self.verbose:
            print(f'  {label} : {Path(self.path).name}')
            if self.is_image:
                print(f'  Size     : {self.n_x} x {self.n_y} px, {self.n_channels} ch, '
                      f'{self.n_harmonics} harmonic(s)  dims={"".join(dims)}')
            else:
                print(f'  Point    : no image (no Y/X axes), {self.n_channels} ch, '
                      f'{self.n_harmonics} harmonic(s)  dims={"".join(dims)}')
            if self.frequency_mhz:
                print(f'  Frequency: {self.frequency_mhz:.3f} MHz')
            else:
                print('  WARNING  : file carries no frequency; pass frequency_mhz=<MHz> '
                      'for the universal circle / lifetime overlay')
            print('  Frequency-domain data: loads into phasor analysis, no lifetime fitting')
            print(' ')

    def _ch_index(self, channel):
        if channel is None:
            return None
        c = int(channel)
        idx = c - 1 if c >= 1 else c
        return idx if 0 <= idx < self.n_channels else 0

    def _auto_channel(self):
        dims = list(self._dims)
        if 'C' not in dims:
            return 0
        ci = dims.index('C')
        sums = self._mean.sum(axis=tuple(i for i in range(self._mean.ndim) if i != ci))
        return int(np.argmax(sums))

    def _reduce(self, arr, dims, ch_idx):
        dims = list(dims)
        if arr.ndim != len(dims):
            dims = dims[-arr.ndim:]
        if 'C' in dims:
            ci = dims.index('C')
            arr = np.take(arr, ch_idx, axis=ci)
            dims = [d for i, d in enumerate(dims) if i != ci]
        while len(dims) > 2:
            extra = next(i for i, d in enumerate(dims) if d not in ('Y', 'X'))
            arr = np.take(arr, 0, axis=extra)
            dims = [d for i, d in enumerate(dims) if i != extra]
        order = [dims.index(d) for d in ('Y', 'X') if d in dims]
        return np.transpose(arr, order) if len(order) == 2 else arr

    def phasor(self, channel=None, harmonic=0, calibrate=True, **kwargs):
        ch = self.channel if channel is None else channel
        ch_idx = self._ch_index(ch)
        if ch_idx is None:
            ch_idx = self._auto_channel()
        real, imag = self._real, self._imag
        if isinstance(self._harmonics, (list, tuple)):
            hi = int(harmonic) if 0 <= int(harmonic) < len(self._harmonics) else 0
            real, imag = real[hi], imag[hi]
        mean2 = self._reduce(self._mean, self._dims, ch_idx)
        real2 = self._reduce(real, self._dims, ch_idx)
        imag2 = self._reduce(imag, self._dims, ch_idx)
        return (mean2.astype(float), real2.astype(float), imag2.astype(float),
                self.frequency_mhz)

    def pixel_stack(self, *args, **kwargs):
        raise ValueError(f'{self.label} is frequency-domain phasor data: there is no '
                         'time-domain decay to fit. Load it into phasor analysis instead.')

    def summed_decay(self, *args, **kwargs):
        return self.pixel_stack()

    def raw_pixel_stack(self, *args, **kwargs):
        return self.pixel_stack()

class _GohlkePhasorFile(PhasorReader):
    _reader_name = None
    _label = None
    _reader_kwargs = {}

    def __init__(self, path, verbose=True, channel=None, **kwargs):
        pio = _import_io()
        fn = getattr(pio, self._reader_name, None)
        if fn is None:
            raise ImportError(f'{self._reader_name} is not available in this '
                              f'phasorpy version; upgrade phasorpy')
        mean, real, imag, attrs = fn(str(path), **self._reader_kwargs)
        super().__init__(path, mean, real, imag, attrs, label=self._label,
                         verbose=verbose, channel=channel, **kwargs)

class SimfcsReferencedFile(_GohlkePhasorFile):
    _reader_name = 'phasor_from_simfcs_referenced'
    _label = 'SimFCS referenced'

class OmeTiffPhasorFile(_GohlkePhasorFile):
    _reader_name = 'phasor_from_ometiff'
    _label = 'PhasorPy OME-TIFF'

class FlimLabsPhasorFile(_GohlkePhasorFile):
    _reader_name = 'phasor_from_flimlabs_json'
    _label = 'FLIM LABS phasor'
