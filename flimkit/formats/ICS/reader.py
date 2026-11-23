import gzip
import numpy as np
from pathlib import Path

_TIME_UNITS = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'µs': 1e-6, 'ns': 1e-9, 'ps': 1e-12}
_LENGTH_UM = {'um': 1.0, 'µm': 1.0, 'micron': 1.0, 'microns': 1.0, 'nm': 1e-3, 'mm': 1e3, 'm': 1e6}
_TIME_NAMES = ('t', 'time', 'micro-time', 'microtime')
_CHANNEL_NAMES = ('ch', 'c', 'channel')

def _axis_name(name):
    n = name.lower()
    if n in _TIME_NAMES:
        return 't'
    if n in _CHANNEL_NAMES:
        return 'c'
    return n

def _read_header(path):
    with open(path, 'rb') as fh:
        first = fh.readline()
        if len(first) < 2 or first[:1] in (b'\n', b'\r') or first.rstrip(b'\r\n') != first[:1]:
            raise ValueError(f'Not an ICS file (missing separator line): {path}')
        sep = first[:1].decode('ascii', 'replace')
        if first[1:] not in (b'\n', b'\r\n'):
            raise ValueError(f'Unsupported ICS record separator {first[1:]!r} in {path}')
        entries = []
        ended = False
        while True:
            raw = fh.readline()
            if not raw:
                break
            line = raw.decode('latin-1').rstrip('\r\n')
            if line.split(sep)[0] == 'end':
                ended = True
                break
            if line.strip():
                entries.append(line.split(sep))
        offset = fh.tell()
    keys = {}
    for tokens in entries:
        keys.setdefault(' '.join(tokens[:2]).strip(), ' '.join(t for t in tokens[2:] if t.strip()))
    if 'ics_version' not in {t[0] for t in entries}:
        raise ValueError(f'Not an ICS file (no ics_version line): {path}')
    version = [t for t in entries if t[0] == 'ics_version'][0][1].strip()
    return entries, keys, version, offset, ended

def _dtype(bits, fmt, sign, big):
    if fmt == 'integer':
        kind = 'u' if sign == 'unsigned' else 'i'
    elif fmt == 'real':
        kind = 'f'
    else:
        raise ValueError(f'Unsupported ICS representation format {fmt!r}')
    nbytes = int(bits) // 8
    if (kind != 'f' and nbytes not in (1, 2, 4, 8)) or (kind == 'f' and nbytes not in (4, 8)):
        raise ValueError(f'Unsupported ICS sample size: {bits} bits {fmt}')
    return np.dtype(('>' if big else '<') + kind + str(nbytes))

def _bin_cube(cube, binning):
    ny, nx, nh = cube.shape
    ny2, nx2 = (ny // binning) * binning, (nx // binning) * binning
    cube = cube[:ny2, :nx2, :]
    return cube.reshape(ny2 // binning, binning, nx2 // binning, binning, nh).sum(axis=(1, 3))

def _laser_hz(meta):
    for keys in (('calibration', 'laserFrequencyHz'), ('timing', 'laserFrequency', 'value'),
                 ('acquisitionMetadata', 'timing', 'laserFrequency', 'value')):
        value = meta
        for k in keys:
            value = value.get(k) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None

def _rep_rate_from_history(history):
    import json
    chunks = sorted((t[1], '\t'.join(t[2:])) for t in history if len(t) > 2 and t[1].startswith('LeicaMetadata_'))
    if not chunks:
        return None
    try:
        return _laser_hz(json.loads(''.join(v for _k, v in chunks)))
    except ValueError:
        return None

def _rep_rate_from_sidecar(path):
    import json
    sidecar = Path(str(path) + '.json')
    if sidecar.exists() == False:
        return None
    try:
        return _laser_hz(json.loads(sidecar.read_text(encoding='utf-8')))
    except ValueError:
        return None

class ICSFile:
    modality = 'time'

    def __init__(self, path, verbose=True, channel=None, **kwargs):
        self.path = str(path)
        self.verbose = verbose
        self._channel = channel
        entries, keys, version, offset, ended = _read_header(self.path)
        self.ics_version = version
        self.history = [t for t in entries if t[0] == 'history']
        self.tags = dict(keys)
        if version.startswith('1'):
            data_path = self._find_ids()
        elif version.startswith('2'):
            if ended == False:
                raise ValueError(f'ICS header has no end line: {self.path}')
            data_path = Path(self.path)
        else:
            raise ValueError(f'Unsupported ICS version {version!r}')
        self._parse_layout(keys)
        self._data = self._open_data(data_path, offset if version.startswith('2') else 0, keys)
        self._total_photons = None
        self.photon_channel = 1 if self.n_channels == 1 else None
        self.is_image = True
        rate = _rep_rate_from_history(self.history) or _rep_rate_from_sidecar(self.path)
        self.sync_rate = rate or 0.0
        self.period_ns = 1e9 / rate if rate else 0.0
        self.n_records = 0
        self.n_sync = None
        if self.verbose:
            print(f'  ICS {version}  : {Path(self.path).name}')
            print(f'  Cube     : {self.n_y} x {self.n_x} x {self.n_bins} bins, {self.n_channels} channel(s)')
            print(f'  TCSPC    : {self.n_bins} bins x {self.tcspc_res*1e12:.2f} ps')
            print(' ')

    def _find_ids(self):
        p = Path(self.path)
        for cand in (p.with_suffix('.ids'), p.with_suffix('.IDS')):
            if cand.exists():
                return cand
        raise ValueError(f'ICS v1 data file ({p.with_suffix(".ids").name}) not found next to {p.name}')

    def _parse_layout(self, keys):
        if 'layout order' not in keys or 'layout sizes' not in keys:
            raise ValueError(f'ICS header lacks layout order/sizes: {self.path}')
        order = [_axis_name(a) for a in keys['layout order'].split()]
        sizes = [int(s) for s in keys['layout sizes'].split()]
        if len(order) != len(sizes) or order[0] != 'bits':
            raise ValueError(f'Unsupported ICS layout order {order} / sizes {sizes}')
        self._bits = sizes[0]
        axes = dict(zip(order[1:], sizes[1:]))
        self._order = order[1:]
        self._sizes = sizes[1:]
        for need in ('x', 'y', 't'):
            if need not in axes:
                raise ValueError(f'ICS layout has no {need!r} axis (order: {order[1:]}); cannot read as TCSPC cube')
        extra = [a for a in axes if a not in ('x', 'y', 't', 'c')]
        for a in extra:
            if axes[a] != 1:
                raise ValueError(f'ICS axis {a!r} of size {axes[a]} is not supported (only x, y, t and a channel axis)')
        self.n_x, self.n_y, self.n_bins = axes['x'], axes['y'], axes['t']
        self.n_channels = axes.get('c', 1)
        scale = keys.get('parameter scale', '').split()
        units = keys.get('parameter units', '').split()
        pos = {a: i for i, a in enumerate(order)}
        def _val(axis, table):
            try:
                v = float(scale[pos[axis]])
                u = units[pos[axis]].lower() if len(units) > pos[axis] else ''
                return v, table.get(u)
            except (KeyError, IndexError, ValueError):
                return None, None
        tv, tf = _val('t', _TIME_UNITS)
        if tv is None or tv <= 0:
            raise ValueError('ICS header has no usable time step in "parameter scale"')
        self.tcspc_res = tv * (tf if tf is not None else 1.0)
        self.time_ns = (np.arange(self.n_bins) + 0.5) * self.tcspc_res * 1e9
        px, pf = _val('x', _LENGTH_UM)
        self.pixel_size_um = px * pf if px is not None and pf is not None and px > 0 else None

    def _open_data(self, data_path, offset, keys):
        fmt = keys.get('representation format', 'integer').strip()
        sign = keys.get('representation sign', 'unsigned').strip()
        comp = keys.get('representation compression', 'uncompressed').strip()
        bo = [int(b) for b in keys.get('representation byte_order', '1').split()]
        dt = _dtype(self._bits, fmt, sign, len(bo) > 1 and bo[0] > bo[-1])
        shape = tuple(reversed(self._sizes))
        if comp == 'uncompressed':
            raw = np.memmap(data_path, dtype=dt, mode='r', offset=offset, shape=shape)
        elif comp == 'gzip':
            with open(data_path, 'rb') as fh:
                fh.seek(offset)
                buf = gzip.decompress(fh.read())
            raw = np.frombuffer(buf, dtype=dt)
            if raw.size != int(np.prod(shape)):
                raise ValueError(f'ICS gzip data holds {raw.size} samples, header expects {int(np.prod(shape))}')
            raw = raw.reshape(shape)
        else:
            raise ValueError(f'Unsupported ICS compression {comp!r}')
        names = list(reversed(self._order))
        for a in [a for a in names if a not in ('x', 'y', 't', 'c')]:
            raw = raw.reshape(tuple(s for s, n in zip(raw.shape, names) if n != a))
            names.remove(a)
        if 'c' not in names:
            raw = raw[np.newaxis]
            names = ['c'] + names
        return raw.transpose([names.index(a) for a in ('c', 'y', 'x', 't')])

    def close(self):
        self._data = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ch_index(self, channel):
        if channel is None:
            channel = self._channel
        if channel is None or int(channel) < 1:
            if self.photon_channel is None:
                totals = [float(self._data[i].sum(dtype=np.float64)) for i in range(self.n_channels)]
                self.photon_channel = int(np.argmax(totals)) + 1
            return self.photon_channel - 1
        c = int(channel)
        if c > self.n_channels:
            raise ValueError(f'Channel {channel} out of range 1..{self.n_channels}')
        return c - 1

    def pixel_stack(self, channel=None, binning=1):
        cube = self._data[self._ch_index(channel)]
        cube = np.clip(cube, 0, None) if cube.dtype.kind in 'if' else cube
        if binning > 1:
            cube = _bin_cube(cube, binning)
        self.n_y, self.n_x = cube.shape[0], cube.shape[1]
        out = np.ascontiguousarray(cube).astype(np.uint32)
        self._total_photons = int(out.sum(dtype=np.uint64))
        self.n_records = self._total_photons
        if self.verbose:
            print(f'  Built cube {out.shape}  ({self._total_photons:,} photons)')
        return out

    def raw_pixel_stack(self, channel=None, binning=1):
        return self.pixel_stack(channel=channel, binning=binning)

    def summed_decay(self, channel=None):
        decay = self._data[self._ch_index(channel)].sum(axis=(0, 1), dtype=np.float64)
        self._total_photons = int(decay.sum())
        return decay

    @property
    def photons_per_pulse(self):
        return None

def read_ics(path, binning=1, channel=None, verbose=False):
    ics = ICSFile(path, verbose=verbose, channel=channel)
    data = ics.pixel_stack(channel=channel, binning=binning)
    metadata = {
        'frequency': ics.sync_rate,
        'tcspc_resolution': ics.tcspc_res,
        'shape': data.shape,
        'dims': ('Y', 'X', 'H'),
        'tags': ics.tags,
        'x_pixel_size': ics.pixel_size_um or 0,
        'y_pixel_size': ics.pixel_size_um or 0,
        'n_bins': ics.n_bins,
        'time_ns': ics.time_ns,
        'photon_channel': ics.photon_channel,
    }
    ics.close()
    return data, metadata
