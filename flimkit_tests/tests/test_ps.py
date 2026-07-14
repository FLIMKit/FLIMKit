import numpy as np
import pytest
from flimkit.formats import FLIMFile, detect_format
from flimkit.formats.flim_file import file_modality
from flimkit.formats.PS.reader import PSFile, read_ps, get_intensity_image
from photonsfile import read_photons, has_dual_tdc
import photonsfile._d7 as pd

MAGIC = b'D7 Photons Data'

def _uv(n):
    out = bytearray()
    while True:
        b = n & 0x7f
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)

def _zz(n):
    return (n << 1) if n >= 0 else ((-n << 1) - 1)

def _ld(tag, payload):
    return bytes([tag]) + _uv(len(payload)) + payload

def _packed_deltas(values):
    vals = np.asarray(values, dtype=np.int64)
    out = bytearray()
    for d in np.diff(vals):
        out += _uv(_zz(int(d)))
    return int(vals[0]), bytes(out)

def _data_block(did, values):
    seed, packed = _packed_deltas(values)
    msg = bytes([0x08]) + _uv(did) + bytes([0x18]) + _uv(_zz(seed)) + _ld(0x22, packed)
    return _ld(0x12, msg)

def _header(datasets):
    msg = bytearray(_ld(0x0a, MAGIC) + bytes([0x10]) + _uv(1))
    for name, tc, _ in datasets:
        di = _ld(0x0a, name.encode()) + bytes([0x10]) + _uv(tc)
        msg += _ld(0x3a, di)
    return _ld(0x0a, bytes(msg))

def _datainfo(did, offset):
    return _ld(0x0a, bytes([0x08]) + _uv(did) + bytes([0x10]) + _uv(offset))

def _attr(key, val):
    return _ld(0x12, _ld(0x0a, key.encode()) + _ld(0x12, val.encode()))

def _write_datasets(path, datasets, attrs):
    logical = bytearray(_header(datasets))
    offsets = {}
    for did, (name, tc, values) in enumerate(datasets):
        offsets[did] = len(logical)
        logical += _data_block(did, values)
    index_off = len(logical)
    body = bytearray()
    for did in range(len(datasets)):
        body += _datainfo(did, offsets[did] + 2)
    for k, v in attrs.items():
        body += _attr(k, v)
    logical += _ld(0x1a, bytes(body))
    epi = bytes([0x08]) + _uv(index_off + 2) + _ld(0x12, b'End of D7 Photons Data File')
    logical += _ld(0x22, epi)
    path.write_bytes(b'\x00\x00' + bytes(logical))
    return path

_ATTRS = {'/photons/PositionBits': '12', '/photons/TacBits': '12',
          '/photons/TacChannel': '12.3', '/photons/TacBias': '1600'}

def _sample(path):
    streams = {
        'x':  [100, 4000, 50, 2048, 300],
        'y':  [3000, 10, 2050, 100, 4095],
        'dt': [500, 1500, 2500, 3500, 100],
        'ms': [0, 10, 25, 25, 60],
    }
    _write_datasets(path, [
        ('/photons/x', 5, streams['x']),
        ('/photons/y', 5, streams['y']),
        ('/photons/dt', 5, streams['dt']),
        ('/photons/ms', 7, streams['ms']),
    ], _ATTRS)
    return streams

def test_detect_and_dispatch(tmp_path):
    path = tmp_path / 'sample.photons'
    _sample(path)
    assert detect_format(str(path)) == 'ps'
    assert file_modality(str(path)) == 'time'
    ps = FLIMFile(str(path), verbose=False)
    assert isinstance(ps, PSFile)

def test_roundtrip_decode(tmp_path):
    path = tmp_path / 'sample.photons'
    streams = _sample(path)
    out = read_photons(str(path))
    for k in ('x', 'y', 'dt', 'ms'):
        assert np.array_equal(out[k], np.array(streams[k]))
    assert out['x'].dtype == np.uint16
    assert out['ms'].dtype == np.uint64

def test_attributes_and_calibration(tmp_path):
    path = tmp_path / 'sample.photons'
    _sample(path)
    ps = PSFile(str(path), verbose=False, n_bins=256)
    assert ps.pos_range == 4096
    assert ps.tac_range == 4096
    assert ps.calib_source == 'TacChannel'
    period_s = 4096 * 12.3e-12
    assert ps.period_ns == pytest.approx(period_s * 1e9)
    assert ps.tcspc_res == pytest.approx(period_s / 256)
    assert ps.frequency == pytest.approx(1.0 / period_s)

def test_period_ns_override(tmp_path):
    path = tmp_path / 'sample.photons'
    _sample(path)
    ps = PSFile(str(path), verbose=False, n_bins=100, period_ns=50.0)
    assert ps.calib_source == 'user'
    assert ps.tcspc_res == pytest.approx(50e-9 / 100)

def test_cube_and_decay(tmp_path):
    path = tmp_path / 'sample.photons'
    streams = _sample(path)
    ps = PSFile(str(path), verbose=False, pixels=2, n_bins=4)
    cube = ps.pixel_stack()
    assert cube.shape == (2, 2, 4)
    assert cube.dtype == np.uint32
    assert int(cube.sum()) == 5
    x = np.array(streams['x']); y = np.array(streams['y']); dt = np.array(streams['dt'])
    xi = (x * 2) // 4096; yi = (y * 2) // 4096; di = (dt * 4) // 4096
    expect = np.zeros((2, 2, 4), dtype=np.uint32)
    np.add.at(expect, (yi, xi, di), 1)
    assert np.array_equal(cube, expect)
    assert np.array_equal(ps.summed_decay(), expect.sum(axis=(0, 1)).astype(float))
    assert np.array_equal(ps.intensity_image(), expect.sum(axis=2).astype(np.uint64))

def test_no_channels(tmp_path):
    path = tmp_path / 'sample.photons'
    _sample(path)
    ps = PSFile(str(path), verbose=False)
    assert ps.n_channels == 1
    assert ps.photon_channel == 1
    assert ps.n_sync is None
    assert ps.photons_per_pulse is None

def test_read_ps_metadata(tmp_path):
    path = tmp_path / 'sample.photons'
    _sample(path)
    data, meta = read_ps(str(path), pixels=2, n_bins=4)
    assert data.shape == (2, 2, 4)
    assert meta['dims'] == ('Y', 'X', 'H')
    assert meta['n_bins'] == 4
    img, imeta = get_intensity_image(str(path), pixels=2, n_bins=4)
    assert img.shape == (2, 2)
    assert int(img.sum()) == 5

def test_dual_tdc(tmp_path):
    path = tmp_path / 'dual.photons'
    x = [100, 200, 300]
    y = [400, 500, 600]
    start = [1000, 2000, 3000]
    stop = [1500, 2400, 3800]
    _write_datasets(path, [
        ('/photons/x', 5, x),
        ('/photons/y', 5, y),
        ('/start/time', 3, start),
        ('/stop/time', 3, stop),
    ], {'/photons/PositionBits': '12'})
    assert has_dual_tdc(str(path))
    out = read_photons(str(path))
    assert np.array_equal(out['dt'], np.array([500, 400, 800]))
    assert np.array_equal(out['x'], np.array(x))
    ps = PSFile(str(path), verbose=False, n_bins=4)
    assert ps.dual_tdc
    ps._ensure_streams()
    assert ps.calib_source == 'dual_tdc'
    assert ps.tac_range == 801

def test_single_tdc_not_dual(tmp_path):
    path = tmp_path / 'sample.photons'
    _sample(path)
    assert not has_dual_tdc(str(path))
    assert PSFile(str(path), verbose=False).dual_tdc is False

def test_numpy_fallback_matches(tmp_path, monkeypatch):
    path = tmp_path / 'sample.photons'
    streams = _sample(path)
    out_default = read_photons(str(path))
    monkeypatch.setattr(pd, '_HAVE_NUMBA', False)
    out_fallback = read_photons(str(path))
    for k in ('x', 'y', 'dt', 'ms'):
        assert np.array_equal(out_default[k], out_fallback[k])
        assert np.array_equal(out_fallback[k], np.array(streams[k]))

def test_high_dataset_id(tmp_path):
    path = tmp_path / 'many.photons'
    f = [0, 1, 2]
    _write_datasets(path, [
        ('/aux/a', 5, f), ('/aux/b', 5, f), ('/aux/c', 5, f), ('/aux/d', 5, f),
        ('/photons/ms', 7, [0, 10, 20]),
        ('/photons/x', 5, [10, 20, 30]),
        ('/photons/y', 5, [40, 50, 60]),
        ('/photons/dt', 5, [1, 2, 3]),
    ], _ATTRS)
    out = read_photons(str(path))
    assert np.array_equal(out['dt'], np.array([1, 2, 3]))
    assert np.array_equal(out['x'], np.array([10, 20, 30]))
    assert np.array_equal(out['y'], np.array([40, 50, 60]))
