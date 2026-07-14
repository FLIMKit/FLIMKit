import struct
import numpy as np

from flimkit.formats import FLIMFile, detect_format
from flimkit.formats.ISS.reader import ISSFile, read_iss

_FRAME, _LINE, _PIXEL = 5, 6, 7

def _write_triplet(base, period_ps=25000, ny=2, nx=3, det_ch=1):
    ch, dec, tim = [], [], []
    t = 0
    expected = np.zeros((ny, nx), dtype=int)
    def ev(c, d):
        nonlocal t
        ch.append(c)
        dec.append(d)
        tim.append(t)
        t += 100
    ev(_FRAME, 0)
    for y in range(ny):
        ev(_LINE, 0)
        for x in range(nx):
            ev(_PIXEL, 0)
            n_ph = (y + 1) * 10 + (x + 1)
            dbin = (y * nx + x) % 16
            expected[y, x] = n_ph
            for _ in range(n_ph):
                ev(det_ch, dbin)
    ch = np.array(ch, dtype='<i4')
    dec = np.array(dec, dtype='<i4')
    tim = np.array(tim, dtype='<i8')
    def write(path, arr, fmt):
        with open(path, 'wb') as fh:
            fh.write(struct.pack('<i', period_ps))
            fh.write(arr.astype(fmt).tobytes())
    write(str(base) + '.TAGTIME', tim, '<i8')
    write(str(base) + '.TAGCHANNEL', ch, '<i4')
    write(str(base) + '.TAGDECAY', dec, '<i4')
    return expected

def test_detect_and_dispatch(tmp_path):
    base = tmp_path / 'region1'
    _write_triplet(base)
    assert detect_format(str(base) + '.TAGTIME') == 'iss_tdflim'
    assert detect_format(str(base)) == 'iss_tdflim'
    assert isinstance(FLIMFile(str(base), verbose=False), ISSFile)

def test_cube_shape_and_intensity(tmp_path):
    base = tmp_path / 'region1'
    expected = _write_triplet(base, ny=2, nx=3)
    data, meta = read_iss(str(base))
    assert data.shape[:2] == (2, 3)
    assert data.dtype == np.uint32
    assert np.array_equal(data.sum(axis=2), expected)
    assert int(data.sum()) == int(expected.sum())
    assert meta['dims'] == ('Y', 'X', 'H')
    assert meta['frequency'] > 0

def test_decay_placement(tmp_path):
    base = tmp_path / 'region1'
    _write_triplet(base, ny=2, nx=3)
    f = ISSFile(str(base), verbose=False, n_bins=16)
    cube = f.pixel_stack(channel=1)
    # pixel (1, 2): 23 photons at decay bin (1*3 + 2) % 16 = 5
    assert cube[1, 2, 5] == 23

def test_surface_matches_ptu(tmp_path):
    base = tmp_path / 'region1'
    _write_triplet(base, ny=2, nx=3)
    f = ISSFile(str(base), verbose=False)
    assert (f.n_y, f.n_x) == (2, 3)
    assert f.sync_rate == f.frequency
    assert f.n_records == f.channel.shape[0]
    assert f.photon_channel == 1
    assert f.n_sync is None
    assert f.photons_per_pulse is None
    f.pixel_stack(channel=1)
    assert f.photons_per_pulse is None
    # binning must not clobber the full-res dims
    f.pixel_stack(channel=1, binning=2)
    assert (f.n_y, f.n_x) == (2, 3)

def test_path_resolution_from_any_member(tmp_path):
    base = tmp_path / 'region1'
    _write_triplet(base)
    a, _ = read_iss(str(base) + '.TAGDECAY')
    b, _ = read_iss(str(base) + '.TAGCHANNEL')
    assert np.array_equal(a, b)
