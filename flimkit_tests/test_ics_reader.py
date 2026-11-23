import gzip
import numpy as np
import pytest
from mock_data import generate_synthetic_decay, MOCK_TCSPC_RES
from flimkit.formats import FLIMFile, detect_format
from flimkit.formats.ICS.reader import ICSFile

def _write_ics(path, cube, bits=16, compress=False, big=False, step=5e-11, order='y x t', history=()):
    dt = np.dtype(('>' if big else '<') + 'u' + str(bits // 8))
    perm = [{'y': 0, 'x': 1, 't': 2}[a] for a in order.split()]
    arr = cube.astype(dt).transpose(perm)
    sizes = ' '.join(str(arr.shape[i]) for i in range(3))
    data = arr.tobytes(order='F')
    if compress:
        data = gzip.compress(data)
    bo = ' '.join(str(i) for i in (range(bits // 8, 0, -1) if big else range(1, bits // 8 + 1)))
    lines = ['\t', 'ics_version\t2.0', 'filename\tsynthetic', 'layout\tparameters\t4',
             'layout\torder\tbits ' + order, 'layout\tsizes\t' + str(bits) + ' ' + sizes,
             'layout\tsignificant_bits\t' + str(bits), 'representation\tformat\tinteger',
             'representation\tsign\tunsigned',
             'representation\tcompression\t' + ('gzip' if compress else 'uncompressed'),
             'representation\tbyte_order\t' + bo]
    scale = {'y': '0.5', 'x': '0.25', 't': repr(step)}
    units = {'y': 'micron', 'x': 'micron', 't': 's'}
    lines += ['parameter\tscale\t1.0 ' + ' '.join(scale[a] for a in order.split()),
              'parameter\tunits\trelative ' + ' '.join(units[a] for a in order.split()),
              'parameter\tlabels\tintensity ' + ' '.join(order.split()),
              'history\tnote\tsome free text'] + list(history) + ['end\t']
    with open(path, 'wb') as fh:
        fh.write(('\n'.join(lines) + '\n').encode('ascii'))
        fh.write(data)

def _cube(ny=3, nx=4, nt=8):
    y, x, t = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nt), indexing='ij')
    return (100 * y + 10 * x + t).astype(np.uint32)

@pytest.mark.parametrize('bits', [16, 32])
@pytest.mark.parametrize('compress', [False, True])
@pytest.mark.parametrize('big', [False, True])
def test_roundtrip(tmp_path, bits, compress, big):
    path = tmp_path / 'a.ics'
    cube = _cube()
    _write_ics(path, cube, bits=bits, compress=compress, big=big)
    ics = FLIMFile(str(path), verbose=False)
    assert isinstance(ics, ICSFile)
    assert (ics.n_y, ics.n_x, ics.n_bins, ics.n_channels) == (3, 4, 8, 1)
    assert ics.photon_channel == 1
    assert ics.is_image == True
    stack = ics.pixel_stack()
    assert stack.dtype == np.uint32
    assert np.array_equal(stack, cube)
    assert np.array_equal(ics.raw_pixel_stack(), cube)
    assert np.array_equal(ics.summed_decay(), cube.sum(axis=(0, 1)))
    assert ics.tcspc_res == pytest.approx(5e-11)
    assert np.allclose(ics.time_ns, (np.arange(8) + 0.5) * 0.05)
    assert ics.pixel_size_um == pytest.approx(0.25)
    assert ics.history[0][1] == 'note'

def test_axis_order_is_read_not_assumed(tmp_path):
    path = tmp_path / 'b.ics'
    cube = _cube()
    _write_ics(path, cube, order='t x y')
    ics = ICSFile(str(path), verbose=False)
    assert np.array_equal(ics.pixel_stack(), cube)

def test_binning(tmp_path):
    path = tmp_path / 'c.ics'
    cube = _cube(4, 4, 8)
    _write_ics(path, cube)
    ics = ICSFile(str(path), verbose=False)
    out = ics.pixel_stack(binning=2)
    assert out.shape == (2, 2, 8)
    assert np.array_equal(out[0, 0], cube[:2, :2].sum(axis=(0, 1)))

def test_registry_picks_ics(tmp_path):
    path = tmp_path / 'd.ics'
    _write_ics(path, _cube())
    assert detect_format(str(path)) == 'ics'

def test_non_ics_file_errors(tmp_path):
    path = tmp_path / 'cal.ics'
    path.write_bytes(b'BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n')
    with pytest.raises(ValueError):
        FLIMFile(str(path))
    with pytest.raises(ValueError, match='Not an ICS file'):
        ICSFile(str(path))

def test_unsupported_axis_errors(tmp_path):
    path = tmp_path / 'e.ics'
    _write_ics(path, _cube())
    text = path.read_bytes().replace(b'bits y x t', b'bits y x z').replace(b'micro-time', b'z')
    path.write_bytes(text)
    with pytest.raises(ValueError, match="no 't' axis"):
        ICSFile(str(path))

def test_ics_v1_pair(tmp_path):
    cube = _cube()
    full = tmp_path / 'v2.ics'
    _write_ics(full, cube)
    raw = full.read_bytes()
    cut = raw.index(b'end\t\n') + 5
    (tmp_path / 'v1.ics').write_bytes(raw[:cut].replace(b'2.0', b'1.0').replace(b'end\t\n', b''))
    (tmp_path / 'v1.ids').write_bytes(raw[cut:])
    ics = ICSFile(str(tmp_path / 'v1.ics'), verbose=False)
    assert np.array_equal(ics.pixel_stack(), cube)
    (tmp_path / 'v1.ids').unlink()
    with pytest.raises(ValueError, match='ids'):
        ICSFile(str(tmp_path / 'v1.ics'), verbose=False)

def test_end_to_end_single_exp_fit(tmp_path):
    from flimkit.FLIM.fitters import fit_summed_tail
    rng = np.random.default_rng(1)
    clean = generate_synthetic_decay(n_bins=256, tau_ns=2.0, bg=5.0, peak_counts=2000.0, noise=False)
    cube = rng.poisson(np.broadcast_to(clean, (4, 4, 256))).astype(np.uint16)
    path = tmp_path / 'f.ics'
    _write_ics(path, cube, step=MOCK_TCSPC_RES)
    ics = FLIMFile(str(path), verbose=False)
    decay = ics.summed_decay()
    _popt, summary = fit_summed_tail(
        decay, ics.tcspc_res, ics.n_bins, True, 1, 0.2, 10.0,
        optimizer='lm_multistart', n_restarts=2, cost_function='poisson')
    assert summary['taus_ns'][0] == pytest.approx(2.0, rel=0.05)

def _leica_history(meta, size=20):
    text = __import__('json').dumps(meta)
    return [f'history\tLeicaMetadata_{i:04d}\t{text[n:n + size]}' for i, n in enumerate(range(0, len(text), size))]

def test_rep_rate_from_header_metadata(tmp_path):
    path = tmp_path / 'h.ics'
    _write_ics(path, _cube(), history=_leica_history({'timing': {'laserFrequency': {'value': 19505000.0}}}))
    (tmp_path / 'h.ics.json').write_text('{"calibration": {"laserFrequencyHz": 40000000.0}}')
    ics = ICSFile(str(path), verbose=False)
    assert ics.sync_rate == 19505000.0
    assert ics.period_ns == pytest.approx(51.2689, rel=1e-4)

def test_rep_rate_falls_back_to_sidecar(tmp_path):
    path = tmp_path / 's.ics'
    _write_ics(path, _cube())
    (tmp_path / 's.ics.json').write_text('{"calibration": {"laserFrequencyHz": 40000000.0}}')
    ics = ICSFile(str(path), verbose=False)
    assert ics.sync_rate == 40000000.0
    assert ics.period_ns == pytest.approx(25.0)

def test_rep_rate_zero_without_header_or_sidecar(tmp_path):
    path = tmp_path / 'n.ics'
    _write_ics(path, _cube())
    (tmp_path / 'n.ics.json').write_text('not json')
    ics = ICSFile(str(path), verbose=False)
    assert ics.sync_rate == 0.0
    assert ics.period_ns == 0.0
