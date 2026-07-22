import argparse
import struct
import numpy as np
import pytest
from flimkit.formats.PTU.reader import read_pck
from flimkit.FLIM.irf_tools import align_irf_to_bin
from flimkit.interactive import _align_measured_irf

def _rec(ident, tagtyp, payload):
    return (ident.encode().ljust(32, b'\x00')
            + struct.pack('<i', -1)
            + struct.pack('<I', tagtyp)
            + payload)

def _write_pck(path, hist, n_chan):
    out = bytearray(b'PQCHECK\x00' + b'\x00' * 8)
    out += _rec('ChkChannels', 0x10000008, struct.pack('<q', n_chan))
    blob = np.asarray(hist, '<u4').tobytes()
    out += _rec('ChkHistogram', 0xFFFFFFFF, struct.pack('<q', len(blob)) + blob)
    out += _rec('Header_End', 0xFFFF0008, b'\x00' * 8)
    with open(path, 'wb') as fh:
        fh.write(bytes(out))
    return path

@pytest.mark.parametrize('n_chan,shape', [(1, (1, 16)), (2, (2, 8)), (3, (1, 16))])
def test_read_pck_shapes(tmp_path, n_chan, shape):
    p = tmp_path / 'irf.pck'
    _write_pck(p, np.arange(16, dtype='<u4'), n_chan)
    hist, tags = read_pck(str(p))
    assert hist.shape == shape
    assert hist.dtype == np.uint32
    assert hist.sum() == 120
    assert tags['ChkChannels'] == n_chan

def test_read_pck_values(tmp_path):
    p = tmp_path / 'irf.pck'
    _write_pck(p, np.arange(16, dtype='<u4'), 2)
    hist, _ = read_pck(str(p))
    assert np.array_equal(hist[0], np.arange(8))
    assert np.array_equal(hist[1], np.arange(8, 16))

def test_read_pck_rejects_non_pck(tmp_path):
    p = tmp_path / 'bad.pck'
    p.write_bytes(b'not a picoquant check file' * 4)
    with pytest.raises(ValueError):
        read_pck(str(p))

def _peaked_irf(n_bins, peak):
    x = np.arange(n_bins, dtype=float)
    irf = np.exp(-(x - peak) ** 2 / (2 * 3.0 ** 2))
    return irf / irf.sum()

@pytest.mark.parametrize('peak,target', [(400, 100), (100, 400), (250, 250)])
def test_align_irf_to_bin_lands_on_target(peak, target):
    irf = _peaked_irf(1024, peak)
    out, shift = align_irf_to_bin(irf, target, 1024)
    assert shift == target - peak
    assert int(np.argmax(out)) == target
    assert out.sum() == pytest.approx(1.0)

def test_align_irf_to_bin_noop_when_already_aligned():
    irf = _peaked_irf(1024, 250)
    out, shift = align_irf_to_bin(irf, 250, 1024)
    assert shift == 0
    assert out is irf

def test_measured_irf_untouched_by_default(capsys):
    irf = _peaked_irf(1024, 800)
    a = argparse.Namespace(irf_shift_bins=2)
    out, strategy = _align_measured_irf(a, irf, 100, 1024, 25e-12)
    assert int(np.argmax(out)) == 800
    assert strategy == 'measured_irf'
    assert 'WARNING' in capsys.readouterr().out

def test_measured_irf_aligned_when_requested():
    irf = _peaked_irf(1024, 800)
    a = argparse.Namespace(align_irf=True, irf_shift_bins=2)
    out, strategy = _align_measured_irf(a, irf, 100, 1024, 25e-12)
    assert int(np.argmax(out)) == 100
    assert 'aligned' in strategy
    assert out.sum() == pytest.approx(1.0)

def test_measured_irf_no_warning_when_close(capsys):
    irf = _peaked_irf(1024, 103)
    a = argparse.Namespace(irf_shift_bins=2)
    _align_measured_irf(a, irf, 100, 1024, 25e-12)
    assert 'WARNING' not in capsys.readouterr().out

def test_read_pck_requires_histogram(tmp_path):
    p = tmp_path / 'nohist.pck'
    out = bytearray(b'PQCHECK\x00' + b'\x00' * 8)
    out += _rec('ChkChannels', 0x10000008, struct.pack('<q', 1))
    out += _rec('Header_End', 0xFFFF0008, b'\x00' * 8)
    p.write_bytes(bytes(out))
    with pytest.raises(ValueError):
        read_pck(str(p))
