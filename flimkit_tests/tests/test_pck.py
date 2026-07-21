import struct
import numpy as np
import pytest
from flimkit.formats.PTU.reader import read_pck

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

def test_read_pck_requires_histogram(tmp_path):
    p = tmp_path / 'nohist.pck'
    out = bytearray(b'PQCHECK\x00' + b'\x00' * 8)
    out += _rec('ChkChannels', 0x10000008, struct.pack('<q', 1))
    out += _rec('Header_End', 0xFFFF0008, b'\x00' * 8)
    p.write_bytes(bytes(out))
    with pytest.raises(ValueError):
        read_pck(str(p))
