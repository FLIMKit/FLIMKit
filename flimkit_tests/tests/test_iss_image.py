import struct
import numpy as np
import pytest
from flimkit.formats import FLIMFile, detect_format
from flimkit.formats.ISS.image import ISSImage, read_ifi

def _write_ifi(path, res_x=3, res_y=2, n_channels=1, n_frames=1):
    buf = bytearray(256)
    buf[0:10] = b'VISTAIMAGE'
    buf[10] = 1
    buf[12] = n_channels
    struct.pack_into('<H', buf, 14, res_x)
    struct.pack_into('<H', buf, 16, res_y)
    struct.pack_into('<H', buf, 18, 1)
    struct.pack_into('<H', buf, 93, n_channels)
    imgs = np.zeros((n_channels, n_frames, res_y, res_x), dtype='<f4')
    for c in range(n_channels):
        for f in range(n_frames):
            imgs[c, f] = 1.0 + c * 100 + f * 10 + np.arange(res_y * res_x).reshape(res_y, res_x)
    with open(path, 'wb') as fh:
        fh.write(buf)
        fh.write(imgs.tobytes())
    return imgs

def test_detect_and_dispatch(tmp_path):
    p = tmp_path / 'scan.ifi'
    _write_ifi(p)
    assert detect_format(str(p)) == 'iss_image'
    assert isinstance(FLIMFile(str(p), verbose=False), ISSImage)

def test_single_channel_image(tmp_path):
    p = tmp_path / 'scan.ifi'
    imgs = _write_ifi(p, res_x=3, res_y=2, n_channels=1, n_frames=1)
    iss = ISSImage(str(p), verbose=False)
    assert (iss.n_y, iss.n_x) == (2, 3)
    assert iss.n_channels == 1 and iss.n_frames == 1
    assert np.array_equal(iss.image(), imgs[0, 0])

def test_multichannel_multiframe_reshape(tmp_path):
    p = tmp_path / 'scan.ifi'
    imgs = _write_ifi(p, res_x=4, res_y=3, n_channels=2, n_frames=2)
    iss = ISSImage(str(p), verbose=False)
    assert iss.n_channels == 2 and iss.n_frames == 2
    assert np.array_equal(iss.image(channel=2), imgs[1].sum(axis=0))
    assert np.array_equal(iss.image(channel=1, frame=0), imgs[0, 0])

def test_read_ifi_sums_frames(tmp_path):
    p = tmp_path / 'scan.ifi'
    imgs = _write_ifi(p, n_channels=1, n_frames=3)
    img, meta = read_ifi(str(p))
    assert img.shape == (2, 3)
    assert np.array_equal(img, imgs[0].sum(axis=0))
    assert meta['n_frames'] == 3
    assert meta['dims'] == ('C', 'T', 'Y', 'X')

def test_lifetime_methods_raise(tmp_path):
    p = tmp_path / 'scan.ifi'
    _write_ifi(p)
    iss = ISSImage(str(p), verbose=False)
    with pytest.raises(ValueError):
        iss.pixel_stack()
    with pytest.raises(ValueError):
        iss.summed_decay()