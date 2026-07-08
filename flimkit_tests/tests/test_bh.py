import os
import glob
import numpy as np
import pytest
from flimkit.formats import FLIMFile, detect_format
from flimkit.formats.BH.reader import BHFile, read_bh, get_flim_data, get_intensity_image
from .sdt_writer import _write_sdt, _write_sdt_multi, _known_cube

def test_channel_auto_select_brightest(tmp_path):
    path = tmp_path / 'twochan.sdt'
    dim = _known_cube()
    bright = _known_cube() * 100
    _write_sdt_multi(path, [dim, bright])
    bh = BHFile(str(path), verbose=False)
    assert bh.n_channels == 2
    assert bh.photon_channel is None
    decay = bh.summed_decay(channel=None)
    assert bh.photon_channel == 2
    assert np.array_equal(decay, bright.sum(axis=(0, 1)))
    assert np.array_equal(bh.summed_decay(channel=1), dim.sum(axis=(0, 1)))
    bh.close()

def test_detect_and_dispatch(tmp_path):
    path = tmp_path / 'sample.sdt'
    _write_sdt(path, _known_cube())
    assert detect_format(str(path)) == 'bh_sdt'
    bh = FLIMFile(str(path), verbose=False)
    assert isinstance(bh, BHFile)
    bh.close()

def test_cube_shape_and_content(tmp_path):
    path = tmp_path / 'sample.sdt'
    cube = _known_cube(iy=2, ix=3, nh=8)
    _write_sdt(path, cube)
    data, meta = read_bh(str(path))
    assert data.shape == (2, 3, 8)
    assert data.dtype == np.uint32
    assert np.array_equal(data, cube.astype(np.uint32))
    assert meta['dims'] == ('Y', 'X', 'H')
    assert meta['n_bins'] == 8
    assert meta['tcspc_resolution'] == pytest.approx(25.0e-9 / (2 * 8))
    assert meta['frequency'] == pytest.approx(4.0e7)

def test_summed_decay_and_intensity(tmp_path):
    path = tmp_path / 'sample.sdt'
    cube = _known_cube(iy=2, ix=3, nh=8)
    _write_sdt(path, cube)
    bh = BHFile(str(path), verbose=False)
    assert np.array_equal(bh.summed_decay(channel=1), cube.sum(axis=(0, 1)))
    assert np.array_equal(bh.intensity_image(channel=1), cube.sum(axis=2))
    img, _ = get_intensity_image(str(path))
    assert np.array_equal(img, cube.sum(axis=2))
    bh.close()

def test_channel_default_and_pileup(tmp_path):
    path = tmp_path / 'sample.sdt'
    _write_sdt(path, _known_cube())
    bh = BHFile(str(path), verbose=False)
    assert bh.n_channels == 1
    assert bh.photon_channel == 1
    assert bh.pileup_fraction is None
    assert np.array_equal(bh.summed_decay(channel=None), bh.summed_decay(channel=1))
    assert np.array_equal(bh.summed_decay(channel=0), bh.summed_decay(channel=1))
    bh.pixel_stack(channel=1)
    assert (bh.n_y, bh.n_x) == (2, 3)
    bh.close()

def test_power_of_two_padding(tmp_path):
    path = tmp_path / 'sample.sdt'
    cube = _known_cube(iy=2, ix=3, nh=8)
    _write_sdt(path, cube, pad_to=(2, 4))
    data, _ = read_bh(str(path))
    assert data.shape == (2, 3, 8)
    assert np.array_equal(data, cube.astype(np.uint32))

def test_binning(tmp_path):
    path = tmp_path / 'sample.sdt'
    cube = _known_cube(iy=4, ix=4, nh=8)
    _write_sdt(path, cube)
    bh = BHFile(str(path), verbose=False)
    binned = bh.pixel_stack(channel=1, binning=2)
    assert binned.shape == (2, 2, 8)
    assert int(binned.sum()) == int(cube.sum())
    full = bh.pixel_stack(channel=1, binning=1)
    assert full.shape == (4, 4, 8)
    bh.close()

_REAL_SAMPLES = sorted(glob.glob(os.path.expanduser('~/Downloads/*.sdt')))

@pytest.mark.skipif(not _REAL_SAMPLES, reason='no real Becker & Hickl .sdt sample present')
def test_real_sample_surface():
    path = _REAL_SAMPLES[0]
    assert detect_format(path) == 'bh_sdt'
    data, meta = get_flim_data(path)
    assert data.ndim == 3
    assert data.shape[2] == meta['n_bins']
    assert data.dtype == np.uint32
    assert meta['tcspc_resolution'] > 0
    assert set(meta).issuperset({'frequency', 'tcspc_resolution', 'n_bins',
                                 'time_ns', 'dims', 'shape', 'photon_channel'})
    assert int(data.sum()) > 0
