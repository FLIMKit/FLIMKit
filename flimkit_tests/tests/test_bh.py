import os
import glob
import struct
import numpy as np
import pytest
from flimkit.formats import FLIMFile, detect_format
from flimkit.formats.BH.reader import BHFile, read_bh, get_flim_data, get_intensity_image
from flimkit.formats.BH import decode as bd

def _build_measure_info(adc_re, tac_r, tac_g, image_x, image_y, meas_mode=13,
                        scan_x=0, scan_y=0, mod_type='SPC-150NX',
                        min_sync=4.0e7, max_sync=4.0e7, length=512):
    stop = bd._STOP_INFO.pack(0, 0, 0.0, 0, 0, 0,
                              min_sync, 0.0, 0.0, 0.0,
                              max_sync, 0.0, 0.0, 0.0,
                              0, b'\x00\x00\x00')
    fcs = b'\x00' * 38
    values = [
        b'12:00:00', b'2020-01-01', mod_type.encode('ascii'),
        meas_mode,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0, 0.0,
        float(tac_r), tac_g, 0.0, 0.0, 0.0,
        adc_re, 0, 0, 0, 0,
        0.0, 0.0, 0, b'\x00',
        0, 0, 0.0,
        0, 0, 0,
        mod_type.encode('ascii'), 0.0, 0,
        0, 0, 0,
        0, 0,
        0, 0, 0, 0,
        0.0, 0, 0,
        scan_x, scan_y, 0, 0,
        0, 0,
        0, 0, 0.0, 0, 0,
        stop,
        fcs,
        image_x, image_y, 0, 0,
    ]
    raw = bd._MEASURE_INFO.pack(*values)
    return raw.ljust(length, b'\x00')

def _write_sdt(path, cube, adc_re=None, tac_r=25.0e-9, tac_g=2,
               image_x=None, image_y=None, pad_to=None, module_code=0x080):
    cube = np.asarray(cube, dtype='<u2')
    iy, ix, nh = cube.shape
    image_x = ix if image_x is None else image_x
    image_y = iy if image_y is None else image_y
    adc_re = nh if adc_re is None else adc_re
    if pad_to is not None:
        yp, xp = pad_to
        stored = np.zeros((yp, xp, nh), dtype='<u2')
        stored[:iy, :ix, :] = cube
    else:
        stored = cube
    data_bytes = stored.tobytes()
    mi = _build_measure_info(adc_re, tac_r, tac_g, image_x, image_y)
    ident = ('*IDENTIFICATION\r\n'
             '  ID : SPC Setup & Data File\r\n'
             '  Title : synthetic\r\n'
             '  Version : 3  990 M\r\n'
             '  Revision : 10 bits ADC\r\n'
             '*END\r\n').encode('ascii')
    info_offs = bd._FILE_HEADER.size
    info_length = len(ident)
    setup_offs = info_offs + info_length
    setup_length = 0
    mdb_offs = setup_offs + setup_length
    mdb_len = len(mi)
    data_block_offs = mdb_offs + mdb_len
    data_offset = data_block_offs + bd._BLOCK_HEADER.size
    next_block_offset = data_offset + len(data_bytes)
    revision = (module_code << 4) | bd.SOFTWARE_REV_CURRENT
    header = bd._FILE_HEADER.pack(
        revision, info_offs, info_length, setup_offs, setup_length,
        data_block_offs, 1, len(data_bytes),
        mdb_offs, 1, mdb_len, bd.HEADER_VALID, 1, 0, 0)
    block_type = bd.IMG_BLOCK | 0x0001 
    block_header = bd._BLOCK_HEADER.pack(
        0, 0, data_offset, next_block_offset, block_type, 0, 1, len(data_bytes))
    with open(path, 'wb') as fh:
        fh.write(header)
        fh.write(ident)
        fh.write(mi)
        fh.write(block_header)
        fh.write(data_bytes)

def _known_cube(iy=2, ix=3, nh=8):
    cube = np.zeros((iy, ix, nh), dtype='<u2')
    for y in range(iy):
        for x in range(ix):
            cube[y, x, (y * ix + x) % nh] = (y + 1) * 10 + (x + 1)
    return cube

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
