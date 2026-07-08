import struct
import numpy as np

HEADER_VALID = 0x5555
SOFTWARE_REV_CURRENT = 15
IMG_BLOCK = 0x60

_FILE_HEADER = struct.Struct('<h l h l H l h L l h h H L H H')
_BLOCK_HEADER = struct.Struct('<B B I I H h I I')
_STOP_INFO = struct.Struct('<H H f i i i ffff ffff b 3s')
_MEASURE_INFO = struct.Struct(
    '<'
    '9s 11s 16s'
    'h'
    'fffff'
    'h f'
    'f h fff'
    'h h h h H'
    'f f h c'
    'h H f'
    'h h h'
    '16s f h'
    'h h h'
    'h h'
    'i i i i'
    'f h h'
    'i i i i'
    'h i'
    'H H f i i'
    '56s'
    '38s'
    '4x'
    'i i i i'
)

def _build_measure_info(adc_re, tac_r, tac_g, image_x, image_y, meas_mode=13,
                        scan_x=0, scan_y=0, mod_type='SPC-150NX',
                        min_sync=4.0e7, max_sync=4.0e7, length=512):
    stop = _STOP_INFO.pack(0, 0, 0.0, 0, 0, 0,
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
    raw = _MEASURE_INFO.pack(*values)
    return raw.ljust(length, b'\x00')

def _known_cube(iy=2, ix=3, nh=8):
    cube = np.zeros((iy, ix, nh), dtype='<u2')
    for y in range(iy):
        for x in range(ix):
            cube[y, x, (y * ix + x) % nh] = (y + 1) * 10 + (x + 1)
    return cube

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
    info_offs = _FILE_HEADER.size
    info_length = len(ident)
    setup_offs = info_offs + info_length
    setup_length = 0
    mdb_offs = setup_offs + setup_length
    mdb_len = len(mi)
    data_block_offs = mdb_offs + mdb_len
    data_offset = data_block_offs + _BLOCK_HEADER.size
    next_block_offset = data_offset + len(data_bytes)
    revision = (module_code << 4) | SOFTWARE_REV_CURRENT
    header = _FILE_HEADER.pack(
        revision, info_offs, info_length, setup_offs, setup_length,
        data_block_offs, 1, len(data_bytes),
        mdb_offs, 1, mdb_len, HEADER_VALID, 1, 0, 0)
    block_type = IMG_BLOCK | 0x0001
    block_header = _BLOCK_HEADER.pack(
        0, 0, data_offset, next_block_offset, block_type, 0, 1, len(data_bytes))
    with open(path, 'wb') as fh:
        fh.write(header)
        fh.write(ident)
        fh.write(mi)
        fh.write(block_header)
        fh.write(data_bytes)

def _write_sdt_multi(path, cubes, tac_r=25.0e-9, tac_g=2, module_code=0x080):
    cubes = [np.asarray(c, dtype='<u2') for c in cubes]
    iy, ix, nh = cubes[0].shape
    mi = _build_measure_info(nh, tac_r, tac_g, ix, iy)
    ident = ('*IDENTIFICATION\r\n  ID : SPC Setup & Data File\r\n'
             '  Title : synthetic\r\n*END\r\n').encode('ascii')
    info_offs = _FILE_HEADER.size
    info_length = len(ident)
    mdb_offs = info_offs + info_length
    mdb_len = len(mi)
    data_block_offs = mdb_offs + mdb_len
    blocks = b''
    offset = data_block_offs
    block_type = IMG_BLOCK | 0x0001
    for i, cube in enumerate(cubes):
        data_bytes = cube.tobytes()
        data_offset = offset + _BLOCK_HEADER.size
        next_offset = data_offset + len(data_bytes)
        block_header = _BLOCK_HEADER.pack(
            0, 0, data_offset, next_offset,
            block_type, 0, i + 1, len(data_bytes))
        blocks += block_header + data_bytes
        offset = next_offset
    revision = (module_code << 4) | SOFTWARE_REV_CURRENT
    header = _FILE_HEADER.pack(
        revision, info_offs, info_length, mdb_offs, 0,
        data_block_offs, len(cubes), len(cubes[0].tobytes()),
        mdb_offs, 1, mdb_len, HEADER_VALID, 1, 0, 0)
    with open(path, 'wb') as fh:
        fh.write(header)
        fh.write(ident)
        fh.write(mi)
        fh.write(blocks)
