import struct
import numpy as np

# revision low-nibble values
SOFTWARE_REV_CURRENT = 15
HEADER_VALID = 0x5555

MODULE_TYPES = {
    0x020: 'SPC-130', 0x021: 'SPC-600', 0x022: 'SPC-630', 0x023: 'SPC-700',
    0x024: 'SPC-730', 0x025: 'SPC-830', 0x026: 'SPC-140', 0x027: 'SPC-930',
    0x028: 'SPC-150', 0x029: 'DPC-230', 0x02a: 'SPC-130EM', 0x02b: 'SPC-160',
    0x02e: 'SPC-150N', 0x080: 'SPC-150NX', 0x081: 'SPC-160X', 0x082: 'SPC-160PCIE',
    0x083: 'SPC-130EMN', 0x084: 'SPC-180N', 0x085: 'SPC-180NX', 0x086: 'SPC-180NXX',
    0x087: 'SPC-180N-USB', 0x088: 'SPC-130IN', 0x089: 'SPC-130INX', 0x08a: 'SPC-130INXX',
    0x08b: 'SPC-QC-104', 0x08c: 'SPC-QC-004', 0x08d: 'SPC-QC-106', 0x08e: 'SPC-QC-006',
    0x180: 'SPC-150NXX',
}

DECAY_BLOCK = 0x00
PAGE_BLOCK = 0x10
FCS_BLOCK = 0x20
FIDA_BLOCK = 0x30
FILDA_BLOCK = 0x40
MCS_BLOCK = 0x50
IMG_BLOCK = 0x60
MCSTA_BLOCK = 0x70
IMG_MCS_BLOCK = 0x80
MOM_BLOCK = 0x90
IMG_INT_BLOCK = 0xa0
IMG_WF_BLOCK = 0xb0

CONTENT_NAMES = {
    DECAY_BLOCK: 'DECAY', PAGE_BLOCK: 'PAGE', FCS_BLOCK: 'FCS', FIDA_BLOCK: 'FIDA',
    FILDA_BLOCK: 'FILDA', MCS_BLOCK: 'MCS', IMG_BLOCK: 'IMG', MCSTA_BLOCK: 'MCSTA',
    IMG_MCS_BLOCK: 'IMG_MCS', MOM_BLOCK: 'MOM', IMG_INT_BLOCK: 'IMG_INT',
    IMG_WF_BLOCK: 'IMG_WF',
}

DATA_USHORT = 0x000
DATA_ULONG = 0x100
DATA_DBL = 0x200
DATA_DTYPES = {DATA_USHORT: np.uint16, DATA_ULONG: np.uint32, DATA_DBL: np.float64}

COMPRESS_ZIP = 0x1000
COINCIDENCE = 0x2000
COMPRESS_LZ4 = 0x4000

_FILE_HEADER = struct.Struct('<h l h l H l h L l h h H L H H')

_BLOCK_HEADER = struct.Struct('<B B I I H h I I')

_MEASURE_INFO = struct.Struct(
    '<'
    '9s 11s 16s'        # time, date, mod_ser_no
    'h'                 # meas_mode
    'fffff'             # cfd_ll, cfd_lh, cfd_zc, cfd_hf, syn_zc
    'h f'               # syn_fd, syn_hf
    'f h fff'           # tac_r, tac_g, tac_of, tac_ll, tac_lh
    'h h h h H'         # adc_re, eal_de, ncx, ncy, page
    'f f h c'           # col_t, rep_t, stopt, overfl
    'h H f'             # use_motor, steps, offset
    'h h h'             # dither, incr, mem_bank
    '16s f h'           # mod_type, syn_th, dead_time_comp
    'h h h'             # polarity_l, polarity_f, polarity_p
    'h h'               # linediv, accumulate
    'i i i i'           # flyback_y, flyback_x, border_u, border_l
    'f h h'             # pix_time, pix_clk, trigger
    'i i i i'           # scan_x, scan_y, scan_rx, scan_ry
    'h i'               # fifo_typ, epx_div
    'H H f i i'         # mod_type_code, mod_fpga_ver, overflow_corr_factor, adc_zoom, cycles
    '56s'               # StopInfo (MeasStopInfo)
    '38s'               # FCSInfo (MeasFCSInfo)
    '4x'                # undocumented 4-byte gap
    'i i i i'           # image_x, image_y, image_rx, image_ry
)

_STOP_INFO = struct.Struct('<H H f i i i ffff ffff b 3s')

def _c_string(raw):
    return raw.split(b'\x00', 1)[0].decode('ascii', 'replace').strip()

def software_revision(revision):
    return revision & 0x000f

def module_type_code(revision):
    return (revision >> 4) & 0x0fff

def module_type_name(revision):
    code = module_type_code(revision)
    return MODULE_TYPES.get(code, f'unknown (0x{code:03x})')

def read_file_header(fh):
    fh.seek(0)
    raw = fh.read(_FILE_HEADER.size)
    if len(raw) != _FILE_HEADER.size:
        raise EOFError(f'BH header truncated: got {len(raw)} of {_FILE_HEADER.size} bytes')
    keys = ['revision', 'info_offs', 'info_length', 'setup_offs', 'setup_length',
            'data_block_offs', 'no_of_data_blocks', 'data_block_length',
            'meas_desc_block_offs', 'no_of_meas_desc_blocks', 'meas_desc_block_length',
            'header_valid', 'reserved1', 'reserved2', 'chksum']
    header = dict(zip(keys, _FILE_HEADER.unpack(raw)))
    if header['no_of_data_blocks'] == 0x7fff:
        header['n_data_blocks'] = header['reserved1']
    else:
        header['n_data_blocks'] = header['no_of_data_blocks']
    header['software_revision'] = software_revision(header['revision'])
    header['module_type'] = module_type_name(header['revision'])
    header['valid'] = header['header_valid'] == HEADER_VALID
    return header

def read_identification(fh, offset, length):
    fh.seek(offset)
    text = fh.read(length).decode('ascii', 'replace')
    entries = {}
    inside = False
    for line in text.splitlines():
        s = line.strip()
        if s == '*IDENTIFICATION':
            inside = True
            continue
        if s == '*END' and inside:
            break
        if inside and ':' in s:
            key, value = s.split(':', 1)
            entries[key.strip().lower().replace(' ', '_')] = value.strip().strip('\x04').strip()
    return entries

def read_measure_info(fh, offset, length):
    fh.seek(offset)
    raw = fh.read(length)
    if len(raw) < _MEASURE_INFO.size:
        raise EOFError(f'MeasureInfo truncated: got {len(raw)} of {_MEASURE_INFO.size} bytes')
    v = _MEASURE_INFO.unpack(raw[:_MEASURE_INFO.size])
    stop = _STOP_INFO.unpack(v[57])
    adc_re = v[16] if v[16] != 0 else 65536
    return {
        'time': _c_string(v[0]),
        'date': _c_string(v[1]),
        'module_serial': _c_string(v[2]),
        'meas_mode': v[3],
        'tac_r': v[11],
        'tac_g': v[12],
        'adc_resolution': adc_re,
        'collection_time': v[21],
        'mod_type': _c_string(v[31]),
        'pixel_time': v[43],
        'scan_x': v[46],
        'scan_y': v[47],
        'image_x': v[59],
        'image_y': v[60],
        'min_sync_rate': stop[6],
        'max_sync_rate': stop[10],
    }

def read_all_measure_info(fh, header):
    base = header['meas_desc_block_offs']
    length = header['meas_desc_block_length']
    return [read_measure_info(fh, base + i * length, length)
            for i in range(header['no_of_meas_desc_blocks'])]

def _decode_block_type(block_type):
    content = block_type & 0x00f0
    data = block_type & 0x0f00
    return {
        'block_type': block_type,
        'creation_mode': block_type & 0x000f,
        'content_type': content,
        'content_name': CONTENT_NAMES.get(content, f'0x{content:02x}'),
        'data_type': data,
        'dtype': DATA_DTYPES.get(data, np.uint16),
        'zip': bool(block_type & COMPRESS_ZIP),
        'lz4': bool(block_type & COMPRESS_LZ4),
        'coincidence': bool(block_type & COINCIDENCE),
    }

def read_block_headers(fh, header):
    blocks = []
    offset = header['data_block_offs']
    seen = set()
    for _ in range(header['n_data_blocks']):
        if offset in seen:
            raise ValueError(f'loop in BH block headers at offset {offset}')
        seen.add(offset)
        fh.seek(offset)
        raw = fh.read(_BLOCK_HEADER.size)
        if len(raw) != _BLOCK_HEADER.size:
            raise EOFError('BH block header truncated')
        (data_ext, next_ext, data_offs, next_offs,
         block_type, meas_desc_no, lblock_no, block_length) = _BLOCK_HEADER.unpack(raw)
        block = _decode_block_type(block_type)
        block.update({
            'data_offset': data_offs | (data_ext << 32),
            'next_block_offset': next_offs | (next_ext << 32),
            'meas_desc_block_no': meas_desc_no,
            'lblock_no': lblock_no,
            'routing_channel': lblock_no & 0xff,
            'block_length': block_length,
        })
        blocks.append(block)
        offset = block['next_block_offset']
        if offset == 0:
            break
    return blocks

def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length() if n > 1 else 1

def _lz4_decompress(raw, target_length):
    try:
        import lz4.frame
    except ImportError:
        raise ImportError('lz4 is required to read LZ4-compressed B&H blocks '
                          '(pip install lz4)')
    parts = []
    total = 0
    decompressor = lz4.frame.LZ4FrameDecompressor()
    remaining = raw
    while remaining:
        part = decompressor.decompress(remaining)
        parts.append(part)
        total += len(part)
        if total >= target_length:
            break
        remaining = decompressor.unused_data
    return b''.join(parts)

def read_block_bytes(fh, block):
    fh.seek(block['data_offset'])
    raw = fh.read(block['next_block_offset'] - block['data_offset'])
    if block['zip']:
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            return archive.read(archive.namelist()[0])
    if block['lz4']:
        return _lz4_decompress(raw, block['block_length'])
    return raw[:block['block_length']]

def _reshape_image(arr, image_x, image_y, last_dim):
    per_pixel = last_dim if last_dim > 0 else 1
    n_pixels = arr.size // per_pixel
    if n_pixels == image_x * image_y:
        out = arr[:image_y * image_x * per_pixel].reshape(image_y, image_x, per_pixel)
    else:
        xpad, ypad = _next_pow2(image_x), _next_pow2(image_y)
        out = arr[:ypad * xpad * per_pixel].reshape(ypad, xpad, per_pixel)
        out = out[:image_y, :image_x, :]
    if last_dim > 0:
        return np.ascontiguousarray(out)
    return np.ascontiguousarray(out[:, :, 0])

def decode_image_block(fh, block, measure_info):
    data = read_block_bytes(fh, block)
    arr = np.frombuffer(data, dtype=np.dtype(block['dtype']))
    return _reshape_image(arr, measure_info['image_x'], measure_info['image_y'],
                          measure_info['adc_resolution'])

def decode_intensity_block(fh, block, measure_info):
    data = read_block_bytes(fh, block)
    arr = np.frombuffer(data, dtype=np.dtype(block['dtype']))
    return _reshape_image(arr, measure_info['image_x'], measure_info['image_y'], 0)

def bin_cube(cube, binning):
    if binning <= 1:
        return cube
    ny, nx, nh = cube.shape
    ny2, nx2 = (ny // binning) * binning, (nx // binning) * binning
    cube = cube[:ny2, :nx2, :]
    return cube.reshape(ny2 // binning, binning, nx2 // binning, binning, nh).sum(axis=(1, 3))