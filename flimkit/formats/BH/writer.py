import numpy as np

def _records():
    from sdtfile.sdtfile import FILE_HEADER, BLOCK_HEADER, MEASURE_INFO
    return (np.dtype(FILE_HEADER), np.dtype(BLOCK_HEADER), np.dtype(MEASURE_INFO))

def _measure_info_bytes(mi_dtype, n_bins, tcspc_res_ns, period_ns, image_x, image_y,
                        col_time_s, module='SPC-150'):
    mi = np.zeros(1, mi_dtype)
    tac_g = 1
    mi['adc_re'] = int(n_bins)
    mi['tac_g'] = tac_g
    mi['tac_r'] = float(tcspc_res_ns * 1e-9 * tac_g * n_bins)
    mi['image_x'] = int(image_x)
    mi['image_y'] = int(image_y)
    mi['scan_x'] = int(image_x)
    mi['scan_y'] = int(image_y)
    mi['meas_mode'] = 13
    mi['col_t'] = float(col_time_s)
    mi['mod_type'] = module.encode('ascii')[:16]
    sync = (1e9 / period_ns) if period_ns > 0 else 0.0
    si = mi['StopInfo']
    si['stop_time'] = float(col_time_s)
    si['min_sync_rate'] = float(sync)
    si['max_sync_rate'] = float(sync)
    return mi.tobytes()

def _file_info_bytes(title='FLIMKit synthetic', date='2026-01-01', time='00:00:00'):
    lines = [
        '*IDENTIFICATION',
        'ID : SPC Setup & Data File',
        f'Title : {title}',
        'Version : 1  850 M',
        f'Date : {date}',
        f'Time : {time}',
        '*END',
        '',
    ]
    return ('\r\n'.join(lines)).encode('windows-1250')

def write_sdt(path, cube, period_ns, tcspc_res_ns, col_time_s=10.0,
              title='FLIMKit synthetic', module='SPC-150'):
    cube = np.ascontiguousarray(cube)
    if cube.ndim != 3:
        raise ValueError('cube must be (Y, X, H)')
    ny, nx, nbins = cube.shape
    data = np.clip(cube, 0, 65535).astype('<u2')
    data_bytes = data.tobytes()
    fh_dtype, bh_dtype, mi_dtype = _records()
    info = _file_info_bytes(title=title)
    mi_bytes = _measure_info_bytes(mi_dtype, nbins, tcspc_res_ns, period_ns,
                                   nx, ny, col_time_s, module=module)
    header_size = fh_dtype.itemsize
    bh_size = bh_dtype.itemsize
    info_offs = header_size
    meas_offs = info_offs + len(info)
    data_block_offs = meas_offs + len(mi_bytes)
    data_offs = data_block_offs + bh_size
    next_block_offs = data_offs + len(data_bytes)
    bh = np.zeros(1, bh_dtype)
    bh['data_offs'] = data_offs
    bh['next_block_offs'] = next_block_offs
    bh['block_type'] = 0x0061
    bh['meas_desc_block_no'] = 0
    bh['lblock_no'] = 1
    bh['block_length'] = len(data_bytes)
    fh = np.zeros(1, fh_dtype)
    fh['revision'] = 15
    fh['info_offs'] = info_offs
    fh['info_length'] = len(info)
    fh['setup_offs'] = 0
    fh['setup_length'] = 0
    fh['data_block_offs'] = data_block_offs
    fh['no_of_data_blocks'] = 1
    fh['data_block_length'] = len(data_bytes)
    fh['meas_desc_block_offs'] = meas_offs
    fh['no_of_meas_desc_blocks'] = 1
    fh['meas_desc_block_length'] = len(mi_bytes)
    fh['header_valid'] = 0x5555
    with open(path, 'wb') as f:
        f.write(fh.tobytes())
        f.write(info)
        f.write(mi_bytes)
        f.write(bh.tobytes())
        f.write(data_bytes)
    return str(path)
