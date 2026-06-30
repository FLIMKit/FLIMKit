import numpy as np
from pathlib import Path
from . import decode as bd

_LASER_RATES_HZ = (20e6, 50e6, 80e6)

class BHFile:
    def __init__(self, path, verbose=True, channel=None):
        self.path = str(path)
        self.verbose = verbose
        self._fh = open(self.path, 'rb')
        self.header = bd.read_file_header(self._fh)
        if not self.header['valid'] and self.verbose:
            print(f"  WARNING: BH header_valid=0x{self.header['header_valid']:04x} "
                  f"(expected 0x5555)")
        self.identification = bd.read_identification(
            self._fh, self.header['info_offs'], self.header['info_length'])
        self.measure_info = bd.read_all_measure_info(self._fh, self.header)
        self.blocks = bd.read_block_headers(self._fh, self.header)
        self._img_blocks = [b for b in self.blocks
                            if b['content_type'] in (bd.IMG_BLOCK, bd.PAGE_BLOCK)]
        self._int_blocks = [b for b in self.blocks
                            if b['content_type'] == bd.IMG_INT_BLOCK]
        if not self._img_blocks:
            raise ValueError(f'No FLIM image/decay blocks found in {self.path}')
        self._cube_cache = {}
        self.n_x = None
        self.n_y = None
        self._total_photons = None
        self._parse_meta(channel)

    @property
    def fh(self):
        if self._fh is None or self._fh.closed:
            self._fh = open(self.path, 'rb')
        return self._fh

    def close(self):
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _parse_meta(self, channel):
        self.n_channels = len(self._img_blocks)
        self.photon_channel = int(channel) if channel is not None else 1
        mi = self.measure_info[self._img_blocks[0]['meas_desc_block_no']]
        self.module_type = mi['mod_type'] or self.header['module_type']
        self.n_bins = mi['adc_resolution']
        if mi['tac_g'] and self.n_bins:
            self.tcspc_res = mi['tac_r'] / (mi['tac_g'] * self.n_bins)
        else:
            self.tcspc_res = 0.0
        self.time_ns = (np.arange(self.n_bins) + 0.5) * self.tcspc_res * 1e9
        sync = mi['max_sync_rate'] if mi['max_sync_rate'] > 0 else mi['min_sync_rate']
        self.sync_rate = float(sync) if sync and sync > 0 else 0.0
        self.period_ns = (1e9 / self.sync_rate) if self.sync_rate > 0 else 0.0
        self.n_records = 0
        self.tags = {
            'BH_ID': self.identification.get('id'),
            'BH_Title': self.identification.get('title'),
            'BH_Date': self.identification.get('date'),
            'BH_Time': self.identification.get('time'),
            'BH_ModuleType': self.module_type,
            'BH_MeasMode': mi['meas_mode'],
            'BH_ADCResolution': self.n_bins,
            'BH_TACRange_s': mi['tac_r'],
            'BH_TACGain': mi['tac_g'],
            'BH_ImageX': mi['image_x'],
            'BH_ImageY': mi['image_y'],
            'BH_CollectionTime_s': mi['collection_time'],
            'BH_MinSyncRate_Hz': mi['min_sync_rate'],
            'BH_MaxSyncRate_Hz': mi['max_sync_rate'],
            'BH_Channels': self.n_channels,
        }
        if self.verbose:
            print(f"  B&H SPC  : {Path(self.path).name}")
            print(f"  Module   : {self.module_type}  ({self.n_channels} channel(s))")
            print(f"  TCSPC    : {self.n_bins} bins x {self.tcspc_res*1e12:.2f} ps")
            sync_mhz = self.sync_rate / 1e6
            print(f"  Sync     : {sync_mhz:.2f} MHz  (period {self.period_ns:.3f} ns)")
            print(' ')

    @property
    def pileup_fraction(self):
        return None

    def _norm_channel(self, channel):
        if channel is None or int(channel) < 1:
            return self.photon_channel
        return int(channel)

    def _img_block(self, channel):
        idx = self._norm_channel(channel) - 1
        if idx < 0 or idx >= len(self._img_blocks):
            raise ValueError(f'Channel {channel} out of range 1..{len(self._img_blocks)}')
        return self._img_blocks[idx]

    def _measure_info_for(self, block):
        return self.measure_info[block['meas_desc_block_no']]

    def _decode_cube(self, channel):
        idx = self._norm_channel(channel)
        if idx in self._cube_cache:
            return self._cube_cache[idx]
        block = self._img_block(idx)
        cube = bd.decode_image_block(self.fh, block, self._measure_info_for(block))
        self._cube_cache[idx] = cube
        return cube

    def pixel_stack(self, channel=None, binning=1):
        cube = self._decode_cube(channel)
        if binning > 1:
            cube = bd.bin_cube(cube, binning)
        self.n_y, self.n_x = cube.shape[0], cube.shape[1]
        self._total_photons = int(cube.sum())
        self.n_records = self._total_photons
        if self.verbose:
            print(f'  Built cube {cube.shape}  ({self._total_photons:,} photons)')
        return cube.astype(np.uint32)

    def raw_pixel_stack(self, channel=None, binning=1):
        return self.pixel_stack(channel=channel, binning=binning)

    def summed_decay(self, channel=None):
        cube = self._decode_cube(channel)
        return cube.sum(axis=(0, 1)).astype(float)

    def intensity_image(self, channel=None, binning=1):
        idx = self._norm_channel(channel)
        block = self._img_block(idx)
        mi = self._measure_info_for(block)
        match = [b for b in self._int_blocks
                 if b['meas_desc_block_no'] == block['meas_desc_block_no']]
        if match:
            img = bd.decode_intensity_block(self.fh, match[0], mi).astype(np.uint64)
        else:
            img = self.pixel_stack(channel=idx, binning=binning).sum(axis=2)
            return img
        if binning > 1:
            ny, nx = img.shape
            ny2, nx2 = (ny // binning) * binning, (nx // binning) * binning
            img = img[:ny2, :nx2].reshape(ny2 // binning, binning,
                                          nx2 // binning, binning).sum(axis=(1, 3))
        return img

def _metadata(bh, data):
    return {
        'frequency': bh.sync_rate,
        'tcspc_resolution': bh.tcspc_res,
        'shape': data.shape,
        'dims': ('Y', 'X', 'H'),
        'tags': bh.tags,
        'x_pixel_size': 0,
        'y_pixel_size': 0,
        'n_bins': bh.n_bins,
        'time_ns': bh.time_ns,
        'photon_channel': bh.photon_channel,
    }

def read_bh(path, binning=1, channel=None, verbose=False):
    bh = BHFile(path, verbose=verbose, channel=channel)
    data = bh.pixel_stack(channel=channel, binning=binning)
    metadata = _metadata(bh, data)
    bh.close()
    return data, metadata

def get_flim_data(path, binning=1, channel=None):
    return read_bh(path, binning=binning, channel=channel, verbose=False)

def get_intensity_image(path, binning=1, channel=None):
    bh = BHFile(path, verbose=False, channel=channel)
    img = bh.intensity_image(channel=channel, binning=binning)
    cube_shape = (bh.n_y, bh.n_x, bh.n_bins)
    metadata = _metadata(bh, np.empty(0))
    metadata['shape'] = cube_shape
    bh.close()
    return img, metadata

def normalise_flim(flim):
    if flim is None:
        return None
    if flim.ndim == 5:
        return flim[0, :, :, 0, :]
    if flim.ndim == 4:
        return flim[0] if flim.shape[0] == 1 else flim[:, :, 0, :]
    if flim.ndim == 3:
        return flim
    return None

def create_time_axis(n_bins, tcspc_resolution):
    return np.arange(n_bins) * tcspc_resolution * 1e9