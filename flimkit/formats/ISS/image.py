import struct
import numpy as np
from pathlib import Path

_HEADER_BYTES = 256
_SIGNATURE = b'VISTAIMAGE'

def _read_header(buf):
    h = {}
    h['signature'] = bytes(buf[0:10])
    h['file_version'] = buf[10]
    h['image_type'] = buf[11]
    h['channel_bits'] = buf[12]
    h['compression'] = buf[13]
    h['res_x'] = struct.unpack_from('<H', buf, 14)[0]
    h['res_y'] = struct.unpack_from('<H', buf, 16)[0]
    h['res_z'] = struct.unpack_from('<H', buf, 18)[0]
    region = struct.unpack_from('<6f', buf, 20)
    h['region'] = {'x': region[0:2], 'y': region[2:4], 'z': region[4:6]}
    h['pixel_acq_time'] = struct.unpack_from('<f', buf, 44)[0]
    h['n_channels'] = struct.unpack_from('<H', buf, 93)[0]
    h['n_time_series'] = struct.unpack_from('<H', buf, 95)[0]
    h['pixel_interval'] = struct.unpack_from('<f', buf, 97)[0]
    h['line_interval'] = struct.unpack_from('<f', buf, 101)[0]
    h['frame_interval'] = struct.unpack_from('<f', buf, 105)[0]
    h['n_frames_integrated'] = struct.unpack_from('<H', buf, 109)[0]
    return h

class ISSImage:
    def __init__(self, path, verbose=True, byteorder='<'):
        self.path = str(path)
        self.verbose = verbose
        with open(self.path, 'rb') as fh:
            buf = fh.read(_HEADER_BYTES)
            data = np.frombuffer(fh.read(), dtype=np.dtype(byteorder + 'f4'))
        if len(buf) < _HEADER_BYTES:
            raise ValueError(f'ISS .ifi header truncated: {self.path}')
        self.header = _read_header(buf)
        if self.header['signature'][:10] != _SIGNATURE and self.verbose:
            print(f"  WARNING: ISS .ifi signature {self.header['signature']!r} != {_SIGNATURE!r}")
        self._parse(data)

    def _parse(self, data):
        h = self.header
        self.n_x = int(h['res_x'])
        self.n_y = int(h['res_y'])
        self.n_z = int(h['res_z'])
        nc = int(h['n_channels'])
        if nc <= 0:
            nc = bin(int(h['channel_bits'])).count('1') or 1
        per_frame = self.n_x * self.n_y
        self.n_frames = (data.size // (nc * per_frame)) if nc * per_frame > 0 else 0
        self.n_channels = nc
        keep = nc * self.n_frames * per_frame
        if keep > 0:
            self.images = data[:keep].reshape(nc, self.n_frames, self.n_y, self.n_x)
        else:
            self.images = np.zeros((nc, 0, self.n_y, self.n_x), dtype=np.float32)
        self.tags = {
            'ISS_FileVersion': int(h['file_version']),
            'ISS_ChannelBits': int(h['channel_bits']),
            'ISS_ResolutionXYZ': (self.n_x, self.n_y, self.n_z),
            'ISS_NumChannels': self.n_channels,
            'ISS_NumFrames': self.n_frames,
            'ISS_NumTimeSeries': int(h['n_time_series']),
            'ISS_PixelAcqTime': float(h['pixel_acq_time']),
            'ISS_Region': h['region'],
        }
        if self.verbose:
            print(f'  ISS image : {Path(self.path).stem}')
            print(f'  Size     : {self.n_x} x {self.n_y} px, {self.n_channels} ch, {self.n_frames} frame(s)')
            print(' ')

    def _channel_index(self, channel):
        c = int(channel)
        idx = c - 1 if c >= 1 else c
        if idx < 0 or idx >= self.n_channels:
            return 0
        return idx
    
    def image(self, channel=None, frame=None):
        if self.n_channels == 0 or self.n_frames == 0:
            return np.zeros((self.n_y, self.n_x), dtype=np.float32)
        ci = 0 if channel is None else self._channel_index(channel)
        if frame is None:
            return self.images[ci].sum(axis=0)
        return self.images[ci, frame]
    
    @property
    def intensity_image(self):
        return self.image()
    
    def pixel_stack(self, *args, **kwargs):
        raise ValueError('ISS .ifi is an intensity image with no lifetime data; use intensity_image / get_intensity_image()')
    
    def summed_decay(self, *args, **kwargs):
        raise ValueError('ISS .ifi is an intensity image with no lifetime data; use intensity_image / get_intensity_image()')

def read_ifi(path, channel=None, verbose=False):
    iss = ISSImage(path, verbose=verbose)
    img = iss.image(channel=channel)
    metadata = {
        'shape': iss.images.shape,
        'dims': ('C', 'T', 'Y', 'X'),
        'n_x': iss.n_x,
        'n_y': iss.n_y,
        'n_channels': iss.n_channels,
        'n_frames': iss.n_frames,
        'tags': iss.tags,
    }
    return img, metadata

def get_intensity_image(path, channel=None):
    return read_ifi(path, channel=channel, verbose=False)
