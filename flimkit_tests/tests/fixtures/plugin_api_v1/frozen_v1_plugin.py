from flimkit.plugins import file_format, format_sniffer, phasor_filter, tool

FLIMKIT_PLUGIN_API = 1


@tool(id='frozen_v1_tool', label='Frozen v1 Tool...', menu='Tools', order=910)
def open_frozen(app):
    return app


@tool(id='frozen_v1_nested', label='Frozen v1 Nested...', menu='Tools/Frozen v1', order=920)
def open_frozen_nested(app):
    return app


@file_format(id='frozen_v1_format', label='Frozen v1 Format',
             exts=('.frozenv1',), modality='time')
class FrozenV1Reader:

    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs


@format_sniffer(tier='magic', order=50)
def sniff_frozen_v1(path):
    try:
        with open(path, 'rb') as fh:
            if fh.read(9) == b'FROZENV1!':
                return 'frozen_v1_format'
    except OSError:
        pass
    return None


@phasor_filter(id='frozen_v1_filter', label='Frozen v1 Filter')
def frozen_v1_filter(real, imag, sigma=1.0):
    return real * sigma, imag * sigma
