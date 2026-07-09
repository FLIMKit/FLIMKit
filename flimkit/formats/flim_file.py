from pathlib import Path
import importlib

_FORMATS = [
    {'id': 'ptu', 'label': 'PicoQuant PTU', 'exts': ('.ptu',),
     'modality': 'time', 'reader': 'flimkit.formats.PTU.reader:PTUFile'},
    {'id': 'bh_sdt', 'label': 'Becker & Hickl SDT', 'exts': ('.sdt',),
     'modality': 'time', 'reader': 'flimkit.formats.BH.reader:BHFile'},
    {'id': 'iss_tdflim', 'label': 'ISS time-tag',
     'exts': ('.tagtime', '.tagchannel', '.tagdecay'),
     'modality': 'time', 'reader': 'flimkit.formats.ISS.reader:ISSFile'},
    {'id': 'iss_image', 'label': 'ISS image', 'exts': ('.ifi',),
     'modality': 'intensity', 'reader': 'flimkit.formats.ISS.image:ISSImage'},
    {'id': 'iss_fdflim', 'label': 'ISS FD-FLIM (phasor)', 'exts': ('.ifli',),
     'modality': 'frequency', 'reader': 'flimkit.formats.ISS.fdflim:ISSFdFlim'},
    {'id': 'ps', 'label': 'Photonscore .photons', 'exts': ('.photons',),
     'modality': 'time', 'reader': 'flimkit.formats.PS.reader:PSFile'},
]

_EXT_TO_ID = {ext: f['id'] for f in _FORMATS for ext in f['exts']}
_MODALITY_BY_ID = {f['id']: f['modality'] for f in _FORMATS}

def _find_sibling(stem, ext_lower):
    direct = Path(str(stem) + ext_lower)
    if direct.exists():
        return direct
    parent = stem.parent if str(stem.parent) else Path('.')
    target = stem.name.lower() + ext_lower
    if parent.exists():
        for f in parent.iterdir():
            if f.name.lower() == target:
                return f
    return None

def _has_iss_triplet(p):
    stem = p.with_suffix('') if p.suffix else p
    return _find_sibling(stem, '.tagtime') is not None

def _sniff_magic(p):
    try:
        with open(p, 'rb') as fh:
            head = fh.read(64)
    except OSError:
        return 'unknown'
    if b'PQTTTR' in head or b'PTU' in head:
        return 'ptu'
    if head[:12] == b'VistaFLImage':
        return 'iss_fdflim'
    if head[:10] == b'VISTAIMAGE':
        return 'iss_image'
    if b'D7 Photons Data' in head:
        return 'ps'
    return 'unknown'

def _clean_path(path):
    s = str(path).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s

def detect_format(path):
    p = Path(_clean_path(path))
    ext = p.suffix.lower()
    if ext in _EXT_TO_ID:
        return _EXT_TO_ID[ext]
    if _has_iss_triplet(p):
        return 'iss_tdflim'
    return _sniff_magic(p)

def file_modality(path):
    return _MODALITY_BY_ID.get(detect_format(path), 'unknown')

def supported_formats():
    return [dict(f) for f in _FORMATS]

def supported_extensions():
    return [ext for f in _FORMATS for ext in f['exts']]

def file_dialog_filetypes():
    all_globs = ' '.join('*' + ext for ext in supported_extensions())
    types = [('FLIM files', all_globs)]
    for f in _FORMATS:
        globs = ' '.join('*' + ext for ext in f['exts'])
        types.append((f['label'], globs))
    return types

def _load_reader(fmt):
    for f in _FORMATS:
        if f['id'] == fmt:
            mod_name, cls_name = f['reader'].split(':')
            return getattr(importlib.import_module(mod_name), cls_name)
    return None

class FLIMFile:
    def __new__(cls, path, **kwargs):
        path = _clean_path(path)
        fmt = detect_format(path)
        reader = _load_reader(fmt)
        if reader is None:
            raise ValueError(f'Unrecognised FLIM file format: {path}')
        return reader(str(path), **kwargs)
