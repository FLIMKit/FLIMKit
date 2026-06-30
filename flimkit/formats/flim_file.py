from pathlib import Path

_PTU_EXTS = {'.ptu'}
_ISS_TRIPLET_EXTS = {'.tagtime', '.tagchannel', '.tagdecay'}

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
            head = fh.read(16)
    except OSError:
        return 'unknown'
    if b'PQTTTR' in head or b'PTU' in head:
        return 'ptu'
    if head[:12] == b'VistaFLImage':
        return 'iss_fdflim'
    if head[:10] == b'VISTAIMAGE':
        return 'iss_image'
    return 'unknown'

def _clean_path(path):
    # strip surrounding quotes/whitespace from a pasted path (drag-and-drop, Copy as Pathname)
    s = str(path).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s

def detect_format(path):
    p = Path(_clean_path(path))
    ext = p.suffix.lower()
    if ext in _PTU_EXTS:
        return 'ptu'
    if ext in _ISS_TRIPLET_EXTS:
        return 'iss_tdflim'
    if ext == '.ifli':
        return 'iss_fdflim'
    if ext == '.ifi':
        return 'iss_image'
    if ext == '.sdt':
        return 'bh_sdt'
    if _has_iss_triplet(p):
        return 'iss_tdflim'
    return _sniff_magic(p)

class FLIMFile:
    def __new__(cls, path, **kwargs):
        path = _clean_path(path)
        fmt = detect_format(path)
        if fmt == 'ptu':
            from flimkit.formats.PTU.reader import PTUFile
            return PTUFile(str(path), **kwargs)
        if fmt == 'iss_tdflim':
            from flimkit.formats.ISS.reader import ISSFile
            return ISSFile(str(path), **kwargs)
        if fmt == 'iss_image':
            from flimkit.formats.ISS.image import ISSImage
            return ISSImage(str(path), **kwargs)
        if fmt == 'iss_fdflim':
            raise NotImplementedError("ISS '.ifli' (frequency-domain) decoding is not implemented yet; see issue #19")
        if fmt == 'bh_sdt':
            from flimkit.formats.BH.reader import BHFile
            return BHFile(str(path), **kwargs)
        raise ValueError(f'Unrecognised FLIM file format: {path}')
