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
    {'id': 'pq_phu', 'label': 'PicoQuant PHU (histogram)', 'exts': ('.phu',),
     'modality': 'time', 'reader': 'flimkit.formats.PTU.phu:PHUFile'},
    {'id': 'pq_bin', 'label': 'PicoQuant BIN', 'exts': ('.bin',),
     'modality': 'time', 'reader': 'flimkit.formats.signal:PQBinFile'},
    {'id': 'simfcs_bh', 'label': 'SimFCS B&H', 'exts': ('.b&h',),
     'modality': 'time', 'reader': 'flimkit.formats.signal:SimfcsBHFile'},
    {'id': 'simfcs_bhz', 'label': 'SimFCS BHZ', 'exts': ('.bhz',),
     'modality': 'time', 'reader': 'flimkit.formats.signal:SimfcsBHZFile'},
    {'id': 'imspector_tiff', 'label': 'ImSpector FLIM TIFF', 'exts': ('.tif', '.tiff'),
     'modality': 'time', 'reader': 'flimkit.formats.signal:ImspectorTIFFFile'},
    {'id': 'iss_vista_tdflim', 'label': 'ISS Vista TDFLIM', 'exts': ('.iss-tdflim', '.tdflim'),
     'modality': 'time', 'reader': 'flimkit.formats.signal:VistaTdflimFile'},
    {'id': 'simfcs_referenced', 'label': 'SimFCS referenced (phasor)', 'exts': ('.ref', '.r64'),
     'modality': 'frequency', 'reader': 'flimkit.formats.phasor:SimfcsReferencedFile'},
    {'id': 'ometiff_phasor', 'label': 'PhasorPy OME-TIFF (phasor)', 'exts': ('.ome.tif',),
     'modality': 'frequency', 'reader': 'flimkit.formats.phasor:OmeTiffPhasorFile'},
    {'id': 'flimlabs_phasor', 'label': 'FLIM LABS phasor (JSON)', 'exts': ('.json',),
     'modality': 'frequency', 'reader': 'flimkit.formats.phasor:FlimLabsPhasorFile'},
    {'id': 'flimlabs_signal', 'label': 'FLIM LABS imaging (JSON)', 'exts': (),
     'modality': 'time', 'reader': 'flimkit.formats.signal:FlimLabsSignalFile'},
]

_cache = {'version': None, 'formats': None, 'ext_to_id': None, 'modality_by_id': None}

def _registry():
    from flimkit import plugins
    plugins.ensure_loaded()
    return plugins.registry

def _all_formats():
    reg = _registry()
    if _cache['version'] != reg.version():
        merged = [dict(f) for f in _FORMATS]
        known = {f['id'] for f in merged}
        for f in reg.formats():
            if f.id in known:
                continue
            merged.append({'id': f.id, 'label': f.label, 'exts': f.exts,
                           'modality': f.modality, 'reader': f})
        ext_to_id = {}
        for f in merged:
            for ext in f['exts']:
                ext_to_id.setdefault(ext, f['id'])
        _cache['formats'] = merged
        _cache['ext_to_id'] = ext_to_id
        _cache['modality_by_id'] = {f['id']: f['modality'] for f in merged}
        _cache['version'] = reg.version()
    return _cache['formats']

def _ext_to_id():
    _all_formats()
    return _cache['ext_to_id']

def _modality_by_id():
    _all_formats()
    return _cache['modality_by_id']

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

def _sniff_tiff(p):
    try:
        with open(p, 'rb') as fh:
            head = fh.read(4)
    except OSError:
        return 'unknown'
    if head not in (b'II*\x00', b'MM\x00*'):
        return 'unknown'
    try:
        import tifffile
        with tifffile.TiffFile(str(p)) as tif:
            make = tif.pages.first.tags.valueof(271, '')
            if make == 'ImSpector':
                return 'imspector_tiff'
            names = [s.name for s in tif.series[:3]]
            if tif.is_ome and names == ['Phasor mean', 'Phasor real', 'Phasor imag']:
                return 'ometiff_phasor'
    except Exception:
        pass
    return 'unknown'

def _scan_bytes(p, needle, chunk=1 << 20):
    try:
        with open(p, 'rb') as fh:
            tail = b''
            while True:
                buf = fh.read(chunk)
                if not buf:
                    return False
                if needle in tail + buf:
                    return True
                tail = buf[-len(needle):]
    except OSError:
        return False

def _sniff_json(p):
    try:
        with open(p, 'rb') as fh:
            head = fh.read(8192)
    except OSError:
        return 'unknown'
    if b'"file_id"' not in head or b'"laser_period_ns"' not in head:
        return 'unknown'
    if _scan_bytes(p, b'"phasors_data"'):
        return 'flimlabs_phasor'
    if _scan_bytes(p, b'"intensities_data"') or _scan_bytes(p, b'"data"'):
        return 'flimlabs_signal'
    return 'unknown'

def _sniff_magic(p):
    try:
        with open(p, 'rb') as fh:
            head = fh.read(64)
    except OSError:
        return 'unknown'
    if b'PQHISTO' in head:
        return 'pq_phu'
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

def _run_sniffers(tier, p):
    for sniffer in _registry().sniffers(tier):
        try:
            found = sniffer(p)
        except Exception as exc:
            print(f'[Plugins] sniffer from {sniffer.source} raised {exc}')
            continue
        if found and found != 'unknown':
            return found
    return None

def detect_format(path):
    p = Path(_clean_path(path))
    ext = p.suffix.lower()
    if ext in ('.tif', '.tiff'):
        return _sniff_tiff(p)
    if ext == '.json':
        return _sniff_json(p)
    found = _run_sniffers('extension', p)
    if found is not None:
        return found
    ext_to_id = _ext_to_id()
    if ext in ext_to_id:
        return ext_to_id[ext]
    if _has_iss_triplet(p):
        return 'iss_tdflim'
    found = _sniff_magic(p)
    if found != 'unknown':
        return found
    found = _run_sniffers('magic', p)
    if found is not None:
        return found
    return 'unknown'

def file_modality(path):
    return _modality_by_id().get(detect_format(path), 'unknown')

def supported_formats():
    return [dict(f) for f in _all_formats()]

def supported_extensions():
    return [ext for f in _all_formats() for ext in f['exts']]

def file_dialog_filetypes():
    all_globs = ' '.join('*' + ext for ext in supported_extensions())
    types = [('FLIM files', all_globs)]
    for f in _all_formats():
        globs = ' '.join('*' + ext for ext in f['exts'])
        types.append((f['label'], globs))
    return types

def _load_reader(fmt):
    for f in _all_formats():
        if f['id'] == fmt:
            spec = f['reader']
            if isinstance(spec, str):
                mod_name, cls_name = spec.split(':')
                return getattr(importlib.import_module(mod_name), cls_name)
            return spec.reader()
    return None

class FLIMFile:
    def __new__(cls, path, **kwargs):
        path = _clean_path(path)
        fmt = detect_format(path)
        reader = _load_reader(fmt)
        if reader is None:
            raise ValueError(f'Unrecognised FLIM file format: {path}')
        return reader(str(path), **kwargs)
