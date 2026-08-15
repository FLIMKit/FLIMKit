import threading

API_VERSION = 1

_lock = threading.RLock()
_tools = {}
_formats = {}
_sniffers = []
_phasor_filters = {}
_panel_buttons = {}
_startups = {}
_version = 0
_current_source = None
MODALITIES = ('time', 'frequency', 'intensity')


class PluginError(Exception):
    pass


class Tool:

    def __init__(self, id, label, callback, menu, order, source):
        self.id = id
        self.label = label
        self.callback = callback
        self.menu = menu
        self.order = order
        self.source = source

    @property
    def menu_path(self):
        return tuple(p.strip() for p in self.menu.split('/') if p.strip())

    def __repr__(self):
        return f'<Tool {self.id} menu={self.menu!r} source={self.source!r}>'


class PanelButton:

    def __init__(self, id, label, callback, panel, order, source):
        self.id = id
        self.label = label
        self.callback = callback
        self.panel = panel
        self.order = order
        self.source = source

    def __repr__(self):
        return f'<PanelButton {self.id} panel={self.panel!r} source={self.source!r}>'


class Startup:

    def __init__(self, id, callback, order, source):
        self.id = id
        self.callback = callback
        self.order = order
        self.source = source

    def __repr__(self):
        return f'<Startup {self.id} source={self.source!r}>'


class Format:

    def __init__(self, id, label, exts, modality, reader, source):
        self.id = id
        self.label = label
        self.exts = tuple(e.lower() for e in exts)
        self.modality = modality
        self._reader = reader
        self.source = source

    def reader(self):
        if isinstance(self._reader, str):
            import importlib
            mod_name, _, cls_name = self._reader.partition(':')
            if not cls_name:
                raise PluginError(
                    f'format {self.id!r} reader {self._reader!r} is not module:Class')
            return getattr(importlib.import_module(mod_name), cls_name)
        return self._reader

    def __repr__(self):
        return f'<Format {self.id} exts={self.exts} source={self.source!r}>'


class Sniffer:

    def __init__(self, fn, tier, order, source):
        self.fn = fn
        self.tier = tier
        self.order = order
        self.source = source

    def __call__(self, path):
        return self.fn(path)

    def __repr__(self):
        return f'<Sniffer tier={self.tier} order={self.order} source={self.source!r}>'


def _source():
    return _current_source or '<direct>'


def _bump():
    global _version
    _version += 1


def version():
    return _version


def count():
    with _lock:
        return (len(_tools) + len(_formats) + len(_sniffers)
                + len(_phasor_filters) + len(_panel_buttons) + len(_startups))


def register_tool(id, label, callback, menu='Tools', order=100, source=None):
    if not id or not isinstance(id, str):
        raise PluginError(f'tool id must be a non-empty string, got {id!r}')
    if not callable(callback):
        raise PluginError(f'tool {id!r} callback is not callable: {callback!r}')
    with _lock:
        existing = _tools.get(id)
        if existing is not None:
            raise PluginError(
                f'tool id {id!r} already registered by {existing.source!r}, '
                f'refused from {source or _source()!r}')
        _tools[id] = Tool(id, label, callback, menu, order, source or _source())
        _bump()
    return _tools[id]


def tool(id, label, menu='Tools', order=100):
    def decorate(fn):
        register_tool(id, label, fn, menu=menu, order=order)
        return fn
    return decorate


def tools(menu=None):
    with _lock:
        found = list(_tools.values())
    if menu is not None:
        want = tuple(p.strip() for p in menu.split('/') if p.strip())
        found = [t for t in found if t.menu_path[:len(want)] == want]
    found.sort(key=lambda t: (t.order, t.label, t.id))
    return found


def get_tool(id):
    with _lock:
        return _tools.get(id)


PANELS = ('roi',)


def register_panel_button(id, label, callback, panel='roi', order=100, source=None):
    if not id or not isinstance(id, str):
        raise PluginError(f'panel button id must be a non-empty string, got {id!r}')
    if not callable(callback):
        raise PluginError(f'panel button {id!r} callback is not callable: {callback!r}')
    if panel not in PANELS:
        raise PluginError(f'unknown panel {panel!r}, expected one of {PANELS}')
    with _lock:
        existing = _panel_buttons.get(id)
        if existing is not None:
            raise PluginError(
                f'panel button id {id!r} already registered by {existing.source!r}, '
                f'refused from {source or _source()!r}')
        _panel_buttons[id] = PanelButton(
            id, label, callback, panel, order, source or _source())
        _bump()
    return _panel_buttons[id]


def panel_button(id, label, panel='roi', order=100):
    def decorate(fn):
        register_panel_button(id, label, fn, panel=panel, order=order)
        return fn
    return decorate


def panel_buttons(panel=None):
    with _lock:
        found = list(_panel_buttons.values())
    if panel is not None:
        found = [b for b in found if b.panel == panel]
    found.sort(key=lambda b: (b.order, b.label, b.id))
    return found


def get_panel_button(id):
    with _lock:
        return _panel_buttons.get(id)


def register_startup(id, callback, order=100, source=None):
    if not id or not isinstance(id, str):
        raise PluginError(f'startup id must be a non-empty string, got {id!r}')
    if not callable(callback):
        raise PluginError(f'startup {id!r} callback is not callable: {callback!r}')
    with _lock:
        existing = _startups.get(id)
        if existing is not None:
            raise PluginError(
                f'startup id {id!r} already registered by {existing.source!r}, '
                f'refused from {source or _source()!r}')
        _startups[id] = Startup(id, callback, order, source or _source())
        _bump()
    return _startups[id]


def startup(id, order=100):
    def decorate(fn):
        register_startup(id, fn, order=order)
        return fn
    return decorate


def startups():
    with _lock:
        found = list(_startups.values())
    found.sort(key=lambda s: (s.order, s.id))
    return found


def run_startups(app):
    failures = []
    for entry in startups():
        try:
            entry.callback(app)
        except Exception as exc:
            failures.append((entry, exc))
    return failures


def register_format(id, label, exts=(), modality='time', reader=None, source=None):
    if not id or not isinstance(id, str):
        raise PluginError(f'format id must be a non-empty string, got {id!r}')
    if modality not in MODALITIES:
        raise PluginError(
            f'format {id!r} modality {modality!r} is not one of {MODALITIES}')
    if reader is None:
        raise PluginError(f'format {id!r} has no reader')
    if isinstance(exts, str):
        exts = (exts,)
    with _lock:
        existing = _formats.get(id)
        if existing is not None:
            raise PluginError(
                f'format id {id!r} already registered by {existing.source!r}, '
                f'refused from {source or _source()!r}')
        _formats[id] = Format(id, label, exts, modality, reader, source or _source())
        _bump()
    return _formats[id]


def file_format(id, label, exts=(), modality='time', reader=None):
    def decorate(obj):
        register_format(id, label, exts=exts, modality=modality,
                        reader=reader if reader is not None else obj)
        return obj
    return decorate


def formats():
    with _lock:
        return sorted(_formats.values(), key=lambda f: f.id)


def get_format(id):
    with _lock:
        return _formats.get(id)


def register_sniffer(fn, tier='magic', order=100, source=None):
    if not callable(fn):
        raise PluginError(f'sniffer is not callable: {fn!r}')
    if tier not in ('extension', 'magic'):
        raise PluginError(f'sniffer tier {tier!r} is not extension or magic')
    with _lock:
        found = Sniffer(fn, tier, order, source or _source())
        _sniffers.append(found)
        _bump()
    return found


def format_sniffer(tier='magic', order=100):
    def decorate(fn):
        register_sniffer(fn, tier=tier, order=order)
        return fn
    return decorate


def sniffers(tier=None):
    with _lock:
        found = list(_sniffers)
    if tier is not None:
        found = [s for s in found if s.tier == tier]
    found.sort(key=lambda s: s.order)
    return found


class PhasorFilter:

    def __init__(self, id, label, fn, source):
        self.id = id
        self.label = label
        self.fn = fn
        self.source = source

    def __call__(self, real, imag, **kwargs):
        import inspect
        params = inspect.signature(self.fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            allowed = kwargs
        else:
            allowed = {k: v for k, v in kwargs.items() if k in params}
        return self.fn(real, imag, **allowed)

    def __repr__(self):
        return f'<PhasorFilter {self.id} source={self.source!r}>'


def register_phasor_filter(id, label, fn, source=None):
    if not id or not isinstance(id, str):
        raise PluginError(f'phasor filter id must be a non-empty string, got {id!r}')
    if not callable(fn):
        raise PluginError(f'phasor filter {id!r} is not callable: {fn!r}')
    id = id.lower()
    with _lock:
        existing = _phasor_filters.get(id)
        if existing is not None:
            raise PluginError(
                f'phasor filter id {id!r} already registered by {existing.source!r}, '
                f'refused from {source or _source()!r}')
        _phasor_filters[id] = PhasorFilter(id, label, fn, source or _source())
        _bump()
    return _phasor_filters[id]


def phasor_filter(id, label):
    def decorate(fn):
        register_phasor_filter(id, label, fn)
        return fn
    return decorate


def phasor_filters():
    with _lock:
        return sorted(_phasor_filters.values(), key=lambda f: f.id)


def get_phasor_filter(id):
    with _lock:
        return _phasor_filters.get(str(id).lower())


def sources():
    with _lock:
        every = (list(_tools.values()) + list(_formats.values()) + list(_sniffers)
                 + list(_phasor_filters.values()))
    return sorted({t.source for t in every})


def _set_source(name):
    global _current_source
    prev = _current_source
    _current_source = name
    return prev


def _rollback(source):
    with _lock:
        dropped = [k for k, t in _tools.items() if t.source == source]
        for k in dropped:
            del _tools[k]
        dropped += [k for k, f in _formats.items() if f.source == source]
        for k in [k for k, f in _formats.items() if f.source == source]:
            del _formats[k]
        keep = [s for s in _sniffers if s.source != source]
        dropped += ['sniffer'] * (len(_sniffers) - len(keep))
        _sniffers[:] = keep
        dropped += [k for k, f in _phasor_filters.items() if f.source == source]
        for k in [k for k, f in _phasor_filters.items() if f.source == source]:
            del _phasor_filters[k]
        _bump()
    return dropped


def clear():
    with _lock:
        _tools.clear()
        _formats.clear()
        _sniffers.clear()
        _phasor_filters.clear()
        _panel_buttons.clear()
        _startups.clear()
        _bump()
