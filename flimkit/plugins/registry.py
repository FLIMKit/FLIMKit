import threading

API_VERSION = 1

_lock = threading.RLock()
_tools = {}
_current_source = None


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


def _source():
    return _current_source or '<direct>'


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


def sources():
    with _lock:
        return sorted({t.source for t in _tools.values()})


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
    return dropped


def clear():
    with _lock:
        _tools.clear()
