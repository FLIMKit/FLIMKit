import importlib
import importlib.util
import os
import sys
import threading
import traceback

from flimkit.plugins import registry
from flimkit.plugins.builtin import BUILTIN

_lock = threading.RLock()
_loaded = False
_report = []


class LoadResult:

    def __init__(self, source, ok, n_registered, error=None):
        self.source = source
        self.ok = ok
        self.n_registered = n_registered
        self.error = error

    def __repr__(self):
        state = 'ok' if self.ok else 'failed'
        return f'<LoadResult {self.source!r} {state} n={self.n_registered}>'


def disabled():
    return os.environ.get('FLIMKIT_NO_PLUGINS', '') not in ('', '0', 'false', 'False')


def _record(result):
    _report.append(result)
    return result


def _api_ok(module, source):
    declared = getattr(module, 'FLIMKIT_PLUGIN_API', registry.API_VERSION)
    if declared != registry.API_VERSION:
        raise registry.PluginError(
            f'plugin {source!r} declares FLIMKIT_PLUGIN_API {declared!r}, '
            f'this FLIMKit provides {registry.API_VERSION}')


def _load(source, importer):
    before = len(registry.tools())
    prev = registry._set_source(source)
    print(f'[Plugins] loading {source}')
    try:
        module = importer()
        _api_ok(module, source)
    except KeyboardInterrupt:
        registry._set_source(prev)
        registry._rollback(source)
        raise
    except BaseException:
        registry._set_source(prev)
        dropped = registry._rollback(source)
        err = traceback.format_exc()
        print(f'[Plugins] {source} failed, {len(dropped)} registration(s) rolled back')
        return _record(LoadResult(source, False, 0, err))
    registry._set_source(prev)
    return _record(LoadResult(source, True, len(registry.tools()) - before))


def load_module(dotted, source=None):
    source = source or dotted

    def importer():
        cached = sys.modules.get(dotted)
        if cached is not None:
            return importlib.reload(cached)
        return importlib.import_module(dotted)

    return _load(source, importer)


def load_path(path, source=None):
    path = os.path.abspath(path)
    name = 'flimkit_plugin_' + os.path.splitext(os.path.basename(path))[0]
    source = source or path

    def importer():
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise registry.PluginError(f'cannot import plugin file {path!r}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    return _load(source, importer)


def ensure_loaded():
    global _loaded
    with _lock:
        if _loaded:
            return list(_report)
        _loaded = True
        if disabled():
            print('[Plugins] FLIMKIT_NO_PLUGINS set, loading nothing')
            return list(_report)
        for dotted in BUILTIN:
            load_module(dotted)
        return list(_report)


def load_report():
    return list(_report)


def failures():
    return [r for r in _report if not r.ok]


def reset():
    global _loaded
    with _lock:
        _loaded = False
        _report.clear()
        registry.clear()
