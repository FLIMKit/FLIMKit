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
    before = registry.count()
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
    return _record(LoadResult(source, True, registry.count() - before))


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
    stem = os.path.basename(path.rstrip(os.sep))
    if os.path.isdir(path):
        init = os.path.join(path, '__init__.py')
        submodules = [path]
    else:
        init = path
        stem = os.path.splitext(stem)[0]
        submodules = None
    name = 'flimkit_plugin_' + stem
    source = source or path

    def importer():
        spec = importlib.util.spec_from_file_location(
            name, init, submodule_search_locations=submodules)
        if spec is None or spec.loader is None:
            raise registry.PluginError(f'cannot import plugin {path!r}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    return _load(source, importer)


def candidates(directory):
    if not os.path.isdir(directory):
        return []
    found = []
    for entry in sorted(os.listdir(directory)):
        if entry.startswith(('_', '.')):
            continue
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path, '__init__.py')):
                found.append(path)
        elif entry.endswith('.py'):
            found.append(path)
    return found


def user_dir():
    return os.path.join(os.path.expanduser('~'), '.flimkit', 'plugins')


def extra_dirs():
    raw = os.environ.get('FLIMKIT_PLUGIN_PATH', '')
    return [os.path.expanduser(p) for p in raw.split(os.pathsep) if p.strip()]


def user_plugins_allowed():
    from flimkit.utils.config_manager import cfg
    return cfg.get('plugins.allow_user_plugins', False) == True


def pending_user_plugins():
    if user_plugins_allowed():
        return []
    return candidates(user_dir())


def allow_user_plugins(allowed=True):
    from flimkit.utils.config_manager import cfg
    cfg.set('plugins.allow_user_plugins', bool(allowed))
    cfg.save()


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
        home = user_dir()
        if user_plugins_allowed():
            for path in candidates(home):
                load_path(path)
        else:
            waiting = candidates(home)
            if waiting:
                print(f'[Plugins] {len(waiting)} plugin(s) in {home} not loaded, '
                      f'set plugins.allow_user_plugins to enable them')
        for directory in extra_dirs():
            for path in candidates(directory):
                load_path(path)
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
