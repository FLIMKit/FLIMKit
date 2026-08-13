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
_skipped = []


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


def archives(directory):
    if not os.path.isdir(directory):
        return []
    found = []
    for entry in sorted(os.listdir(directory)):
        if entry.startswith(('_', '.')):
            continue
        if entry.endswith(('.whl', '.zip')):
            found.append(os.path.join(directory, entry))
    return found


def portable_wheel(path):
    name = os.path.basename(path)
    if not name.endswith('.whl'):
        return True
    parts = name[:-4].split('-')
    if len(parts) < 5:
        return False
    return parts[-1] == 'any' and parts[-2] == 'none'


def add_archive(path):
    name = short_name(path)
    if name in disabled_plugins():
        print(f'[Plugins] {name} disabled, skipping')
        _skipped.append(name)
        return False
    if not portable_wheel(path):
        print(f'[Plugins] {name} is built for one platform and cannot be imported '
              f'from a zip, install it with pip instead')
        _record(LoadResult(path, False, 0,
                           'wheel is not py3-none-any, so it cannot be imported from a zip'))
        return False
    if path not in sys.path:
        sys.path.insert(0, path)
        importlib.invalidate_caches()
        print(f'[Plugins] added {path} to the import path')
    return True


def user_dir():
    return os.path.join(os.path.expanduser('~'), '.flimkit', 'plugins')


def extra_dirs():
    raw = os.environ.get('FLIMKIT_PLUGIN_PATH', '')
    found = [os.path.expanduser(p) for p in raw.split(os.pathsep) if p.strip()]
    for p in config_dirs():
        if p not in found:
            found.append(p)
    return found


def config_dirs():
    from flimkit.utils.config_manager import cfg
    found = cfg.get('plugins.paths', [])
    if isinstance(found, str):
        found = found.split(os.pathsep)
    if isinstance(found, (list, tuple)) == False:
        return []
    return [os.path.expanduser(str(p)) for p in found if str(p).strip()]


def set_config_dirs(paths):
    from flimkit.utils.config_manager import cfg
    cfg.set('plugins.paths', [str(p) for p in paths if str(p).strip()])
    cfg.save()


def plugins_enabled():
    from flimkit.utils.config_manager import cfg
    return cfg.get('plugins.enabled', True) != False


def set_plugins_enabled(enabled=True):
    from flimkit.utils.config_manager import cfg
    cfg.set('plugins.enabled', bool(enabled))
    cfg.save()


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


def short_name(source):
    name = os.path.basename(str(source).rstrip(os.sep))
    if name.endswith(('.whl', '.zip')):
        return name.rsplit('.', 1)[0].split('-')[0]
    if name.endswith('.py'):
        name = name[:-3]
    if '.' in name and os.sep not in name:
        name = name.rsplit('.', 1)[-1]
    return name


def disabled_plugins():
    from flimkit.utils.config_manager import cfg
    found = cfg.get('plugins.disabled', [])
    if isinstance(found, (list, tuple)) == False:
        return []
    return [str(x) for x in found]


def set_plugin_disabled(name, disabled=True):
    from flimkit.utils.config_manager import cfg
    current = [n for n in disabled_plugins() if n != name]
    if disabled:
        current.append(name)
    cfg.set('plugins.disabled', sorted(current))
    cfg.save()
    return current


def plugin_config(name):
    return PluginConfig(name)


class PluginConfig:

    def __init__(self, name):
        self.section = 'plugin:' + short_name(name)

    def get(self, key, default=None):
        from flimkit.utils.config_manager import cfg
        return cfg.get(self.section + '.' + key, default)

    def set(self, key, value):
        from flimkit.utils.config_manager import cfg
        cfg.set(self.section + '.' + key, value)

    def save(self):
        from flimkit.utils.config_manager import cfg
        cfg.save()

    def all(self):
        from flimkit.utils.config_manager import cfg
        return cfg.get(self.section)

    def __repr__(self):
        return f'<PluginConfig {self.section!r}>'


ENTRY_POINT_GROUP = 'flimkit.plugins'


def entry_points():
    try:
        from importlib.metadata import entry_points as _eps
    except ImportError:
        return []
    try:
        return sorted(_eps(group=ENTRY_POINT_GROUP), key=lambda e: e.name)
    except Exception as exc:
        print(f'[Plugins] entry point lookup failed: {exc}')
        return []


def _load_entry_point(ep):
    if ep.name in disabled_plugins():
        print(f'[Plugins] {ep.name} disabled, skipping')
        _skipped.append(ep.name)
        return None
    source = f'{ENTRY_POINT_GROUP}:{ep.name}'
    return _load(source, ep.load)


def _load_unless_disabled(target, is_module=False):
    name = short_name(target)
    if name in disabled_plugins():
        print(f'[Plugins] {name} disabled, skipping')
        _skipped.append(name)
        return None
    return load_module(target) if is_module else load_path(target)


def ensure_loaded():
    global _loaded
    with _lock:
        if _loaded:
            return list(_report)
        _loaded = True
        if disabled():
            print('[Plugins] FLIMKIT_NO_PLUGINS set, loading nothing')
            return list(_report)
        if plugins_enabled() == False:
            print('[Plugins] plugins.enabled is false in the config, loading nothing')
            return list(_report)
        for dotted in BUILTIN:
            _load_unless_disabled(dotted, is_module=True)
        home = user_dir()
        allowed = user_plugins_allowed()
        if allowed:
            for path in archives(home):
                add_archive(path)
        for directory in extra_dirs():
            for path in archives(directory):
                add_archive(path)
        for ep in entry_points():
            _load_entry_point(ep)
        if allowed:
            for path in candidates(home):
                _load_unless_disabled(path)
        else:
            waiting = candidates(home) + archives(home)
            if waiting:
                print(f'[Plugins] {len(waiting)} plugin(s) in {home} not loaded, '
                      f'set plugins.allow_user_plugins to enable them')
        for directory in extra_dirs():
            for path in candidates(directory):
                _load_unless_disabled(path)
        return list(_report)


def load_report():
    return list(_report)


def failures():
    return [r for r in _report if not r.ok]


def skipped():
    return list(_skipped)


def reset():
    global _loaded
    with _lock:
        _loaded = False
        _report.clear()
        _skipped.clear()
        registry.clear()
