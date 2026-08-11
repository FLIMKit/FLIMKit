import os
import sys
from pathlib import Path

import pytest

from flimkit import plugins
from flimkit.plugins import loader, registry
from flimkit.plugins.builtin import BUILTIN

REPO = Path(__file__).resolve().parents[2]
BUILTIN_IDS = {'irf_builder', 'synth_generator', 'batch_tiled', 'batch_fov', 'batch_timelapse'}


@pytest.fixture
def clean_registry():
    loader.reset()
    yield
    loader.reset()
    plugins.ensure_loaded()


def write_plugin(tmp_path, name, body):
    path = tmp_path / (name + '.py')
    path.write_text(body)
    return path


def test_builtins_register_every_tools_menu_entry(clean_registry):
    plugins.ensure_loaded()
    assert {t.id for t in plugins.tools()} == BUILTIN_IDS


def test_builtin_menu_order_matches_the_old_hard_coded_menu(clean_registry):
    plugins.ensure_loaded()
    top = [t.label for t in plugins.tools('Tools') if t.menu_path == ('Tools',)]
    assert top == ['Machine IRF Builder', 'Generate Synthetic PTU...']
    batch = [t.label for t in plugins.tools('Tools/Batch Processing')]
    assert batch == ['Multi-Tile ROI Fit', 'Single FOV Fit', 'Timelapse Fit']


def test_builtin_tuple_matches_the_directory(clean_registry):
    directory = REPO / 'flimkit' / 'plugins' / 'builtin'
    on_disk = {'flimkit.plugins.builtin.' + p.stem
               for p in directory.glob('*.py') if p.stem != '__init__'}
    assert set(BUILTIN) == on_disk


def test_no_plugins_env_var_loads_nothing(clean_registry, monkeypatch):
    monkeypatch.setenv('FLIMKIT_NO_PLUGINS', '1')
    plugins.ensure_loaded()
    assert plugins.tools() == []
    assert plugins.load_report() == []


def test_ensure_loaded_is_idempotent(clean_registry):
    plugins.ensure_loaded()
    first = plugins.load_report()
    plugins.ensure_loaded()
    assert plugins.load_report() == first
    assert {t.id for t in plugins.tools()} == BUILTIN_IDS


def test_duplicate_id_is_refused(clean_registry):
    plugins.register_tool('dup', 'One', lambda app: None)
    with pytest.raises(registry.PluginError):
        plugins.register_tool('dup', 'Two', lambda app: None)


def test_duplicate_id_rolls_back_the_whole_offending_plugin(clean_registry, tmp_path):
    plugins.ensure_loaded()
    path = write_plugin(tmp_path, 'clashing', (
        'from flimkit.plugins import tool\n'
        "@tool(id='clash_first', label='First')\n"
        'def a(app):\n'
        '    pass\n'
        "@tool(id='irf_builder', label='Stolen')\n"
        'def b(app):\n'
        '    pass\n'
    ))
    result = plugins.load_path(str(path))
    assert result.ok == False
    assert plugins.get_tool('clash_first') is None
    assert plugins.get_tool('irf_builder').source == 'flimkit.plugins.builtin.core_tools'


def test_broken_plugin_does_not_stop_the_next_one(clean_registry, tmp_path):
    bad = write_plugin(tmp_path, 'bad', 'raise RuntimeError("boom")\n')
    good = write_plugin(tmp_path, 'good', (
        'from flimkit.plugins import tool\n'
        "@tool(id='good_tool', label='Good')\n"
        'def a(app):\n'
        '    pass\n'
    ))
    first = plugins.load_path(str(bad))
    second = plugins.load_path(str(good))
    assert first.ok == False
    assert 'boom' in first.error
    assert second.ok == True
    assert plugins.get_tool('good_tool') is not None
    assert len(plugins.failures()) == 1


def test_plugin_calling_sys_exit_does_not_kill_the_app(clean_registry, tmp_path):
    path = write_plugin(tmp_path, 'suicidal', 'import sys\nsys.exit(3)\n')
    result = plugins.load_path(str(path))
    assert result.ok == False
    assert 'SystemExit' in result.error


def test_api_version_mismatch_is_refused(clean_registry, tmp_path):
    path = write_plugin(tmp_path, 'future', (
        'from flimkit.plugins import tool\n'
        'FLIMKIT_PLUGIN_API = 99\n'
        "@tool(id='future_tool', label='Future')\n"
        'def a(app):\n'
        '    pass\n'
    ))
    result = plugins.load_path(str(path))
    assert result.ok == False
    assert 'FLIMKIT_PLUGIN_API' in result.error
    assert plugins.get_tool('future_tool') is None


def test_example_plugin_loads(clean_registry):
    plugins.ensure_loaded()
    path = REPO / 'examples' / 'plugins' / 'hello_tool.py'
    result = plugins.load_path(str(path))
    assert result.ok == True
    hello = plugins.get_tool('hello_example')
    assert hello.label == 'Hello Plugin...'
    assert hello.menu_path == ('Tools',)
    assert hello.source == str(path)


def test_registration_records_its_source(clean_registry):
    plugins.ensure_loaded()
    assert plugins.sources() == ['flimkit.plugins.builtin.core_tools']
    plugins.register_tool('manual', 'Manual', lambda app: None)
    assert plugins.get_tool('manual').source == '<direct>'


def test_non_callable_callback_is_refused(clean_registry):
    with pytest.raises(registry.PluginError):
        plugins.register_tool('nope', 'Nope', 'not a function')


def test_plugin_format_is_detected_by_extension(clean_registry, tmp_path):
    from flimkit.formats import detect_format, file_modality, supported_extensions
    plugins.ensure_loaded()
    path = write_plugin(tmp_path, 'fmt', (
        'from flimkit.plugins import file_format\n'
        "@file_format(id='mine', label='My Format', exts=('.mine',), modality='frequency')\n"
        'class MyReader:\n'
        '    def __init__(self, path, **kw):\n'
        '        self.path = path\n'
    ))
    assert plugins.load_path(str(path)).ok == True
    assert detect_format('a.mine') == 'mine'
    assert file_modality('a.mine') == 'frequency'
    assert '.mine' in supported_extensions()


def test_plugin_format_opens_through_flimfile(clean_registry, tmp_path):
    from flimkit.formats import FLIMFile
    plugins.ensure_loaded()
    path = write_plugin(tmp_path, 'fmt2', (
        'from flimkit.plugins import file_format\n'
        "@file_format(id='mine2', label='My Format 2', exts=('.mine2',))\n"
        'class MyReader:\n'
        '    def __init__(self, path, **kw):\n'
        '        self.path = path\n'
    ))
    plugins.load_path(str(path))
    handle = FLIMFile('a.mine2')
    assert handle.path == 'a.mine2'


def test_plugin_sniffer_runs_after_the_builtin_magic(clean_registry, tmp_path):
    from flimkit.formats import detect_format
    plugins.ensure_loaded()
    target = tmp_path / 'thing.unknownext'
    target.write_bytes(b'MYMAGIC and then some')
    path = write_plugin(tmp_path, 'sniffer', (
        'from flimkit.plugins import file_format, format_sniffer\n'
        "@file_format(id='magicfmt', label='Magic Format', reader='flimkit.formats.PTU.reader:PTUFile')\n"
        'def _unused():\n'
        '    pass\n'
        "@format_sniffer(tier='magic')\n"
        'def sniff(p):\n'
        "    with open(p, 'rb') as fh:\n"
        "        if fh.read(7) == b'MYMAGIC':\n"
        "            return 'magicfmt'\n"
        '    return None\n'
    ))
    plugins.load_path(str(path))
    assert detect_format(str(target)) == 'magicfmt'


def test_a_raising_sniffer_does_not_break_detection(clean_registry, tmp_path):
    from flimkit.formats import detect_format
    plugins.ensure_loaded()
    path = write_plugin(tmp_path, 'badsniff', (
        'from flimkit.plugins import format_sniffer\n'
        "@format_sniffer(tier='magic')\n"
        'def sniff(p):\n'
        '    raise ValueError("bad sniffer")\n'
    ))
    plugins.load_path(str(path))
    assert detect_format('a.ptu') == 'ptu'
    assert detect_format('a.nothing') == 'unknown'


def test_duplicate_format_id_is_refused(clean_registry):
    plugins.register_format('dupfmt', 'One', exts=('.one',), reader='a:B')
    with pytest.raises(registry.PluginError):
        plugins.register_format('dupfmt', 'Two', exts=('.two',), reader='a:B')


def test_bad_modality_is_refused(clean_registry):
    with pytest.raises(registry.PluginError):
        plugins.register_format('badmod', 'Bad', exts=('.bad',),
                                modality='wavelength', reader='a:B')


def test_format_registrations_roll_back_with_their_plugin(clean_registry, tmp_path):
    from flimkit.formats import detect_format
    plugins.ensure_loaded()
    path = write_plugin(tmp_path, 'halfbad', (
        'from flimkit.plugins import file_format, format_sniffer\n'
        "@file_format(id='rollback_fmt', label='Rollback', exts=('.rb',))\n"
        'class R:\n'
        '    pass\n'
        "@format_sniffer(tier='magic')\n"
        'def sniff(p):\n'
        '    return None\n'
        'raise RuntimeError("too late")\n'
    ))
    assert plugins.load_path(str(path)).ok == False
    assert plugins.get_format('rollback_fmt') is None
    assert plugins.sniffers() == []
    assert detect_format('a.rb') == 'unknown'


def test_plugin_phasor_filter_runs(clean_registry, tmp_path):
    import numpy as np
    from flimkit.phasor.filters import phasor_filter, phasor_filter_methods
    plugins.ensure_loaded()
    path = write_plugin(tmp_path, 'pf', (
        'from flimkit.plugins import phasor_filter\n'
        "@phasor_filter(id='double', label='Double it')\n"
        'def double(real, imag):\n'
        '    return real * 2, imag * 2\n'
    ))
    assert plugins.load_path(str(path)).ok == True
    real = np.ones((2, 2))
    imag = np.ones((2, 2))
    out_real, out_imag = phasor_filter(real, imag, 'double')
    assert out_real.max() == 2.0
    assert out_imag.max() == 2.0
    assert 'double' in phasor_filter_methods()


def test_plugin_phasor_filter_only_gets_the_kwargs_it_asks_for(clean_registry):
    import numpy as np
    from flimkit.phasor.filters import phasor_filter
    seen = {}

    def scaled(real, imag, sigma=1.0):
        seen['sigma'] = sigma
        return real * sigma, imag
    plugins.register_phasor_filter('scaled', 'Scaled', scaled)
    phasor_filter(np.ones((2, 2)), np.ones((2, 2)), 'scaled', sigma=3.0, size=9)
    assert seen == {'sigma': 3.0}


def test_builtin_phasor_filters_still_win(clean_registry):
    import numpy as np
    from flimkit.phasor.filters import phasor_filter
    plugins.register_phasor_filter('gaussian', 'Hijack', lambda real, imag: (real * 0, imag * 0))
    out_real, _ = phasor_filter(np.ones((4, 4)), np.ones((4, 4)), 'gaussian', sigma=1.0)
    assert out_real.max() > 0


def test_unknown_phasor_filter_lists_what_is_available(clean_registry):
    import numpy as np
    plugins.register_phasor_filter('custom', 'Custom', lambda real, imag: (real, imag))
    from flimkit.phasor.filters import phasor_filter
    with pytest.raises(ValueError) as excinfo:
        phasor_filter(np.ones((2, 2)), np.ones((2, 2)), 'nonesuch')
    assert 'custom' in str(excinfo.value)


@pytest.fixture
def user_dir(clean_registry, tmp_path, monkeypatch):
    home = tmp_path / '.flimkit' / 'plugins'
    home.mkdir(parents=True)
    monkeypatch.setattr(loader, 'user_dir', lambda: str(home))
    monkeypatch.setattr(loader, 'user_plugins_allowed', lambda: True)
    return home


def test_user_directory_is_scanned_when_allowed(user_dir):
    write_plugin(user_dir, 'mine', (
        'from flimkit.plugins import tool\n'
        "@tool(id='mine', label='Mine')\n"
        'def a(app):\n'
        '    pass\n'
    ))
    plugins.ensure_loaded()
    assert plugins.get_tool('mine').source == str(user_dir / 'mine.py')


def test_user_directory_is_skipped_when_not_allowed(clean_registry, tmp_path, monkeypatch):
    home = tmp_path / 'plugins'
    home.mkdir()
    write_plugin(home, 'mine', (
        'from flimkit.plugins import tool\n'
        "@tool(id='mine', label='Mine')\n"
        'def a(app):\n'
        '    pass\n'
    ))
    monkeypatch.setattr(loader, 'user_dir', lambda: str(home))
    monkeypatch.setattr(loader, 'user_plugins_allowed', lambda: False)
    plugins.ensure_loaded()
    assert plugins.get_tool('mine') is None
    assert plugins.pending_user_plugins() == [str(home / 'mine.py')]


def test_underscored_and_non_python_files_are_ignored(user_dir):
    (user_dir / '_private.py').write_text('raise RuntimeError("must not load")\n')
    (user_dir / 'notes.txt').write_text('hello\n')
    (user_dir / '.hidden.py').write_text('raise RuntimeError("must not load")\n')
    plugins.ensure_loaded()
    assert plugins.failures() == []


def test_package_directory_loads(user_dir):
    pkg = user_dir / 'toolpack'
    pkg.mkdir()
    (pkg / 'helper.py').write_text("LABEL = 'From A Package'\n")
    (pkg / '__init__.py').write_text(
        'from flimkit.plugins import tool\n'
        'from .helper import LABEL\n'
        "@tool(id='packaged', label=LABEL)\n"
        'def a(app):\n'
        '    pass\n'
    )
    plugins.ensure_loaded()
    assert plugins.get_tool('packaged').label == 'From A Package'


def test_user_plugins_load_after_builtins(user_dir):
    write_plugin(user_dir, 'later', (
        'from flimkit.plugins import tool\n'
        "@tool(id='later', label='Later')\n"
        'def a(app):\n'
        '    pass\n'
    ))
    plugins.ensure_loaded()
    order = [r.source for r in plugins.load_report()]
    assert order[0] == 'flimkit.plugins.builtin.core_tools'
    assert order[-1] == str(user_dir / 'later.py')


def test_no_plugins_env_var_beats_the_user_directory(user_dir, monkeypatch):
    write_plugin(user_dir, 'mine', (
        'from flimkit.plugins import tool\n'
        "@tool(id='mine', label='Mine')\n"
        'def a(app):\n'
        '    pass\n'
    ))
    monkeypatch.setenv('FLIMKIT_NO_PLUGINS', '1')
    plugins.ensure_loaded()
    assert plugins.tools() == []


def test_plugin_path_env_var_is_scanned(clean_registry, tmp_path, monkeypatch):
    extra = tmp_path / 'extra'
    extra.mkdir()
    write_plugin(extra, 'fromenv', (
        'from flimkit.plugins import tool\n'
        "@tool(id='fromenv', label='From Env')\n"
        'def a(app):\n'
        '    pass\n'
    ))
    monkeypatch.setattr(loader, 'user_dir', lambda: str(tmp_path / 'missing'))
    monkeypatch.setenv('FLIMKIT_PLUGIN_PATH', str(extra))
    plugins.ensure_loaded()
    assert plugins.get_tool('fromenv') is not None


def test_a_broken_user_plugin_leaves_the_builtins_alone(user_dir):
    write_plugin(user_dir, 'bad', 'raise ValueError("nope")\n')
    plugins.ensure_loaded()
    assert {t.id for t in plugins.tools()} == BUILTIN_IDS
    assert len(plugins.failures()) == 1


def test_missing_user_directory_is_not_created(clean_registry, tmp_path, monkeypatch):
    home = tmp_path / 'never'
    monkeypatch.setattr(loader, 'user_dir', lambda: str(home))
    monkeypatch.setattr(loader, 'user_plugins_allowed', lambda: True)
    plugins.ensure_loaded()
    assert home.exists() == False


def menu_labels(menu):
    out = []
    for i in range(menu.index('end') + 1):
        if menu.type(i) in ('command', 'cascade'):
            out.append(menu.entrycget(i, 'label'))
    return out


@pytest.fixture
def menu_app(clean_registry):
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    root.withdraw()
    from flimkit.UI.gui import _UIBuilder
    b = _UIBuilder.__new__(_UIBuilder)
    b.root = root
    b._build_menu_bar()
    yield b
    root.destroy()


def submenu(menu, label):
    for i in range(menu.index('end') + 1):
        if menu.type(i) == 'cascade' and menu.entrycget(i, 'label') == label:
            return menu.nametowidget(menu.entrycget(i, 'menu'))
    raise AssertionError(f'no submenu {label!r} in {menu_labels(menu)}')


def test_tools_menu_is_unchanged_by_the_registry(menu_app):
    menubar = menu_app.root.nametowidget(menu_app.root['menu'])
    tools = submenu(menubar, 'Tools')
    assert menu_labels(tools) == ['Machine IRF Builder',
                                  'Generate Synthetic PTU...',
                                  'Batch Processing']
    assert menu_labels(submenu(tools, 'Batch Processing')) == ['Multi-Tile ROI Fit',
                                                              'Single FOV Fit',
                                                              'Timelapse Fit']


def test_a_plugin_tool_appears_in_the_menu(clean_registry):
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    root.withdraw()
    plugins.ensure_loaded()
    plugins.load_path(str(REPO / 'examples' / 'plugins' / 'hello_tool.py'))
    from flimkit.UI.gui import _UIBuilder
    b = _UIBuilder.__new__(_UIBuilder)
    b.root = root
    b._build_menu_bar()
    menubar = root.nametowidget(root['menu'])
    assert 'Hello Plugin...' in menu_labels(submenu(menubar, 'Tools'))
    root.destroy()


def test_user_plugins_are_off_by_default(clean_registry, monkeypatch):
    from flimkit.utils import config_manager
    monkeypatch.setattr(config_manager.cfg, '_global', {})
    monkeypatch.setattr(config_manager.cfg, '_project', {})
    assert loader.user_plugins_allowed() == False


def test_enabling_user_plugins_writes_the_config_key(clean_registry, monkeypatch):
    from flimkit.utils import config_manager
    monkeypatch.setattr(config_manager.cfg, '_global', {})
    monkeypatch.setattr(config_manager.cfg, '_project', {})
    monkeypatch.setattr(config_manager.cfg, 'save', lambda: None)
    loader.allow_user_plugins(True)
    assert loader.user_plugins_allowed() == True


def test_help_menu_offers_the_plugins_dialog(menu_app):
    menubar = menu_app.root.nametowidget(menu_app.root['menu'])
    assert 'Plugins...' in menu_labels(submenu(menubar, 'Help'))


def test_plugin_report_names_what_loaded(menu_app):
    report = menu_app._plugin_report_text()
    assert 'flimkit.plugins.builtin.core_tools' in report
    assert 'User plugin folder' in report


def test_plugin_report_includes_the_traceback_of_a_failure(menu_app, tmp_path):
    path = write_plugin(tmp_path, 'bad', 'raise ValueError("nope")\n')
    plugins.load_path(str(path))
    report = menu_app._plugin_report_text()
    assert 'FAILED' in report
    assert 'ValueError: nope' in report


def test_startup_prompt_asks_once(clean_registry, monkeypatch):
    from flimkit.UI.gui import _UIBuilder
    from flimkit.utils import config_manager
    monkeypatch.setattr(config_manager.cfg, '_global', {})
    monkeypatch.setattr(config_manager.cfg, '_project', {})
    monkeypatch.setattr(config_manager.cfg, 'save', lambda: None)
    monkeypatch.setattr(plugins, 'pending_user_plugins', lambda: ['/somewhere/mine.py'])
    asked = []
    b = _UIBuilder.__new__(_UIBuilder)
    monkeypatch.setattr(_UIBuilder, '_enable_user_plugins',
                        lambda self, win=None: asked.append(1))
    b._maybe_prompt_user_plugins()
    b._maybe_prompt_user_plugins()
    assert len(asked) == 1


def test_startup_prompt_stays_quiet_with_an_empty_folder(clean_registry, monkeypatch):
    from flimkit.UI.gui import _UIBuilder
    from flimkit.utils import config_manager
    monkeypatch.setattr(config_manager.cfg, '_global', {})
    monkeypatch.setattr(config_manager.cfg, '_project', {})
    monkeypatch.setattr(plugins, 'pending_user_plugins', lambda: [])
    asked = []
    b = _UIBuilder.__new__(_UIBuilder)
    monkeypatch.setattr(_UIBuilder, '_enable_user_plugins',
                        lambda self, win=None: asked.append(1))
    b._maybe_prompt_user_plugins()
    assert asked == []


def test_a_failing_tool_is_reported_not_raised(clean_registry, monkeypatch):
    plugins.register_tool('boom', 'Boom', lambda app: 1 / 0)
    from flimkit.UI.gui import _UIBuilder
    shown = []
    monkeypatch.setattr('flimkit.UI.gui.messagebox.showerror',
                        lambda title, msg: shown.append(msg))
    b = _UIBuilder.__new__(_UIBuilder)
    b._run_plugin_tool(plugins.get_tool('boom'))
    assert 'ZeroDivisionError' in shown[0]
