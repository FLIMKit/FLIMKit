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


def test_a_failing_tool_is_reported_not_raised(clean_registry, monkeypatch):
    plugins.register_tool('boom', 'Boom', lambda app: 1 / 0)
    from flimkit.UI.gui import _UIBuilder
    shown = []
    monkeypatch.setattr('flimkit.UI.gui.messagebox.showerror',
                        lambda title, msg: shown.append(msg))
    b = _UIBuilder.__new__(_UIBuilder)
    b._run_plugin_tool(plugins.get_tool('boom'))
    assert 'ZeroDivisionError' in shown[0]
