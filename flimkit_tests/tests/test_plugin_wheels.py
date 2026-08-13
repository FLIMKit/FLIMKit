import subprocess
import sys
from pathlib import Path

import pytest

from flimkit import plugins
from flimkit.plugins import loader

REAL_ENTRY_POINTS = loader.entry_points

PLUGIN_SOURCE = (
    'from flimkit.plugins import tool\n'
    'FLIMKIT_PLUGIN_API = 1\n'
    "@tool(id='wheeled', label='From A Wheel...')\n"
    'def open_it(app):\n'
    '    return app\n'
)

PYPROJECT = """
[build-system]
requires = ['setuptools>=68']
build-backend = 'setuptools.build_meta'

[project]
name = 'wheeled-addon'
version = '0.1.0'
description = 'test fixture'

[project.entry-points.'flimkit.plugins']
wheeled = 'wheeled_addon'

[tool.setuptools.packages.find]
include = ['wheeled_addon*']
"""


@pytest.fixture(scope='session')
def built_wheel(tmp_path_factory):
    src = tmp_path_factory.mktemp('wheelsrc')
    (src / 'wheeled_addon').mkdir()
    (src / 'wheeled_addon' / '__init__.py').write_text(PLUGIN_SOURCE)
    (src / 'pyproject.toml').write_text(PYPROJECT)
    out = tmp_path_factory.mktemp('wheelout')
    result = subprocess.run([sys.executable, '-m', 'build', '--wheel', '-o', str(out), str(src)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f'could not build a wheel here: {result.stderr[-200:]}')
    wheels = list(out.glob('*.whl'))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture
def plugin_dir(tmp_path, monkeypatch):
    home = tmp_path / 'plugins'
    home.mkdir()
    monkeypatch.setattr(loader, 'user_dir', lambda: str(home))
    monkeypatch.setattr(loader, 'user_plugins_allowed', lambda: True)
    monkeypatch.setattr(loader, 'entry_points', REAL_ENTRY_POINTS)
    yield home
    for entry in list(sys.path):
        if str(home) in entry:
            sys.path.remove(entry)
    sys.modules.pop('wheeled_addon', None)


def test_a_wheel_is_a_candidate_archive(plugin_dir, built_wheel):
    target = plugin_dir / built_wheel.name
    target.write_bytes(built_wheel.read_bytes())
    assert plugins.archives(str(plugin_dir)) == [str(target)]
    assert plugins.candidates(str(plugin_dir)) == []


def test_a_wheel_in_the_folder_registers_through_its_entry_point(plugin_dir, built_wheel):
    target = plugin_dir / built_wheel.name
    target.write_bytes(built_wheel.read_bytes())
    plugins.ensure_loaded()
    found = plugins.get_tool('wheeled')
    assert found is not None
    assert found.label == 'From A Wheel...'
    assert found.source == 'flimkit.plugins:wheeled'


def test_the_wheel_lands_on_the_path_before_entry_points_are_scanned(plugin_dir, built_wheel):
    target = plugin_dir / built_wheel.name
    target.write_bytes(built_wheel.read_bytes())
    plugins.ensure_loaded()
    assert str(target) in sys.path


def test_the_gate_refuses_a_wheel_too(plugin_dir, built_wheel, monkeypatch):
    target = plugin_dir / built_wheel.name
    target.write_bytes(built_wheel.read_bytes())
    monkeypatch.setattr(loader, 'user_plugins_allowed', lambda: False)
    plugins.ensure_loaded()
    assert plugins.get_tool('wheeled') is None
    assert str(target) not in sys.path


def test_a_disabled_wheel_is_skipped(plugin_dir, built_wheel, monkeypatch):
    from flimkit.utils import config_manager
    target = plugin_dir / built_wheel.name
    target.write_bytes(built_wheel.read_bytes())
    monkeypatch.setattr(config_manager.cfg, '_global', {})
    plugins.set_plugin_disabled(plugins.short_name(str(target)), True)
    plugins.ensure_loaded()
    assert plugins.get_tool('wheeled') is None


def test_a_platform_wheel_is_refused_with_a_reason(plugin_dir):
    target = plugin_dir / 'native-1.0-cp314-cp314-macosx_11_0_arm64.whl'
    target.write_bytes(b'not really a wheel')
    plugins.ensure_loaded()
    failed = [r for r in plugins.load_report() if r.ok == False]
    assert len(failed) == 1
    assert 'py3-none-any' in failed[0].error
    assert str(target) not in sys.path


def test_short_name_of_a_wheel_is_the_distribution():
    assert plugins.short_name('/x/wheeled_addon-0.1.0-py3-none-any.whl') == 'wheeled_addon'
    assert plugins.short_name('/x/bundle.zip') == 'bundle'
    assert plugins.short_name('/x/plain.py') == 'plain'


def test_portable_wheel_reads_the_tag():
    assert plugins.portable_wheel('/x/foo-1.0-py3-none-any.whl') == True
    assert plugins.portable_wheel('/x/foo-1.0-cp314-cp314-manylinux_2_17_x86_64.whl') == False
    assert plugins.portable_wheel('/x/foo-1.0-py3-none-macosx_11_0_arm64.whl') == False
    assert plugins.portable_wheel('/x/bundle.zip') == True
