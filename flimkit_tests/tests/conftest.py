import os

import pytest


@pytest.fixture(autouse=True)
def isolate_plugins(tmp_path_factory, monkeypatch):
    from flimkit.plugins import loader
    from flimkit.utils import config_manager
    empty = tmp_path_factory.mktemp('no_plugins')
    monkeypatch.setattr(loader, 'user_dir', lambda: str(empty / 'never_created'))
    monkeypatch.setenv('FLIMKIT_PLUGIN_PATH', '')
    monkeypatch.delenv('FLIMKIT_NO_PLUGINS', raising=False)
    monkeypatch.setattr(config_manager.cfg, '_global', {})
    monkeypatch.setattr(config_manager.cfg, '_project', {})
    monkeypatch.setattr(config_manager.cfg, 'save', lambda: None)
    monkeypatch.setattr(loader, 'entry_points', lambda: [])
    loader.reset()
    yield
    loader.reset()
