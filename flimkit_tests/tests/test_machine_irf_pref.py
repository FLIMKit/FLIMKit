import json

import numpy as np

from flimkit.configs import preferred_machine_irf_path

def _write_config(home, value):
    cfg_dir = home / '.flimkit'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / 'config.json').write_text(json.dumps({'preferences': {'machine_irf_path': value}}))

def test_no_config_means_no_preference(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    assert preferred_machine_irf_path() is None

def test_saved_preference_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    irf = tmp_path / 'scope_a.npy'
    np.save(irf, np.linspace(0, 1, 32))
    _write_config(tmp_path, str(irf))
    assert preferred_machine_irf_path() == irf

def test_missing_preferred_file_falls_back(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('HOME', str(tmp_path))
    _write_config(tmp_path, str(tmp_path / 'gone.npy'))
    assert preferred_machine_irf_path() is None
    assert 'missing' in capsys.readouterr().out
