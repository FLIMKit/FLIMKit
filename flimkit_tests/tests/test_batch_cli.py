import csv
import sys
from pathlib import Path
from unittest import mock

import pytest

from flimkit import batch_cli


def _argv_tile(xlif_dir, ptu_dir, out):
    return ['x', '--xlif-dir', str(xlif_dir), '--ptu-dir', str(ptu_dir), '--output-dir', str(out)]


def test_tile_batch_writes_row_per_roi_and_survives_error(tmp_path, monkeypatch):
    for n in ('a', 'b', 'c'):
        (tmp_path / (n + '.xlif')).write_text('')
    seen = []

    def fake_fit(xlif, ptu_dir, out, p):
        seen.append(xlif.stem)
        if xlif.stem == 'b':
            raise RuntimeError('boom')
        return {'roi': xlif.stem, 'status': 'OK', 'tau_mean_amp_global_ns': 1.5}

    monkeypatch.setattr(sys, 'argv', _argv_tile(tmp_path, tmp_path, tmp_path / 'out'))
    with mock.patch('flimkit.batch_cli.fitRoi', fake_fit):
        csv_path = batch_cli.tileBatchFit()
    rows = list(csv.DictReader(open(csv_path)))
    assert seen == ['a', 'b', 'c']
    assert [r['status'][:2] for r in rows] == ['OK', 'ER', 'OK']


def test_fov_batch_no_files_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['x', '--fov-dir', str(tmp_path)])
    with pytest.raises(ValueError):
        batch_cli.fovBatchFit()


def test_fov_batch_runs_each_file(tmp_path, monkeypatch):
    (tmp_path / 'one.ptu').write_text('')
    (tmp_path / 'two.ptu').write_text('')
    monkeypatch.setattr(sys, 'argv', ['x', '--fov-dir', str(tmp_path)])
    with mock.patch('flimkit.batch_cli.fitFov', lambda f, o, p: {'fov': f.stem, 'status': 'OK', 'tau1': 2.0}):
        csv_path = batch_cli.fovBatchFit()
    rows = list(csv.DictReader(open(csv_path)))
    assert [r['fov'] for r in rows] == ['one', 'two']
    assert Path(csv_path).parent == tmp_path / 'save'


def test_params_defaults_match_gui(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', _argv_tile(tmp_path, tmp_path, tmp_path))
    seen = {}

    def grab(xlif, ptu_dir, out, p):
        seen['p'] = p
        return {'roi': xlif.stem, 'status': 'OK'}

    (tmp_path / 'a.xlif').write_text('')
    with mock.patch('flimkit.batch_cli.fitRoi', grab):
        batch_cli.tileBatchFit()
    p = seen['p']
    assert (p.gamma, p.tau_lo, p.tau_hi, p.save_lifetime, p.save_rgb) == (0.4, 0.0, 10.0, True, True)
