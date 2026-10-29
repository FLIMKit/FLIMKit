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

    def fake_fit(args):
        seen.append(args.ptu_basename)
        if args.ptu_basename == 'b':
            raise RuntimeError('boom')
        return {'global_summary': {'tau_mean_amp_global_ns': 1.5}}

    monkeypatch.setattr(sys, 'argv', _argv_tile(tmp_path, tmp_path, tmp_path / 'out'))
    with mock.patch('flimkit.interactive._run_tile_fit', fake_fit):
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
    with mock.patch('flimkit.interactive._run_flim_fit', lambda a: {'global_summary': {'tau1': 2.0}, 'pileup_pct': 1.0}):
        csv_path = batch_cli.fovBatchFit()
    rows = list(csv.DictReader(open(csv_path)))
    assert [r['fov'] for r in rows] == ['one', 'two']
    assert Path(csv_path).parent == tmp_path / 'save'
