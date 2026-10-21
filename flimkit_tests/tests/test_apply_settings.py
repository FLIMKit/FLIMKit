import argparse
import json

import numpy as np
import pytest

from flimkit.project import ScanRecord
from flimkit.utils.apply_settings import fov_targets, read_session_settings, settings_summary, target_args


def source_args(folder, out=None):
    return argparse.Namespace(
        ptu=str(folder / 'a.ptu'),
        xlsx=str(folder / 'a.xlsx'),
        irf_xlsx=str(folder / 'a.xlsx'),
        out=str(out or folder / 'a'),
        nexp=2,
        tau_min=0.1,
        tau_max=10.0,
        cell_mask=True,
    )


def test_target_takes_its_own_file_export_and_output(tmp_path):
    a = target_args(source_args(tmp_path), tmp_path / 'b.ptu', str(tmp_path / 'b.xlsx'))
    assert a.ptu == str(tmp_path / 'b.ptu')
    assert a.xlsx == str(tmp_path / 'b.xlsx')
    assert a.irf_xlsx == str(tmp_path / 'b.xlsx')
    assert a.out == str(tmp_path / 'b')


def test_every_fit_setting_is_carried_and_the_source_is_untouched(tmp_path):
    src = source_args(tmp_path)
    a = target_args(src, tmp_path / 'b.ptu', str(tmp_path / 'b.xlsx'))
    assert (a.nexp, a.tau_min, a.tau_max, a.cell_mask) == (2, 0.1, 10.0, True)
    assert src.ptu == str(tmp_path / 'a.ptu')
    assert src.out == str(tmp_path / 'a')


def test_an_xlsx_irf_needs_the_targets_own_export(tmp_path):
    with pytest.raises(ValueError):
        target_args(source_args(tmp_path), tmp_path / 'b.ptu', None, 'irf_xlsx')


def test_other_irf_methods_do_not_need_an_export(tmp_path):
    src = source_args(tmp_path)
    src.irf_xlsx = None
    a = target_args(src, tmp_path / 'b.ptu', None, 'machine_irf')
    assert a.xlsx is None
    assert a.irf_xlsx is None


def test_a_custom_output_folder_is_kept(tmp_path):
    src = source_args(tmp_path, out=tmp_path / 'results' / 'a')
    a = target_args(src, tmp_path / 'b.ptu', str(tmp_path / 'b.xlsx'))
    assert a.out == str(tmp_path / 'results' / 'b')


def test_session_settings_are_read_back(tmp_path):
    path = tmp_path / 'a.roi_session.npz'
    form = {'fit_model_fov': 'discrete', 'nexp_fov': 2}
    scale = {'vmin': 1.5, 'vmax': 4.0, 'gamma': 0.8, 'cmap': 'hsv'}
    np.savez_compressed(path, form_state_json=json.dumps(form), fov_color_scale=json.dumps(scale))
    assert read_session_settings(path) == (form, scale)


def test_a_missing_session_gives_nothing(tmp_path):
    assert read_session_settings(tmp_path / 'none.npz') == (None, None)


class FakeProject:

    def __init__(self, scans):
        self.scans = scans

    def sorted_scans(self):
        yield from sorted(self.scans.items())


def test_targets_are_the_other_single_fov_files(tmp_path):
    for name in ('a.ptu', 'b.ptu', 'c.ptu'):
        (tmp_path / name).write_text('x')
    project = FakeProject({
        'a': ScanRecord('a', 'fov', str(tmp_path / 'a.ptu')),
        'b': ScanRecord('b', 'fov', str(tmp_path / 'b.ptu')),
        'c': ScanRecord('c', 'fov', str(tmp_path / 'c.ptu')),
        'tiles': ScanRecord('tiles', 'xlif', str(tmp_path / 'tiles.xlif')),
        'stack': ScanRecord('stack', 'zstack', str(tmp_path)),
    })
    assert [stem for stem, _ in fov_targets(project, tmp_path / 'a.ptu')] == ['b', 'c']


def test_the_summary_names_the_model_irf_and_display():
    text = settings_summary({'fit_model_fov': 'discrete', 'nexp_fov': 2, 'irf_method': 'machine_irf',
                             'tau_min_fov': '0.1', 'tau_max_fov': '10'},
                            {'vmin': 1.5, 'vmax': None, 'gamma': 0.8, 'cmap': 'hsv'},
                            {'vmin': None, 'vmax': 800, 'cmap': 'gray'})
    assert 'Intensity display: 0 to 800, gray' in text
    assert 'discrete, 2-exp' in text
    assert 'machine_irf' in text
    assert '1.50 to auto ns' in text
    assert 'hsv' in text


def test_only_fitted_single_fov_files_are_exported(tmp_path):
    from flimkit.utils.apply_settings import fitted_fovs
    for stem in ('a', 'c'):
        np.savez(tmp_path / f'{stem}.roi_session.npz', x=np.zeros(1))
    project = FakeProject({
        'a': ScanRecord('a', 'fov', str(tmp_path / 'a.ptu')),
        'b': ScanRecord('b', 'fov', str(tmp_path / 'b.ptu')),
        'c': ScanRecord('c', 'fov', str(tmp_path / 'c.ptu')),
        'tiles': ScanRecord('tiles', 'xlif', str(tmp_path / 'tiles.xlif')),
    })
    assert [stem for stem, _ in fitted_fovs(project)] == ['a', 'c']
    assert fitted_fovs(None) == []


def test_the_fit_summary_is_the_table_the_gui_shows(tmp_path):
    from flimkit.utils.apply_settings import write_fit_summary
    np.savez(tmp_path / 's.npz',
             summary_params=np.array(['τ1', 'α1', 'χ²_r(tail) Pearson'], dtype=object),
             summary_values=np.array(['5.4335', '1.080e+04', '2.2614'], dtype=object),
             summary_units=np.array(['ns', '', ''], dtype=object),
             strategy='machine_irf',
             decay=np.full(10, 5.0))
    session = dict(np.load(tmp_path / 's.npz', allow_pickle=True))
    out = tmp_path / 's_fit_summary.txt'
    assert write_fit_summary(session, out, name='s.ptu') == True
    lines = out.read_text(encoding='utf-8').splitlines()
    assert 'File: s.ptu' in lines
    assert 'IRF: machine_irf' in lines
    assert 'Total photons: 50' in lines
    assert 'τ1                  5.4335 ns' in lines
    assert 'χ²_r(tail) Pearson  2.2614' in lines


def test_a_session_without_a_summary_writes_nothing(tmp_path):
    from flimkit.utils.apply_settings import write_fit_summary
    out = tmp_path / 'x.txt'
    assert write_fit_summary({'decay': np.ones(3)}, out) == False
    assert not out.exists()
