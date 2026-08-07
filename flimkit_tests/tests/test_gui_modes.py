import pytest
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock, patch

from flimkit.UI.gui import _UIBuilder
from flimkit.UI.irf_widget import IRFWidget

MODES = ['fov', 'stitch', 'phasor', 'batch', 'irf']


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    root.withdraw()
    b = _UIBuilder.__new__(_UIBuilder)
    b.root = root
    b._init_ui()
    yield b
    root.destroy()


def test_init_builds_all_form_frames(app):
    for mode in MODES:
        assert mode in app._form_inner_frames


def test_switch_through_every_mode(app):
    # Switching to each mode in turn must not raise and must update the
    # current-form marker. Guards the _switch_form machinery before Phase 3.
    for mode in MODES:
        app._switch_form(mode)
        assert app._current_form == mode


def test_switch_round_trips(app):
    # Bounce between notebook modes (fov/stitch) and traditional modes to catch
    # show/hide state that leaks between switches.
    for mode in ['fov', 'phasor', 'stitch', 'batch', 'fov', 'irf', 'stitch']:
        app._switch_form(mode)
        assert app._current_form == mode


def test_phasor_autopopulates_ptu_from_fov(app):
    app.sv_ptu.set('/data/sample.ptu')
    app.sv_ph_ptu.set('')
    app._switch_form('phasor')
    assert app.sv_ph_ptu.get() == '/data/sample.ptu'


def test_mode_classes_importable():
    from flimkit.UI.modes.fov_mode import FovMode
    from flimkit.UI.modes.stitch_mode import StitchMode
    from flimkit.UI.modes.batch_mode import BatchMode
    from flimkit.UI.modes.irf_mode import IrfMode
    from flimkit.UI.modes.phasor_mode import PhasorMode
    from flimkit.UI.modes.base import BaseMode
    for cls in (FovMode, StitchMode, BatchMode, IrfMode, PhasorMode):
        assert issubclass(cls, BaseMode)


def test_each_mode_builds_its_run_button(app):
    # Each per-mode build() runs during _init_ui and creates a distinct run
    # button on the builder. Guards the Phase 3b extraction.
    for attr in ['_btn_fov', '_btn_st', '_btn_batch', '_btn_mirf', '_btn_ph']:
        assert hasattr(app, attr)


def test_stitch_mode_builds_fit_and_tab(app):
    # StitchMode.build() also wires the inner fit frame via build_fit().
    assert hasattr(app, '_fit_frame')
    assert hasattr(app, '_btn_st')


def test_controller_created(app):
    from flimkit.UI.controller import FLIMKitController
    assert isinstance(app._controller, FLIMKitController)


def test_fov_args_maps_state(app):
    # FLIMKitController.fov_args() harvests AppState into the fit args namespace.
    app.sv_ptu.set('/data/sample.ptu')
    app.sv_xlsx.set('/data/sample.xlsx')
    app.iv_nexp_fov.set(3)
    app.sv_mode_fov.set('summed')
    app.sv_tau_min_fov.set('0.2')
    app.sv_tau_max_fov.set('9.5')
    app.sv_out_fov.set('myout')
    app.bv_cell.set(True)
    app.bv_correct_pileup.set(True)
    a = app._controller.fov_args()
    assert a.ptu == '/data/sample.ptu'
    assert a.xlsx == '/data/sample.xlsx'
    assert a.nexp == 3
    assert a.mode == 'summed'
    assert a.tau_min == 0.2
    assert a.tau_max == 9.5
    assert a.cell_mask == True
    assert a.correct_pileup == True
    # bare output name is rebased next to the PTU file
    assert a.out == str(Path('/data/sample.ptu').parent / 'myout')


def test_stitch_args_tile_fit_pipeline(app):
    app.sv_xlif.set('/data/scan one.xlif')
    app.sv_ptu_dir.set('/data/tiles')
    app.sv_out_st.set('/data/out')
    app.iv_nexp_st.set(2)
    app.bv_rotate.set(False)
    app.sv_pipeline.set('tile_fit')
    a = app._controller.stitch_args()
    assert a.xlif == '/data/scan one.xlif'
    assert a.ptu_dir == '/data/tiles'
    assert a.ptu_basename == 'scan one'
    assert a.rotate_tiles == False
    assert a.nexp == 2
    assert a.mode == 'both'
    assert a.no_plots == True
    # roi_name spaces are normalised for the output subfolder
    assert a.output_dir == str(Path('/data/out') / 'scan_one')


def test_stitch_args_stitch_only_pipeline(app):
    app.sv_xlif.set('/d/s.xlif')
    app.sv_ptu_dir.set('/d/t')
    app.sv_out_st.set('/d/o')
    app.sv_pipeline.set('stitch_only')
    app.bv_perpix.set(True)
    a = app._controller.stitch_args()
    assert a.mode == 'both'
    assert a.no_plots == False
    assert a.irf_xlsx_dir is None


def test_irf_widget_browse_accepts_csv_exports():
    widget = IRFWidget.__new__(IRFWidget)
    widget.sv_method = MagicMock()
    widget.sv_method.get.return_value = 'irf_xlsx'
    widget.sv_path = MagicMock()

    with patch('flimkit.UI.irf_widget._browse_file') as browse:
        widget._browse_irf_path()

    filetypes = browse.call_args.args[2]
    assert any('*.csv' in pattern for _, pattern in filetypes)


def test_series_fit_pipeline_toggles_series_controls(app):
    app._switch_form('stitch')
    app.sv_pipeline.set('series_fit')
    app._pipeline_changed()
    assert app._series_frame.winfo_manager()
    assert app._btn_st.cget('text') == '▶  Run Series Fit'

def test_series_controls_hidden_for_other_pipelines(app):
    app._switch_form('stitch')
    for mode in ('series_fit', 'stitch_only', 'tile_fit', 'stitch_fit'):
        app.sv_pipeline.set(mode)
        app._pipeline_changed()
    assert not app._series_frame.winfo_manager()

def test_series_fit_args_derive_roi_from_ptu_dir(app, tmp_path):
    app._switch_form('stitch')
    app.sv_pipeline.set('series_fit')
    app.sv_xlif.set('')
    app.sv_ptu_dir.set(str(tmp_path / 'my series'))
    app.sv_out_st.set(str(tmp_path / 'out'))
    a = app._controller.stitch_args()
    assert a.ptu_basename is None
    assert Path(a.output_dir).name == 'my_series'

def test_non_series_args_still_derive_roi_from_xlif(app, tmp_path):
    app._switch_form('stitch')
    app.sv_pipeline.set('tile_fit')
    app.sv_xlif.set(str(tmp_path / 'R 2.xlif'))
    app.sv_ptu_dir.set(str(tmp_path))
    app.sv_out_st.set(str(tmp_path / 'out'))
    a = app._controller.stitch_args()
    assert a.ptu_basename == 'R 2'
    assert Path(a.output_dir).name == 'R_2'


def _series_planes():
    return [{'t': 1, 'z': 1, 'name': 'R_t1_z1', 'dir': 'R_t1_z1'},
            {'t': 1, 'z': 2, 'name': 'R_t1_z2', 'dir': 'R_t1_z2'},
            {'t': 2, 'z': 1, 'name': 'R_t2_z1', 'dir': 'R_t2_z1'}]

def test_display_series_shows_slider_over_all_planes(app, tmp_path):
    fov = app._fov_preview
    fov.display_series(_series_planes(), tmp_path)
    assert fov._zbar.winfo_manager()
    assert float(fov._z_slider.cget('to')) == 2.0
    assert fov._z_label.get().startswith('t1 z1')
    assert '(1/3)' in fov._z_label.get()

def test_series_slider_moves_between_planes(app, tmp_path):
    fov = app._fov_preview
    fov.display_series(_series_planes(), tmp_path)
    fov._on_z_slider('2')
    assert fov._z_i == 2
    assert fov._z_label.get().startswith('t2 z1')
    assert fov._series is not None

def test_series_slider_clamps_out_of_range(app, tmp_path):
    fov = app._fov_preview
    fov.display_series(_series_planes(), tmp_path)
    fov._on_z_slider('99')
    assert fov._z_i == 2

def test_display_series_with_no_planes_hides_bar(app, tmp_path):
    fov = app._fov_preview
    fov.display_series([], tmp_path)
    assert not fov._zbar.winfo_manager()
    assert fov._series is None

def test_loading_a_single_fov_clears_the_series_bar(app, tmp_path):
    fov = app._fov_preview
    fov.display_series(_series_planes(), tmp_path)
    fov._hide_zstack()
    assert fov._series is None
    assert not fov._zbar.winfo_manager()
