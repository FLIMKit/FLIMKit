import pytest
import tkinter as tk
from pathlib import Path

from flimkit.UI.gui import _UIBuilder

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
