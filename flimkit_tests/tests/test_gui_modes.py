import pytest
import tkinter as tk

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
