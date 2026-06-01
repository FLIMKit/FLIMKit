import pytest
import tkinter as tk

from flimkit.UI.app_state import AppState
from flimkit.UI.gui import _UIBuilder


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def builder():
    b = _UIBuilder.__new__(_UIBuilder)
    b.state = AppState()
    return b


def test_appstate_values_snapshots_tk_vars(root):
    st = AppState()
    st.sv_name = tk.StringVar(value='abc')
    st.bv_flag = tk.BooleanVar(value=True)
    st.iv_count = tk.IntVar(value=3)
    snap = st.values()
    assert snap == {'sv_name': 'abc', 'bv_flag': True, 'iv_count': 3}


def test_appstate_values_skips_non_var_attributes(root):
    st = AppState()
    st.sv_name = tk.StringVar(value='x')
    st.not_a_var = 'plain string'
    snap = st.values()
    assert snap == {'sv_name': 'x'}


def test_delegation_returns_var_from_state(builder, root):
    builder.state.sv_ptu = tk.StringVar(value='hi')
    assert builder.sv_ptu.get() == 'hi'


def test_delegation_write_reaches_state(builder, root):
    builder.state.sv_ptu = tk.StringVar(value='hi')
    builder.sv_ptu.set('bye')
    assert builder.state.sv_ptu.get() == 'bye'


def test_hasattr_false_when_state_empty(builder):
    # Session save/restore guards rely on hasattr(self, 'sv_x') being False
    # until the variable is actually built.
    assert hasattr(builder, 'sv_ptu') == False


def test_hasattr_true_after_var_created(builder, root):
    builder.state.sv_ptu = tk.StringVar()
    assert hasattr(builder, 'sv_ptu') == True


def test_missing_attribute_raises(builder):
    with pytest.raises(AttributeError):
        builder.totally_missing_attribute


def test_no_state_yet_does_not_crash():
    b = _UIBuilder.__new__(_UIBuilder)
    assert hasattr(b, 'sv_ptu') == False
    with pytest.raises(AttributeError):
        b.sv_ptu


def test_full_ui_routes_all_vars_through_state(root):
    b = _UIBuilder.__new__(_UIBuilder)
    b.root = root
    b._init_ui()
    # Every tk variable created while building the UI lives on AppState, and a
    # representative sample from each mode group resolves through delegation.
    sample = ['sv_ptu', 'sv_out_fov', 'iv_nexp_fov', 'bv_cell', 'sv_tau_min_fov',
              'sv_xlif', 'sv_pipeline', 'bv_register', 'sv_batch_mode',
              'iv_nexp_batch', 'sv_mirf_src', 'sv_ph_ptu', 'sv_ph_fret_taud',
              'current_mode', 'mode_status']
    for nam in sample:
        assert hasattr(b, nam) == True
        assert nam in b.state.__dict__
    assert b.current_mode.get() == 'fov'
    assert len(b.state.values()) > 0
