import pytest

tk = pytest.importorskip('tkinter')
from flimkit.UI.irf_widget import IRFWidget

@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip('no display')
    window.withdraw()
    yield window
    window.destroy()

def test_the_selector_is_one_dropdown_with_every_choice(root):
    widget = IRFWidget(root, default='irf_xlsx', xlsx_var=tk.StringVar(value='a.xlsx'))
    assert str(widget._combo.cget('state')) == 'readonly'
    assert list(root.tk.splitlist(widget._combo.cget('values'))) == [lbl for lbl, _ in IRFWidget.CHOICES]

def test_picking_a_label_sets_the_method_key(root):
    widget = IRFWidget(root, default='irf_xlsx', xlsx_var=tk.StringVar(value='a.xlsx'))
    widget._combo.set('Measured IRF file (scatter PTU or .pck)')
    widget._combo.event_generate('<<ComboboxSelected>>')
    root.update()
    assert widget.sv_method.get() == 'file'
    assert widget._path_e.winfo_manager() != ''

def test_setting_the_method_from_code_updates_label_and_rows(root):
    widget = IRFWidget(root, default='file')
    widget.sv_method.set('raw')
    root.update()
    assert widget.sv_label.get() == 'Estimate from decay - raw'
    assert widget._path_e.winfo_manager() == ''
    assert widget.get_args()['estimate_irf'] == 'raw'
