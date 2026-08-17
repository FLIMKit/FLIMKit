import pytest

from flimkit import plugins
from flimkit.plugins import registry

tk = pytest.importorskip('tkinter')


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear()
    yield
    registry.clear()


def test_panel_renders_a_plugin_button(root):
    from flimkit.UI.roi_tools import RoiAnalysisPanel

    clicked = []

    @plugins.panel_button('render_demo', 'Send to Demo', panel='roi')
    def send(app):
        clicked.append(app)

    panel = RoiAnalysisPanel(root)
    panel.app = object()

    labels = []
    for child in panel.frame.winfo_children():
        for widget in child.winfo_children():
            text = widget.cget('text') if 'text' in widget.keys() else ''
            if text:
                labels.append(text)

    assert 'Send to Demo' in labels

    panel._run_plugin_button(plugins.get_panel_button('render_demo'))
    assert clicked == [panel.app]


def test_panel_builds_with_no_plugin_buttons(root):
    from flimkit.UI.roi_tools import RoiAnalysisPanel

    panel = RoiAnalysisPanel(root)

    assert panel.frame is not None
