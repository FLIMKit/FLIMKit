import pytest
import tkinter as tk

from flimkit.UI.fit_help import FIT_HELP, TOPIC_ORDER, FitHelpWindow, help_button
from flimkit.UI.utils import _section


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    r.withdraw()
    yield r
    r.destroy()


def test_every_topic_has_content():
    assert set(TOPIC_ORDER) == set(FIT_HELP)
    for key in TOPIC_ORDER:
        title, entries, note = FIT_HELP[key]
        assert title
        assert len(entries) >= 2
        assert note
        for name, body in entries:
            assert name and body


def test_no_em_or_en_dashes_in_copy():
    for title, entries, note in FIT_HELP.values():
        blob = title + note + ''.join(n + b for n, b in entries)
        assert '—' not in blob
        assert '–' not in blob


def test_window_opens_at_each_topic(root):
    for key in TOPIC_ORDER:
        win = FitHelpWindow(root, topic=key)
        root.update_idletasks()
        assert win._text.get('1.0', 'end').strip()
        assert key in win._marks
        win.destroy()


def test_window_contains_every_topic_regardless_of_entry_point(root):
    win = FitHelpWindow(root, topic='optimizer')
    root.update_idletasks()
    body = win._text.get('1.0', 'end')
    for key in TOPIC_ORDER:
        assert FIT_HELP[key][0] in body
    win.destroy()


def test_help_button_binds_without_opening(root):
    btn = help_button(root, 'fit_model')
    assert btn.bind('<Button-1>')
    assert not [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]


def test_section_with_help_topic_uses_labelwidget(root):
    plain = _section(root, 'Fitting Parameters')
    assert not plain.cget('labelwidget')
    with_help = _section(root, 'Fitting Parameters', help_topic='fit_model')
    assert with_help.cget('labelwidget')
