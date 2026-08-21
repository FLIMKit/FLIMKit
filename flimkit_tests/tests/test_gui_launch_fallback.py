import pytest
import tkinter as tk
from tkinter import ttk

from flimkit.UI import gui


@pytest.fixture
def root():
    try:
        found = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    found.withdraw()
    yield found
    try:
        found.destroy()
    except tk.TclError:
        pass


def test_discard_default_root_clears_a_live_root(root):
    assert tk._default_root is not None
    gui.discard_default_root()
    assert tk._default_root is None


def test_the_fallback_theme_is_one_the_interpreter_knows(root):
    chosen = gui.apply_fallback_theme(root)
    assert chosen
    if 'sv_ttk' not in chosen:
        assert chosen in ttk.Style(root).theme_names()


def test_a_theme_the_tk_cannot_load_falls_back_rather_than_raising(monkeypatch):
    if not gui.HAS_TKMT:
        pytest.skip('TKinterModernThemes is not installed')
    try:
        tk.Tk().destroy()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    gui.discard_default_root()
    seen = {}

    def refuse(*args, **kwargs):
        tk.Tk()
        raise tk.TclError('invalid command name "set_theme"')

    class Stub:
        def __init__(self, built):
            seen['root'] = built
            seen['theme'] = ttk.Style(built).theme_use()

    monkeypatch.setattr(gui.FLIMKitGUIThemed, '__init__', refuse)
    monkeypatch.setattr(gui, 'FLIMKitGUIFallback', Stub)
    monkeypatch.setattr(tk.Misc, 'mainloop', lambda self, n=0: seen.setdefault('mainloop', True))
    monkeypatch.setattr(gui, 'init_crash_handler', lambda: None, raising=False)
    gui.launch_gui()

    assert seen['mainloop'] is True
    assert seen['theme']
    seen['root'].destroy()
    gui.discard_default_root()


def test_a_root_is_built_even_when_tkdnd_will_not_load(monkeypatch):
    try:
        tk.Tk().destroy()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    gui.discard_default_root()
    monkeypatch.setattr(gui, 'HAS_DND', True)
    import tkinterdnd2

    def refuse(*args, **kwargs):
        raise RuntimeError('Unable to load tkdnd library.')

    monkeypatch.setattr(tkinterdnd2.TkinterDnD.Tk, '__init__', refuse)
    built = gui.plain_root()
    assert isinstance(built, tk.Tk)
    built.destroy()
    gui.discard_default_root()
