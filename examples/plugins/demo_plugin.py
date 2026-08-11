import numpy as np

from flimkit.plugins import file_format, format_sniffer, phasor_filter, plugin_config, tool

FLIMKIT_PLUGIN_API = 1


@tool(id='demo_registry', label='Add-on Self Test...', menu='Tools', order=800)
def open_registry_window(app):
    import tkinter as tk
    from tkinter import ttk
    from tkinter.scrolledtext import ScrolledText
    from flimkit import plugins
    win = tk.Toplevel(app.root)
    win.title('Add-on Self Test')
    win.geometry('640x460')
    text = ScrolledText(win, wrap=tk.WORD)
    text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    lines = [f'This window was opened by {plugins.short_name(__file__)}.py, not by FLIMKit.', '']
    lines.append(f'FLIMKit handed it the live {type(app).__name__}, whose window is titled '
                 f'{app.root.title()!r}, and this Toplevel is parented to it.')
    lines.append('')
    lines.append(f'API version {plugins.API_VERSION}')
    lines.append('')
    lines.append('Tools registered:')
    for t in plugins.tools():
        lines.append(f'  {"/".join(t.menu_path):24s} {t.label:32s} {t.source}')
    lines.append('')
    lines.append('Formats registered by add-ons:')
    for f in plugins.formats():
        lines.append(f'  {f.id:16s} {str(f.exts):24s} {f.modality:10s} {f.source}')
    lines.append('')
    lines.append('Phasor filters registered by add-ons:')
    for pf in plugins.phasor_filters():
        lines.append(f'  {pf.id:16s} {pf.label:28s} {pf.source}')
    lines.append('')
    lines.append('Load report:')
    for result in plugins.load_report():
        state = 'ok    ' if result.ok else 'FAILED'
        lines.append(f'  {state} {result.n_registered} registration(s)  {result.source}')
    for name in plugins.skipped():
        lines.append(f'  skipped                    {name}')
    lines.append('')
    cfg = plugin_config('demo_plugin')
    opened = int(cfg.get('times_opened', 0) or 0) + 1
    cfg.set('times_opened', opened)
    cfg.save()
    lines.append(f'This add-on has its own config section and has been opened {opened} time(s).')
    lines.append('That count lives in plugin:demo_plugin in ~/.flimkit/config.json.')
    text.insert(tk.END, '\n'.join(lines))
    text.config(state=tk.DISABLED)
    row = ttk.Frame(win)
    row.pack(fill=tk.X, padx=4, pady=(0, 6))
    ttk.Button(row, text='Raise on purpose', command=lambda: 1 / 0).pack(side=tk.LEFT)
    ttk.Button(row, text='Close', command=win.destroy).pack(side=tk.RIGHT)


@tool(id='demo_nested', label='Nested Entry...', menu='Tools/Add-on Demo', order=810)
def open_nested(app):
    from tkinter import messagebox
    messagebox.showinfo(
        'Nested Entry',
        'Menu nesting works: this entry was registered with '
        "menu='Tools/Add-on Demo' and FLIMKit built the submenu for it.")


@tool(id='demo_broken', label='Break On Purpose...', menu='Tools/Add-on Demo', order=820)
def open_broken(app):
    raise RuntimeError('this add-on failed on purpose, FLIMKit should still be running')


@file_format(id='demo_format', label='Add-on Demo Format', exts=('.demoflim',),
             modality='time')
class DemoReader:

    def __init__(self, path, **kwargs):
        self.path = path
        self.n_bins = 256
        self.tcspc_res = 25e-12
        self.decay = np.zeros(self.n_bins, dtype=np.float64)

    def summed_decay(self, channel=None):
        return self.decay


@format_sniffer(tier='magic', order=200)
def sniff_demo(path):
    try:
        with open(path, 'rb') as fh:
            if fh.read(8) == b'DEMOFLIM':
                return 'demo_format'
    except OSError:
        pass
    return None


@phasor_filter(id='demo_passthrough', label='Add-on Demo (no-op)')
def demo_passthrough(real, imag):
    return real, imag
