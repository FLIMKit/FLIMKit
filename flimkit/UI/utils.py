from __future__ import annotations
import re
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path
from typing import Optional
import numpy as np
from flimkit.UI.fit_help import help_button
from flimkit.utils.config_snapshot import _C
from flimkit.utils.session import (
    _reconstruct_dict_from_session,
    _safe_array_from_json,
    _parse_summary,
)
try:
    from tkinterdnd2 import DND_FILES, DND_TEXT
    HAS_DND = True
except ImportError:
    HAS_DND = False

def _enable_dnd(root):
    if not HAS_DND:
        return False
    try:
        from tkinterdnd2 import TkinterDnD
        TkinterDnD._require(root)
        return True
    except Exception:
        return False

class _Redirect:

    def __init__(self, widget: scrolledtext.ScrolledText, buf: list, root=None, is_stderr=False):
        self.widget = widget
        self.buf = buf
        self.root = root
        self._is_stderr = is_stderr
        self._batch = []
        self._batch_size = 5000
        self._last_flush = time.time()
        self._flush_interval = 0.5

    def write(self, text: str):
        if not text:
            return
        self.buf.append(text)
        self._batch.append(text)
        if self._is_stderr:
            try:
                from flimkit.utils.crash_handler import log_event
                log_event(f'STDERR: {text.rstrip()}', level='warning')
            except Exception:
                pass
        should_flush = False
        if len(''.join(self._batch)) >= self._batch_size:
            should_flush = True
        elif time.time() - self._last_flush >= self._flush_interval:
            should_flush = True
        if should_flush:
            self._flush_batch()

    def _flush_batch(self):
        if not self._batch:
            return
        text = ''.join(self._batch)
        self._batch.clear()
        if self.root:
            self.root.after(0, self._update_widget, text)
        else:
            self._update_widget(text)

    def _update_widget(self, text):
        try:
            self.widget.configure(state='normal')
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)
            self.widget.configure(state='disabled')
            self.widget.update_idletasks()
        except Exception:
            pass
        self._last_flush = time.time()

    def flush(self):
        self._flush_batch()

class _FileRedirect:

    def __init__(self, filepath: str, buf: list):
        self.filepath = filepath
        self.buf = buf
        self._file = None
        try:
            self._file = open(filepath, 'w', buffering=1)
        except Exception:
            pass

    def write(self, text: str):
        if not text:
            return
        self.buf.append(text)
        if self._file:
            try:
                self._file.write(text)
            except Exception:
                pass

    def flush(self):
        if self._file:
            try:
                self._file.flush()
            except Exception:
                pass

    def close(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

class _FileTailer:

    def __init__(self, filepath: str, widget: scrolledtext.ScrolledText, update_interval_ms: int = 200):
        self.filepath = filepath
        self.widget = widget
        self.update_interval_ms = update_interval_ms
        self._file = None
        self._last_pos = 0
        self._running = False

    def start(self, root):
        self._running = True
        self._poll_file(root)

    def _poll_file(self, root):
        if not self._running:
            return
        try:
            if not Path(self.filepath).exists():
                root.after(self.update_interval_ms, lambda: self._poll_file(root))
                return
            with open(self.filepath, 'r') as f:
                f.seek(self._last_pos)
                new_content = f.read()
                self._last_pos = f.tell()
            if new_content:
                self.widget.configure(state='normal')
                self.widget.insert(tk.END, new_content)
                self.widget.see(tk.END)
                self.widget.configure(state='disabled')
                self.widget.update_idletasks()
        except Exception:
            pass
        root.after(self.update_interval_ms, lambda: self._poll_file(root))

    def stop(self):
        self._running = False

PAD = dict(padx=8, pady=4)

from flimkit.formats import file_dialog_filetypes
FLIM_FILETYPES = file_dialog_filetypes() + [('All', '*.*')]

def _browse_file(var, title='Select file', filetypes=(('All', '*.*'),)):
    p = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if p:
        var.set(p)

def _browse_dir(var, title='Select directory'):
    p = filedialog.askdirectory(title=title)
    if p:
        var.set(p)

def _row(parent, label, var, row, browse_fn, width=45, state='normal'):
    ttk.Label(parent, text=label).grid(
        row=row, column=0, sticky='e', padx=6, pady=3)
    e = ttk.Entry(parent, textvariable=var, width=width, state=state)
    e.grid(row=row, column=1, sticky='ew', padx=4, pady=3)
    ttk.Button(parent, text='Browse...', command=browse_fn).grid(
        row=row, column=2, padx=4, pady=3)
    if HAS_DND:
        try:
            def _drop_handler(event):
                data = event.data.strip()
                if data.startswith('{') and data.endswith('}'):
                    data = data[1:-1]
                var.set(data)
            e.drop_target_register(DND_FILES, DND_TEXT)
            e.dnd_bind('<<Drop>>', _drop_handler)
        except Exception:
            pass
    return e

def _section(parent, text: str, help_topic: Optional[str] = None) -> ttk.LabelFrame:
    lf = ttk.LabelFrame(parent, text=f'  {text}  ', padding=(10, 6))
    if help_topic:
        holder = ttk.Frame(lf)
        ttk.Label(holder, text=f'  {text} ').pack(side='left')
        help_button(holder, help_topic).pack(side='left', padx=(0, 6))
        lf.configure(labelwidget=holder)
    return lf

def _tog(bvar: tk.BooleanVar, entry: ttk.Entry):
    entry.configure(state='normal' if bvar.get() else 'disabled')

def _flt(sv: tk.StringVar) -> Optional[float]:
    v = sv.get().strip()
    return float(v) if v and v.lower() != 'none' else None

def _thresh(bvar: tk.BooleanVar, sv: tk.StringVar):
    if not bvar.get():
        return None
    v = sv.get().strip()
    return int(v) if v else None
