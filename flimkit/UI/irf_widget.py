from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Optional
from flimkit.UI.utils import _browse_file, _section


class IRFWidget:
    # Sentinel to detect that the path was auto-filled (not user-entered)
    _AUTO_FILL = object()

    CHOICES = [
        ('Leica analytical model (XLSX)',                'irf_xlsx'),
        ('Machine IRF (.npy pre-built)',                 'machine_irf'),
        ('Machine IRF + full σ broadening',               'machine_irf_sigma_full'),
        ('Machine IRF + half σ broadening (σ≤0.5)',       'machine_irf_sigma_half'),
        ('Scatter PTU (measured IRF)',                   'file'),
        ('Estimate from decay - raw',                    'raw'),
        ('Estimate from decay - parametric',             'parametric'),
        ('Gaussian (fallback)',                          'gaussian'),
    ]

    def __init__(self, parent, default='irf_xlsx', xlsx_var=None, machine_irf_default: str = ''):
        self.xlsx_var  = xlsx_var
        self._machine_irf_default = machine_irf_default
        self.sv_method = tk.StringVar(value=default)
        self.sv_path   = tk.StringVar()

        self.frame = _section(parent, 'Instrument Response Function (IRF)')
        self.frame.columnconfigure(1, weight=1)

        for i, (lbl, val) in enumerate(self.CHOICES):
            ttk.Radiobutton(self.frame, text=lbl, variable=self.sv_method,
                            value=val, command=self._update).grid(
                row=i, column=0, columnspan=3, sticky='w', padx=4, pady=1)

        r = len(self.CHOICES)
        self._path_lbl = ttk.Label(self.frame, text='IRF file')
        self._path_lbl.grid(row=r, column=0, sticky='e', padx=6, pady=3)
        self._path_e = ttk.Entry(self.frame, textvariable=self.sv_path, width=45)
        self._path_e.grid(row=r, column=1, sticky='ew', padx=4, pady=3)
        self._path_btn = ttk.Button(
            self.frame, text='Browse...',
            command=self._browse_irf_path)
        self._path_btn.grid(row=r, column=2, padx=4, pady=3)

        self._note = ttk.Label(
            self.frame,
            text='Uses the XLSX entered in Input Files above',
            foreground='grey')
        self._note.grid(row=r, column=0, columnspan=3, sticky='w', padx=8, pady=3)

        self._update()

    def _browse_irf_path(self):
        if self.sv_method.get().startswith('machine_irf'):
            _browse_file(self.sv_path, 'Select machine IRF',
                         [('NumPy array', '*.npy'), ('All', '*.*')])
        else:
            _browse_file(self.sv_path, 'Select IRF file',
                         [('PTU / XLSX', '*.ptu *.xlsx'), ('All', '*.*')])

    def _show_browse(self):
        method = self.sv_method.get()
        self._path_lbl.config(
            text='Machine IRF (.npy) path' if method.startswith('machine_irf') else 'IRF file')
        if method.startswith('machine_irf') and not self.sv_path.get().endswith('.npy'):
            self.sv_path.set(self._machine_irf_default)
        self._path_lbl.grid()
        self._path_e.grid()
        self._path_btn.grid()
        self._note.grid_remove()

    def _show_note(self):
        self._path_lbl.grid_remove()
        self._path_e.grid_remove()
        self._path_btn.grid_remove()
        self._note.grid()

    def _hide_all(self):
        self._path_lbl.grid_remove()
        self._path_e.grid_remove()
        self._path_btn.grid_remove()
        self._note.grid_remove()

    def _update(self):
        method = self.sv_method.get()
        if method == 'irf_xlsx':
            self._show_note() if self.xlsx_var is not None else self._show_browse()
        elif method in ('file',) or method.startswith('machine_irf'):
            self._show_browse()
        else:
            self._hide_all()

    def grid(self, **kw):
        self.frame.grid(**kw)

    def get_args(self, xlsx_fallback: Optional[str] = None) -> dict:
        method = self.sv_method.get()
        path   = self.sv_path.get().strip() or None
        if method == 'irf_xlsx':
            xlsx = (self.xlsx_var.get().strip() if self.xlsx_var else None) \
                   or xlsx_fallback or path
            return dict(irf=None, irf_xlsx=xlsx, estimate_irf='none', no_xlsx_irf=False, machine_irf=None)
        elif method.startswith('machine_irf'):
            return dict(irf=None, irf_xlsx=None, estimate_irf=method, no_xlsx_irf=True, machine_irf=path)
        elif method == 'file':
            return dict(irf=path, irf_xlsx=None, estimate_irf='none', no_xlsx_irf=True, machine_irf=None)
        elif method in ('raw', 'parametric'):
            return dict(irf=None, irf_xlsx=None, estimate_irf=method, no_xlsx_irf=True, machine_irf=None)
        else:
            return dict(irf=None, irf_xlsx=None, estimate_irf='none', no_xlsx_irf=True, machine_irf=None)
