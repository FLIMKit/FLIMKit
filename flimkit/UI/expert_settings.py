from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from flimkit.UI.utils import _C
from typing import Optional

_EXPERT_DEFAULTS = {
    'binning_factor': 1,
    'optimizer': 'de',
    'lm_restarts': 8,
    'de_population': 30,
    'de_maxiter': 5000,
    'n_workers': -1,
    'cost_function': 'poisson',
    'channels': '',
    'min_photons': 10,
    'irf_fwhm': None,
    'irf_align': 'steepest_rise',
    'irf_shift_bins': 2,
    'free_tau_perpixel': False,
}


class ExpertSettingsDialog(tk.Toplevel):

    def __init__(self, parent, current: dict):
        super().__init__(parent)
        self.title('Expert Fit Settings')
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: Optional[dict] = None
        cfg = _C()

        vals = dict(_EXPERT_DEFAULTS)
        vals.update({
            'binning_factor': cfg['binning_factor'],
            'optimizer': cfg['Optimizer'],
            'lm_restarts': cfg['lm_restarts'],
            'de_population': cfg['de_population'],
            'de_maxiter': cfg['de_maxiter'],
            'n_workers': cfg['n_workers'],
            'min_photons': cfg['MIN_PHOTONS_PERPIX'],
        })
        vals.update(current)

        PAD = {'padx': 4, 'pady': 3}
        row = 0
        f = ttk.Frame(self, padding=12)
        f.pack(fill='both', expand=True)

        ttk.Label(f, text='Optimizer:').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_optimizer = tk.StringVar(value=vals['optimizer'])
        opt_frame = ttk.Frame(f)
        opt_frame.grid(row=row, column=1, columnspan=3, sticky='w', **PAD)
        ttk.Radiobutton(opt_frame, text='Differential Evolution (DE)',
                        variable=self._sv_optimizer, value='de').pack(side='left', padx=(0, 8))
        ttk.Radiobutton(opt_frame, text='Levenberg-Marquardt (LM)',
                        variable=self._sv_optimizer, value='lm_multistart').pack(side='left')

        row += 1
        ttk.Label(f, text='DE population:').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_de_pop = tk.StringVar(value=str(vals['de_population']))
        ttk.Entry(f, textvariable=self._sv_de_pop, width=8).grid(row=row, column=1, sticky='w', **PAD)
        ttk.Label(f, text='DE max iterations:').grid(row=row, column=2, sticky='w', **PAD)
        self._sv_de_maxiter = tk.StringVar(value=str(vals['de_maxiter']))
        ttk.Entry(f, textvariable=self._sv_de_maxiter, width=8).grid(row=row, column=3, sticky='w', **PAD)

        row += 1
        ttk.Label(f, text='LM random restarts:').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_lm_restarts = tk.StringVar(value=str(vals['lm_restarts']))
        ttk.Entry(f, textvariable=self._sv_lm_restarts, width=8).grid(row=row, column=1, sticky='w', **PAD)

        row += 1
        ttk.Label(f, text='Spatial binning (NxN):').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_binning = tk.StringVar(value=str(vals['binning_factor']))
        ttk.Entry(f, textvariable=self._sv_binning, width=8).grid(row=row, column=1, sticky='w', **PAD)
        ttk.Label(f, text='(1 = no binning)', foreground='grey').grid(row=row, column=2, columnspan=2, sticky='w', **PAD)

        row += 1
        ttk.Label(f, text='CPU workers:').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_workers = tk.StringVar(value=str(vals['n_workers']))
        ttk.Entry(f, textvariable=self._sv_workers, width=8).grid(row=row, column=1, sticky='w', **PAD)
        ttk.Label(f, text='(-1 = all cores)', foreground='grey').grid(row=row, column=2, columnspan=2, sticky='w', **PAD)

        row += 1
        ttk.Label(f, text='Min photons/pixel:').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_min_ph = tk.StringVar(value=str(vals['min_photons']))
        ttk.Entry(f, textvariable=self._sv_min_ph, width=8).grid(row=row, column=1, sticky='w', **PAD)

        row += 1
        ttk.Label(f, text='Cost function:').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_cost = tk.StringVar(value=vals['cost_function'])
        cf_frame = ttk.Frame(f)
        cf_frame.grid(row=row, column=1, columnspan=3, sticky='w', **PAD)
        ttk.Radiobutton(cf_frame, text='Poisson deviance',
                        variable=self._sv_cost, value='poisson').pack(side='left', padx=(0, 8))
        ttk.Radiobutton(cf_frame, text='Chi² (legacy)',
                        variable=self._sv_cost, value='chi2').pack(side='left')

        row += 1
        ttk.Label(f, text='Channel filter:').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_channels = tk.StringVar(value=str(vals.get('channels', '') or ''))
        ttk.Entry(f, textvariable=self._sv_channels, width=12).grid(row=row, column=1, sticky='w', **PAD)
        ttk.Label(f, text='(blank = all channels)', foreground='grey').grid(row=row, column=2, columnspan=2, sticky='w', **PAD)

        row += 1
        ttk.Label(f, text='IRF FWHM (ns):').grid(row=row, column=0, sticky='w', **PAD)
        _irf_fwhm_val = vals.get('irf_fwhm')
        self._sv_irf_fwhm = tk.StringVar(value='' if _irf_fwhm_val is None else str(_irf_fwhm_val))
        ttk.Entry(f, textvariable=self._sv_irf_fwhm, width=12).grid(row=row, column=1, sticky='w', **PAD)
        ttk.Label(f, text='(blank = 1 bin auto, e.g. 0.097)', foreground='grey').grid(row=row, column=2, columnspan=2, sticky='w', **PAD)

        row += 1
        ttk.Label(f, text='IRF alignment:').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_irf_align = tk.StringVar(value=vals.get('irf_align', 'steepest_rise'))
        align_frame = ttk.Frame(f)
        align_frame.grid(row=row, column=1, columnspan=3, sticky='w', **PAD)
        ttk.Radiobutton(align_frame, text='Steepest rise (recommended)',
                        variable=self._sv_irf_align, value='steepest_rise').pack(side='left', padx=(0, 8))
        ttk.Radiobutton(align_frame, text='Decay peak (legacy)',
                        variable=self._sv_irf_align, value='decay_peak').pack(side='left')

        row += 1
        ttk.Label(f, text='IRF shift bound (±bins):').grid(row=row, column=0, sticky='w', **PAD)
        self._sv_irf_shift = tk.StringVar(value=str(vals.get('irf_shift_bins', 2)))
        ttk.Entry(f, textvariable=self._sv_irf_shift, width=8).grid(row=row, column=1, sticky='w', **PAD)
        ttk.Label(f, text='(2 = recommended; 5 = legacy)', foreground='grey').grid(row=row, column=2, columnspan=2, sticky='w', **PAD)

        row += 1
        self._bv_free_tau = tk.BooleanVar(value=bool(vals.get('free_tau_perpixel', False)))
        ttk.Checkbutton(f, text='Free τ per pixel  (slower - reveals τ spatial variation for n_exp > 1)',
                        variable=self._bv_free_tau).grid(
            row=row, column=0, columnspan=4, sticky='w', **PAD)

        row += 1
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=(12, 0))
        ttk.Button(btn_frame, text='Confirm', command=self._confirm).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='Reset Defaults', command=self._reset).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='Cancel', command=self.destroy).pack(side='left', padx=4)

        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

    def _collect(self) -> dict:
        ch = self._sv_channels.get().strip()
        _fwhm_s = self._sv_irf_fwhm.get().strip()
        return {
            'optimizer': self._sv_optimizer.get(),
            'de_population': int(self._sv_de_pop.get() or 30),
            'de_maxiter': int(self._sv_de_maxiter.get() or 5000),
            'lm_restarts': int(self._sv_lm_restarts.get() or 8),
            'binning_factor': int(self._sv_binning.get() or 1),
            'n_workers': int(self._sv_workers.get() or -1),
            'min_photons': int(self._sv_min_ph.get() or 10),
            'cost_function': self._sv_cost.get(),
            'channels': int(ch) if ch.isdigit() else (None if ch == '' else ch),
            'irf_fwhm': float(_fwhm_s) if _fwhm_s else None,
            'irf_align': self._sv_irf_align.get(),
            'irf_shift_bins': int(self._sv_irf_shift.get() or 2),
            'free_tau_perpixel': self._bv_free_tau.get(),
        }

    def _confirm(self):
        try:
            self.result = self._collect()
        except ValueError as e:
            messagebox.showerror('Invalid value', str(e), parent=self)
            return
        self.destroy()

    def _reset(self):
        d = _EXPERT_DEFAULTS
        self._sv_optimizer.set(d['optimizer'])
        self._sv_de_pop.set(str(d['de_population']))
        self._sv_de_maxiter.set(str(d['de_maxiter']))
        self._sv_lm_restarts.set(str(d['lm_restarts']))
        self._sv_binning.set(str(d['binning_factor']))
        self._sv_workers.set(str(d['n_workers']))
        self._sv_min_ph.set(str(d['min_photons']))
        self._sv_cost.set(d['cost_function'])
        self._sv_channels.set('')
        self._sv_irf_fwhm.set('')
        self._sv_irf_align.set('steepest_rise')
        self._sv_irf_shift.set('2')
        self._bv_free_tau.set(False)
