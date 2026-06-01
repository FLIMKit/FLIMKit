import tkinter as tk
from tkinter import ttk

from flimkit.UI.modes.base import BaseMode
from flimkit.UI.utils import PAD, _C, _section, _row, _browse_file
from flimkit.UI.phasor_panel import PhasorViewPanel


class PhasorMode(BaseMode):
    def build(self):
        # Get outer and inner frames from tuple
        outer, inner = self.b._form_inner_frames['phasor']
        inner.columnconfigure(0, weight=1)
        #  Controls strip (fixed height, top) ─
        ctrl = ttk.Frame(inner, padding=(6, 4))
        ctrl.grid(row=0, column=0, sticky='ew')
        ctrl.columnconfigure(0, weight=1)

        # Input mode
        mode_fr = _section(ctrl, 'Input Mode')
        mode_fr.grid(row=0, column=0, sticky='ew', pady=(0, 4))
        mode_fr.columnconfigure(1, weight=1)

        self.b.state.sv_ph_mode = tk.StringVar(value='new')
        ttk.Radiobutton(mode_fr, text='New PTU file',
                        variable=self.b.sv_ph_mode, value='new',
                        command=self.b._ph_mode_changed).grid(
            row=0, column=0, sticky='w', padx=4, pady=1)
        ttk.Radiobutton(mode_fr, text='Resume session (.npz)',
                        variable=self.b.sv_ph_mode, value='session',
                        command=self.b._ph_mode_changed).grid(
            row=0, column=1, sticky='w', padx=4, pady=1)

        # New-PTU sub-frame
        self.b._ph_new = ttk.Frame(ctrl)
        self.b._ph_new.columnconfigure(0, weight=1)
        self.b._ph_new.grid(row=1, column=0, sticky='ew')
        fn = _section(self.b._ph_new, 'New Analysis')
        fn.grid(row=0, column=0, sticky='ew')
        fn.columnconfigure(1, weight=1)
        self.b.state.sv_ph_ptu  = tk.StringVar()
        self.b.state.sv_ph_irf  = tk.StringVar()
        self.b.state.sv_ph_mirf = tk.StringVar(
            value=str(_C()['MACHINE_IRF_DEFAULT_PATH']))
        _row(fn, 'PTU file *',             self.b.sv_ph_ptu,  0,
             lambda: _browse_file(self.b.sv_ph_ptu, 'PTU file',
                                  [('PTU', '*.ptu'), ('All', '*.*')]))
        _row(fn, 'IRF XLSX (optional)',    self.b.sv_ph_irf,  1,
             lambda: _browse_file(self.b.sv_ph_irf, 'IRF XLSX',
                                  [('Excel', '*.xlsx'), ('All', '*.*')]))
        _row(fn, 'Machine IRF (optional)', self.b.sv_ph_mirf, 2,
             lambda: _browse_file(self.b.sv_ph_mirf, 'Machine IRF',
                                  [('NumPy', '*.npy'), ('All', '*.*')]))
        ttk.Label(fn, text='XLSX takes priority if both supplied',
                  foreground='grey').grid(
            row=3, column=1, columnspan=2, sticky='w', padx=4)

        # Session sub-frame
        self.b._ph_sess = ttk.Frame(ctrl)
        self.b._ph_sess.columnconfigure(0, weight=1)
        self.b._ph_sess.grid(row=2, column=0, sticky='ew')
        fs = _section(self.b._ph_sess, 'Resume Session')
        fs.grid(row=0, column=0, sticky='ew')
        fs.columnconfigure(1, weight=1)
        self.b.state.sv_ph_session = tk.StringVar()
        _row(fs, 'Session (.npz) *', self.b.sv_ph_session, 0,
             lambda: _browse_file(self.b.sv_ph_session, 'Session file',
                                  [('NPZ', '*.npz'), ('All', '*.*')]))
        self.b._ph_sess.grid_remove()

        # Display options
        opt_fr = _section(ctrl, 'Display Options')
        opt_fr.grid(row=3, column=0, sticky='ew', pady=(4, 0))
        ttk.Label(opt_fr, text='Min photons (fraction):').grid(
            row=0, column=0, sticky='w', **PAD)
        self.b.state.sv_ph_minph = tk.StringVar(value='0.01')
        ttk.Entry(opt_fr, textvariable=self.b.sv_ph_minph, width=8).grid(
            row=0, column=1, sticky='w', padx=4)
        ttk.Label(opt_fr, text='Max cursors:').grid(
            row=0, column=2, sticky='w', padx=8)
        self.b.state.sv_ph_maxc = tk.StringVar(value='6')
        ttk.Entry(opt_fr, textvariable=self.b.sv_ph_maxc, width=4).grid(
            row=0, column=3, sticky='w', padx=4)

        # Run button
        self.b._btn_ph = ttk.Button(ctrl, text='▶  Load & Analyse',
                                   command=self.b._run_phasor)
        self.b._btn_ph.grid(row=4, column=0, pady=(6, 2), ipadx=16, ipady=3,
                          sticky='w')

        # Find Peaks section
        peaks_fr = _section(ctrl, 'Find Peaks')
        peaks_fr.grid(row=5, column=0, sticky='ew', pady=(6, 0))
        peaks_fr.columnconfigure(1, weight=1)
        ttk.Label(peaks_fr, text='Smooth σ:').grid(
            row=0, column=0, sticky='w', **PAD)
        self.b.state.sv_ph_pk_sigma = tk.StringVar(value='3.0')
        ttk.Entry(peaks_fr, textvariable=self.b.sv_ph_pk_sigma, width=6).grid(
            row=0, column=1, sticky='w', padx=4)
        ttk.Label(peaks_fr, text='Threshold:').grid(
            row=1, column=0, sticky='w', **PAD)
        self.b.state.sv_ph_pk_thresh = tk.StringVar(value='0.10')
        ttk.Entry(peaks_fr, textvariable=self.b.sv_ph_pk_thresh, width=6).grid(
            row=1, column=1, sticky='w', padx=4)
        ttk.Button(peaks_fr, text='🔍  Find Peaks',
                   command=self.b._run_ph_find_peaks).grid(
            row=2, column=0, columnspan=2, sticky='w', pady=(4, 0), padx=8)

        # FRET Analysis section
        fret_fr = _section(ctrl, 'FRET Analysis')
        fret_fr.grid(row=6, column=0, sticky='ew', pady=(6, 0))
        fret_fr.columnconfigure(1, weight=1)
        ttk.Label(fret_fr, text='Donor τ (ns):').grid(
            row=0, column=0, sticky='w', **PAD)
        self.b.state.sv_ph_fret_taud = tk.StringVar(value='4.0')
        ttk.Entry(fret_fr, textvariable=self.b.sv_ph_fret_taud, width=8).grid(
            row=0, column=1, sticky='w', padx=4)
        ttk.Label(fret_fr, text='Acceptor τ (ns):').grid(
            row=1, column=0, sticky='w', **PAD)
        self.b.state.sv_ph_fret_taua = tk.StringVar(value='')
        ttk.Entry(fret_fr, textvariable=self.b.sv_ph_fret_taua, width=8).grid(
            row=1, column=1, sticky='w', padx=4)
        ttk.Label(fret_fr, text='(blank = donor-only)',
                  foreground='grey').grid(
            row=1, column=2, sticky='w', padx=2)
        ttk.Label(fret_fr, text='Donor fretting:').grid(
            row=2, column=0, sticky='w', **PAD)
        self.b.state.sv_ph_fret_fretting = tk.StringVar(value='1.0')
        ttk.Entry(fret_fr, textvariable=self.b.sv_ph_fret_fretting, width=8).grid(
            row=2, column=1, sticky='w', padx=4)
        _fret_btn_fr = ttk.Frame(fret_fr)
        _fret_btn_fr.grid(row=3, column=0, columnspan=3,
                          sticky='w', pady=(4, 0), padx=8)
        ttk.Button(_fret_btn_fr, text='↔  Overlay Trajectory',
                   command=self.b._run_ph_fret_overlay).pack(side='left', padx=(0, 4))
        ttk.Button(_fret_btn_fr, text='Fit Donor FRET',
                   command=self.b._run_ph_fit_fret).pack(side='left', padx=(0, 4))
        ttk.Button(_fret_btn_fr, text='✕  Clear Overlay',
                   command=lambda: self.b._phasor_panel.clear_fret_overlay()).pack(
            side='left')

        # (PhasorViewPanel lives in the right FOV-preview panel - see _init_ui)
