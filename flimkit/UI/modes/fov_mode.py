import tkinter as tk
from tkinter import ttk

from flimkit.UI.modes.base import BaseMode
from flimkit.UI.utils import PAD, _C, _section, _row, _browse_file, _browse_dir, _tog, FLIM_FILETYPES
from flimkit.UI.irf_widget import IRFWidget

class FovMode(BaseMode):
    def build(self):
        outer, tab = self.b._form_inner_frames['fov']
        tab.columnconfigure(0, weight=1)
        ff = _section(tab, 'Input Files')
        ff.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ff.columnconfigure(0, weight=1)
        atr = ttk.Frame(ff)
        atr.grid(row=0, column=0, sticky='ew', pady=(0, 2))
        ttk.Label(atr, text='Analysis:').pack(side='left', padx=(2, 8))
        self.b.state.sv_fov_analysis = tk.StringVar(value='single')
        ttk.Radiobutton(atr, text='Single FOV', variable=self.b.sv_fov_analysis,
                        value='single', command=self.b._on_fov_analysis_changed).pack(side='left', padx=2)
        ttk.Radiobutton(atr, text='Z-stack', variable=self.b.sv_fov_analysis,
                        value='zstack', command=self.b._on_fov_analysis_changed).pack(side='left', padx=2)
        ttk.Label(atr, text='(z-stack: a folder of region_zX.ptu slices, fitted as one FOV)',
                  foreground='grey').pack(side='left', padx=(8, 0))
        self.b.state.sv_ptu = tk.StringVar()
        self.b.state.sv_xlsx = tk.StringVar()
        self.b.state.sv_zstack_dir = tk.StringVar()
        self.b.sv_ptu.trace_add('write', self.b._on_fov_ptu_changed)
        self.b.sv_zstack_dir.trace_add('write', self.b._on_zstack_dir_changed)
        single_fr = ttk.Frame(ff)
        single_fr.grid(row=1, column=0, sticky='ew')
        single_fr.columnconfigure(1, weight=1)
        _row(single_fr, 'Data file *', self.b.sv_ptu, 0,
             lambda: _browse_file(self.b.sv_ptu, 'FLIM file', FLIM_FILETYPES))
        self.b._fov_single_fr = single_fr
        zstack_fr = ttk.Frame(ff)
        zstack_fr.grid(row=1, column=0, sticky='ew')
        zstack_fr.columnconfigure(1, weight=1)
        _row(zstack_fr, 'Z-stack folder *', self.b.sv_zstack_dir, 0,
             lambda: _browse_dir(self.b.sv_zstack_dir,
                                 'Folder of region_zX.ptu z-slices'))
        self.b._fov_zstack_fr = zstack_fr
        zstack_fr.grid_remove()
        xlsx_fr = ttk.Frame(ff)
        xlsx_fr.grid(row=2, column=0, sticky='ew')
        xlsx_fr.columnconfigure(1, weight=1)
        _row(xlsx_fr, 'LAS X export (optional)', self.b.sv_xlsx, 0,
             lambda: _browse_file(self.b.sv_xlsx, 'LAS X export',
                                  [('LAS X export', '*.xlsx *.csv'), ('All', '*.*')]))
        self.b._irf_fov = IRFWidget(tab, default='irf_xlsx', xlsx_var=self.b.sv_xlsx,
                                   machine_irf_default=str(_C()['MACHINE_IRF_DEFAULT_PATH']))
        self.b._irf_fov.grid(row=1, column=0, sticky='ew', pady=(0, 6))
        fp = _section(tab, 'Fitting Parameters')
        fp.grid(row=2, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(fp, text='Fit model:').grid(row=0, column=0, sticky='w', **PAD)
        self.b.state.sv_fit_model_fov = tk.StringVar(value='discrete')
        ttk.Radiobutton(fp, text='n-exp', variable=self.b.sv_fit_model_fov,
                        value='discrete').grid(row=0, column=1, sticky='w', padx=1)
        ttk.Radiobutton(fp, text='Gaussian dist.', variable=self.b.sv_fit_model_fov,
                        value='gaussian').grid(row=0, column=2, sticky='w', padx=1)
        ttk.Radiobutton(fp, text='Lorentzian dist.', variable=self.b.sv_fit_model_fov,
                        value='lorentzian').grid(row=0, column=3, sticky='w', padx=1)
        ttk.Radiobutton(fp, text='n-exp tail', variable=self.b.sv_fit_model_fov,
                        value='tail').grid(row=0, column=4, sticky='w', padx=1)
        self.b.state.iv_nexp_fov = tk.IntVar(value=2)
        self.b.state.iv_ncomp_dist_fov = tk.IntVar(value=1)
        nexp_frame = ttk.Frame(fp)
        nexp_frame.grid(row=1, column=0, columnspan=5, sticky='w')
        ttk.Label(nexp_frame, text='Components:').pack(side='left', padx=(4, 8))
        for n in (1, 2, 3):
            ttk.Radiobutton(nexp_frame, text=str(n), variable=self.b.iv_nexp_fov,
                            value=n).pack(side='left', padx=1)
        dist_frame = ttk.Frame(fp)
        ttk.Label(dist_frame, text='Components:').pack(side='left', padx=(4, 8))
        ttk.Radiobutton(dist_frame, text='1 (unimodal)', variable=self.b.iv_ncomp_dist_fov,
                        value=1).pack(side='left', padx=1)
        ttk.Radiobutton(dist_frame, text='2 (bimodal)', variable=self.b.iv_ncomp_dist_fov,
                        value=2).pack(side='left', padx=1)
        
        tail_note = ttk.Label(fp, text='Tail fit: no IRF used, fitted past the decay peak',
                              foreground='#888')
        def _on_fov_model_change(*_):
            _m = self.b.sv_fit_model_fov.get()
            if _m in ('discrete', 'tail'):
                dist_frame.grid_remove()
                nexp_frame.grid(row=1, column=0, columnspan=5, sticky='w')
            else:
                nexp_frame.grid_remove()
                dist_frame.grid(row=1, column=0, columnspan=5, sticky='w')
            if _m == 'tail':
                tail_note.grid(row=5, column=0, columnspan=5, sticky='w', padx=4)
                self.b._irf_fov.grid_remove()
            else:
                tail_note.grid_remove()
                self.b._irf_fov.grid(row=1, column=0, sticky='ew', pady=(0, 6))
        self.b.sv_fit_model_fov.trace_add('write', _on_fov_model_change)
        mode_row = ttk.Frame(fp)
        mode_row.grid(row=2, column=0, columnspan=5, sticky='w', pady=(2, 0))
        ttk.Label(mode_row, text='Fitting mode:').pack(side='left', padx=(0, 10))
        self.b.state.sv_mode_fov = tk.StringVar(value='both')
        radio_frame = ttk.Frame(mode_row)
        radio_frame.pack(side='left')
        ttk.Radiobutton(radio_frame, text='Full', variable=self.b.sv_mode_fov,
                        value='both').pack(side='left', padx=2)
        ttk.Radiobutton(radio_frame, text='Fast', variable=self.b.sv_mode_fov,
                        value='summed').pack(side='left', padx=2)
        ttk.Label(mode_row, text='(fast = no FLIM image)',
                  foreground='grey').pack(side='left', padx=(10, 0))
        ttk.Label(fp, text='Fit window (ns):').grid(row=3, column=0, sticky='w', **PAD)
        self.b.state.sv_tau_min_fov = tk.StringVar(value=str(_C()['Tau_min']))
        self.b.state.sv_tau_max_fov = tk.StringVar(value=str(_C()['Tau_max']))
        ttk.Entry(fp, textvariable=self.b.sv_tau_min_fov, width=7).grid(row=3, column=1, sticky='w', padx=4)
        ttk.Label(fp, text='to').grid(row=3, column=2)
        ttk.Entry(fp, textvariable=self.b.sv_tau_max_fov, width=7).grid(row=3, column=3, sticky='w', padx=4)
        ttk.Label(fp, text='ns  (fitting range)', foreground='grey').grid(row=3, column=4, sticky='w')
        ttk.Label(fp, text='Output prefix:').grid(row=4, column=0, sticky='w', **PAD)
        self.b.state.sv_out_fov = tk.StringVar(value='flim_out')
        ttk.Entry(fp, textvariable=self.b.sv_out_fov, width=35).grid(
            row=4, column=1, columnspan=3, sticky='ew', padx=4)
        fm = _section(tab, 'Masking & Thresholding')
        fm.grid(row=3, column=0, sticky='ew', pady=(0, 6))
        self.b.state.bv_cell = tk.BooleanVar(value=False)
        ttk.Checkbutton(fm, text='Apply cell mask (Cellpose-SAM)',
                        variable=self.b.bv_cell).grid(
            row=0, column=0, columnspan=3, sticky='w', **PAD)
        self.b.state.bv_thr_fov = tk.BooleanVar(value=False)
        self.b.state.sv_thr_fov = tk.StringVar()
        ttk.Checkbutton(fm, text='Intensity threshold (min photons/px):',
                        variable=self.b.bv_thr_fov,
                        command=lambda: _tog(self.b.bv_thr_fov, self.b._thr_fov_e)).grid(
            row=1, column=0, sticky='w', **PAD)
        self.b._thr_fov_e = ttk.Entry(fm, textvariable=self.b.sv_thr_fov,
                                    width=8, state='disabled')
        self.b._thr_fov_e.grid(row=1, column=1, sticky='w', padx=4)
        ttk.Label(fm, text='(leave blank for no threshold)',
                  foreground='grey').grid(row=1, column=2, sticky='w')
        self.b.state.bv_correct_pileup = tk.BooleanVar(value=False)
        ttk.Checkbutton(fm, text='Apply Coates pile-up correction (recommended if pile-up > 5%)',
                        variable=self.b.bv_correct_pileup).grid(
            row=2, column=0, columnspan=3, sticky='w', **PAD)
        ttk.Label(fm, text='Time-varying background PTU:').grid(row=3, column=0, sticky='w', **PAD)
        self.b.state.sv_tvb_ptu_fov = tk.StringVar()
        ttk.Entry(fm, textvariable=self.b.sv_tvb_ptu_fov, width=24).grid(
            row=3, column=1, sticky='ew', padx=4)
        ttk.Button(fm, text='Browse...',
                   command=lambda: _browse_file(self.b.sv_tvb_ptu_fov,
                                                'Background reference PTU',
                                                FLIM_FILETYPES)).grid(
            row=3, column=2, sticky='w', padx=4)
        ttk.Label(fm, text='(optional: fits a measured fluorophore-free background decay, FLIMfit-style)',
                  foreground='grey').grid(row=4, column=0, columnspan=3, sticky='w', padx=8)
        self.b._expert_banner_fov = ttk.Label(
            tab, text='⚙  Custom expert settings active',
            foreground='#e8a838', font=('TkDefaultFont', 9, 'bold'))
        self.b._expert_banner_fov.grid(row=4, column=0, sticky='w', padx=8)
        self.b._expert_banner_fov.grid_remove()
        btn_row_fov = ttk.Frame(tab)
        btn_row_fov.grid(row=5, column=0, pady=8)
        ttk.Button(btn_row_fov, text='⚙  Expert Settings',
                   command=self.b._open_expert_settings).pack(side='left', padx=4)
        self.b._btn_fov = ttk.Button(btn_row_fov, text='▶  Run Single-FOV Fit',
                                   command=self.b._run_fov)
        self.b._btn_fov.pack(side='left', padx=4, ipadx=20, ipady=4)
