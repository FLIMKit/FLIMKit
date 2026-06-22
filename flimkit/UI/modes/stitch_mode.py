import tkinter as tk
from tkinter import ttk

from flimkit.UI.modes.base import BaseMode
from flimkit.UI.utils import PAD, _C, _section, _row, _browse_file, _browse_dir, _tog
from flimkit.UI.irf_widget import IRFWidget


class StitchMode(BaseMode):
    def build(self):
        outer, tab = self.b._form_inner_frames['stitch']
        tab.columnconfigure(0, weight=1)

        ff = _section(tab, 'Input Files')
        ff.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.b.state.sv_xlif    = tk.StringVar()
        self.b.sv_xlif.trace_add('write', self.b._on_xlif_changed)
        self.b.state.sv_ptu_dir = tk.StringVar()
        self.b.state.sv_out_st  = tk.StringVar()

        _row(ff, 'XLIF metadata *',      self.b.sv_xlif,    0,
             lambda: _browse_file(self.b.sv_xlif, 'XLIF file',
                                  [('XLIF', '*.xlif'), ('All', '*.*')]))
        _row(ff, 'PTU tile directory *', self.b.sv_ptu_dir, 1,
             lambda: _browse_dir(self.b.sv_ptu_dir, 'PTU tile directory'))
        _row(ff, 'Base output dir *',    self.b.sv_out_st,  2,
             lambda: _browse_dir(self.b.sv_out_st, 'Output directory'))

        ttk.Label(ff, text='(A sub-folder named after the ROI will be created inside)',
                  foreground='grey').grid(row=3, column=1, columnspan=2,
                                         sticky='w', padx=4)
        self.b.state.bv_rotate = tk.BooleanVar(value=True)
        ttk.Checkbutton(ff, text='Rotate tiles 90° CW (recommended for FLIM microscope)',
                        variable=self.b.bv_rotate).grid(
            row=4, column=0, columnspan=3, sticky='w', padx=4, pady=(4, 0))

        fp = _section(tab, 'Pipeline')
        fp.grid(row=1, column=0, sticky='ew', pady=(4, 2))
        self.b.state.sv_pipeline = tk.StringVar(value='stitch_only')
        for r, (val, lbl) in enumerate([
            ('stitch_only', 'Stitch tiles only'),
            ('stitch_fit',  'Stitch then fit full ROI'),
            ('tile_fit',    'Per-tile fit  [recommended - fits each tile independently]'),
        ]):
            ttk.Radiobutton(fp, text=lbl, variable=self.b.sv_pipeline,
                            value=val, command=self.b._pipeline_changed).grid(
                row=r, column=0, sticky='w', padx=4, pady=1)

        self.b._fit_frame = ttk.Frame(tab)
        self.b._fit_frame.columnconfigure(0, weight=1)
        self.b._fit_frame.grid(row=2, column=0, sticky='ew')
        self.build_fit(self.b._fit_frame)
        self.b._fit_frame.grid_remove()

        self.b._expert_banner_st = ttk.Label(
            tab, text='⚙  Custom expert settings active',
            foreground='#e8a838', font=('TkDefaultFont', 9, 'bold'))
        self.b._expert_banner_st.grid(row=3, column=0, sticky='w', padx=8)
        self.b._expert_banner_st.grid_remove()

        btn_row_st = ttk.Frame(tab)
        btn_row_st.grid(row=4, column=0, pady=8)
        self.b._btn_expert_st = ttk.Button(btn_row_st, text='⚙  Expert Settings',
                   command=self.b._open_expert_settings)
        self.b._btn_expert_st.pack(side='left', padx=4)
        self.b._btn_st = ttk.Button(btn_row_st, text='▶  Run Tile Stitch',
                                  command=self.b._run_stitch)
        self.b._btn_st.pack(side='left', padx=4, ipadx=20, ipady=4)

    def build_fit(self, parent):
        self.b._irf_st = IRFWidget(parent, default='machine_irf',
                                  machine_irf_default=str(_C()['MACHINE_IRF_DEFAULT_PATH']))
        self.b._irf_st.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        self.b._irf_st.frame.columnconfigure(1, weight=1)

        fp = _section(parent, 'Fitting Parameters')
        fp.grid(row=1, column=0, sticky='ew', pady=(0, 6))

        ttk.Label(fp, text='Fit model:').grid(row=0, column=0, sticky='w', **PAD)
        self.b.state.sv_fit_model_st = tk.StringVar(value='discrete')
        ttk.Radiobutton(fp, text='n-exp', variable=self.b.sv_fit_model_st,
                        value='discrete').grid(row=0, column=1, sticky='w', padx=1)
        ttk.Radiobutton(fp, text='Gaussian dist.', variable=self.b.sv_fit_model_st,
                        value='gaussian').grid(row=0, column=2, sticky='w', padx=1)
        ttk.Radiobutton(fp, text='Lorentzian dist.', variable=self.b.sv_fit_model_st,
                        value='lorentzian').grid(row=0, column=3, sticky='w', padx=1)

        self.b.state.iv_nexp_st = tk.IntVar(value=2)
        self.b.state.iv_ncomp_dist_st = tk.IntVar(value=1)

        nexp_frame_st = ttk.Frame(fp)
        nexp_frame_st.grid(row=1, column=0, columnspan=5, sticky='w')
        ttk.Label(nexp_frame_st, text='Components:').pack(side='left', padx=(4, 8))
        for n in (1, 2, 3):
            ttk.Radiobutton(nexp_frame_st, text=str(n), variable=self.b.iv_nexp_st,
                            value=n).pack(side='left', padx=4)

        dist_frame_st = ttk.Frame(fp)
        ttk.Label(dist_frame_st, text='Components:').pack(side='left', padx=(4, 8))
        ttk.Radiobutton(dist_frame_st, text='1 (unimodal)', variable=self.b.iv_ncomp_dist_st,
                        value=1).pack(side='left', padx=4)
        ttk.Radiobutton(dist_frame_st, text='2 (bimodal)', variable=self.b.iv_ncomp_dist_st,
                        value=2).pack(side='left', padx=4)

        def _on_st_model_change(*_):
            if self.b.sv_fit_model_st.get() == 'discrete':
                dist_frame_st.grid_remove()
                nexp_frame_st.grid(row=1, column=0, columnspan=5, sticky='w')
            else:
                nexp_frame_st.grid_remove()
                dist_frame_st.grid(row=1, column=0, columnspan=5, sticky='w')
        self.b.sv_fit_model_st.trace_add('write', _on_st_model_change)

        self.b.state.bv_perpix = tk.BooleanVar(value=False)
        ttk.Checkbutton(fp, text='Per-pixel fitting [REQUIRED FOR ROI ANALYSIS]',
                        variable=self.b.bv_perpix,
                        command=self.b._perpix_toggled).grid(
            row=2, column=0, columnspan=4, sticky='w', **PAD)

        self.b._pxf = ttk.Frame(fp)
        self.b._pxf.grid(row=3, column=0, columnspan=4, sticky='ew', padx=20)

        self.b.state.bv_save_tau_weighted = tk.BooleanVar(value=True)
        self.b.state.bv_save_int_weighted = tk.BooleanVar(value=True)
        self.b.state.bv_save_amp_weighted = tk.BooleanVar(value=False)
        self.b.state.bv_save_ind = tk.BooleanVar(value=False)

        ttk.Checkbutton(self.b._pxf, text='Export τ-weighted map',
                        variable=self.b.bv_save_tau_weighted).grid(row=0, column=0, sticky='w', padx=(0, 8))
        ttk.Checkbutton(self.b._pxf, text='Export intensity-weighted map',
                        variable=self.b.bv_save_int_weighted).grid(row=0, column=1, sticky='w', padx=(0, 8))
        ttk.Checkbutton(self.b._pxf, text='Export amplitude-weighted map',
                        variable=self.b.bv_save_amp_weighted).grid(row=1, column=0, sticky='w', padx=(0, 8))
        ttk.Checkbutton(self.b._pxf, text='Save individual component maps',
                        variable=self.b.bv_save_ind).grid(row=1, column=1, sticky='w')

        self.b.state.sv_tau_lo = tk.StringVar()
        self.b.state.sv_tau_hi = tk.StringVar()
        self.b.state.sv_int_lo = tk.StringVar()
        self.b.state.sv_int_hi = tk.StringVar()

        ttk.Label(self.b._pxf, text='Lifetime display (ns):').grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(self.b._pxf, textvariable=self.b.sv_tau_lo, width=7).grid(row=2, column=1, padx=4)
        ttk.Label(self.b._pxf, text='to').grid(row=2, column=2)
        ttk.Entry(self.b._pxf, textvariable=self.b.sv_tau_hi, width=7).grid(row=2, column=3, padx=4)
        ttk.Label(self.b._pxf, text='(blank = auto)', foreground='grey').grid(row=2, column=4, padx=4)

        ttk.Label(self.b._pxf, text='Intensity display:').grid(row=3, column=0, sticky='w', pady=2)
        ttk.Entry(self.b._pxf, textvariable=self.b.sv_int_lo, width=7).grid(row=3, column=1, padx=4)
        ttk.Label(self.b._pxf, text='to').grid(row=3, column=2)
        ttk.Entry(self.b._pxf, textvariable=self.b.sv_int_hi, width=7).grid(row=3, column=3, padx=4)
        ttk.Label(self.b._pxf, text='(blank = auto)', foreground='grey').grid(row=3, column=4, padx=4)

        self.b._pxf.grid_remove()

        self.b.state.sv_tau_fit_lo = tk.StringVar(value=str(_C()['Tau_min']))
        self.b.state.sv_tau_fit_hi = tk.StringVar(value=str(_C()['Tau_max']))
        ttk.Label(fp, text='Fit window (ns):').grid(row=3, column=0, sticky='w', pady=2)
        ttk.Entry(fp, textvariable=self.b.sv_tau_fit_lo, width=7).grid(row=3, column=1, padx=4)
        ttk.Label(fp, text='to').grid(row=3, column=2)
        ttk.Entry(fp, textvariable=self.b.sv_tau_fit_hi, width=7).grid(row=3, column=3, padx=4)
        ttk.Label(fp, text='ns  (fitting range)', foreground='grey').grid(row=3, column=4, padx=4)

        fm = _section(parent, 'Masking & Thresholding')
        fm.grid(row=2, column=0, sticky='ew', pady=(0, 6))

        self.b.state.bv_thr_st = tk.BooleanVar(value=False)
        self.b.state.sv_thr_st = tk.StringVar()
        ttk.Checkbutton(fm, text='Intensity threshold (min photons/px):',
                        variable=self.b.bv_thr_st,
                        command=lambda: _tog(self.b.bv_thr_st, self.b._thr_st_e)).grid(
            row=0, column=0, sticky='w', **PAD)
        self.b._thr_st_e = ttk.Entry(fm, textvariable=self.b.sv_thr_st,
                                   width=8, state='disabled')
        self.b._thr_st_e.grid(row=0, column=1, sticky='w', padx=4)
        ttk.Label(fm, text='(leave blank for no threshold)',
                  foreground='grey').grid(row=0, column=2, sticky='w')

        self.b.state.bv_correct_pileup_st = tk.BooleanVar(value=False)
        ttk.Checkbutton(fm, text='Apply Coates pile-up correction (recommended if pile-up > 5%)',
                        variable=self.b.bv_correct_pileup_st).grid(
            row=1, column=0, columnspan=3, sticky='w', **PAD)
        ttk.Label(fm, text='Time-varying background PTU:').grid(row=2, column=0, sticky='w', **PAD)
        self.b.state.sv_tvb_ptu_st = tk.StringVar()
        ttk.Entry(fm, textvariable=self.b.sv_tvb_ptu_st, width=24).grid(
            row=2, column=1, sticky='ew', padx=4)
        ttk.Button(fm, text='Browse...',
                   command=lambda: _browse_file(self.b.sv_tvb_ptu_st,
                                                'Background reference PTU',
                                                [('PTU', '*.ptu'), ('All', '*.*')])).grid(
            row=2, column=2, sticky='w', padx=4)
        ttk.Label(fm, text='(optional: measured background, aligned per tile)',
                  foreground='grey').grid(row=3, column=0, columnspan=3, sticky='w', padx=8)
        freg = _section(parent, 'Tile Registration')
        freg.grid(row=3, column=0, sticky='ew', pady=(0, 6))
        self.b.state.bv_register = tk.BooleanVar(value=True)
        ttk.Checkbutton(freg, text='Phase-correlation registration (fixes stage Y/X drift)',
                        variable=self.b.bv_register).grid(
            row=0, column=0, columnspan=3, sticky='w', **PAD)
        ttk.Label(freg, text='Max shift (px):').grid(row=1, column=0, sticky='w', **PAD)
        self.b.state.sv_reg_max_shift = tk.StringVar(value='120')
        ttk.Entry(freg, textvariable=self.b.sv_reg_max_shift, width=6).grid(
            row=1, column=1, sticky='w', padx=4)
        ttk.Label(freg, text='(increase if drift > 120px)',
                  foreground='grey').grid(row=1, column=2, sticky='w')

        self.b._tile_extras_frame = ttk.Frame(parent)
        self.b._tile_extras_frame.columnconfigure(0, weight=1)
        self.b._tile_extras_frame.grid(row=4, column=0, sticky='ew', pady=(0, 4))
        fte = _section(self.b._tile_extras_frame, 'Per-Tile IRF Directory (optional)')
        fte.grid(row=0, column=0, sticky='ew')
        fte.columnconfigure(1, weight=1)
        self.b.state.sv_tile_irf_dir = tk.StringVar()
        _row(fte, 'IRF XLSX dir', self.b.sv_tile_irf_dir, 0,
             lambda: _browse_dir(self.b.sv_tile_irf_dir, 'Directory of per-tile IRF xlsx files'))
        ttk.Label(fte, text='One <tile_name>.xlsx per tile; leave blank to use IRF method above',
                  foreground='grey').grid(row=1, column=1, columnspan=2, sticky='w', padx=4)
        self.b._tile_extras_frame.grid_remove()
