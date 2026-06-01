import tkinter as tk
from tkinter import ttk

from flimkit.UI.modes.base import BaseMode
from flimkit.UI.utils import PAD, _C, _section, _row, _browse_file, _browse_dir, _tog


class BatchMode(BaseMode):
    def build(self):
        outer, tab = self.b._form_inner_frames['batch']
        tab.columnconfigure(0, weight=1)

        self.b.state.sv_batch_mode = tk.StringVar(value='tiled')

        # Mode label (updated by _batch_mode_changed when menu item chosen)
        self.b._batch_mode_label = ttk.Label(
            tab, text='Mode: Multi-Tile ROI Fit',
            font=('TkDefaultFont', 9, 'bold'), foreground='#555')
        self.b._batch_mode_label.grid(row=0, column=0, sticky='w', padx=8, pady=(6, 2))

        ff = _section(tab, 'Input / Output')
        ff.grid(row=1, column=0, sticky='ew', pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.b.state.sv_batch_xlif_dir = tk.StringVar()
        self.b.state.sv_batch_ptu_dir  = tk.StringVar()
        self.b.state.sv_batch_out_dir  = tk.StringVar()

        # XLIF folder row wrapped in its own frame so it can be hidden in FOV mode
        self.b._batch_xlif_fr = ttk.Frame(ff)
        self.b._batch_xlif_fr.grid(row=0, column=0, columnspan=3, sticky='ew')
        self.b._batch_xlif_fr.columnconfigure(1, weight=1)
        ttk.Label(self.b._batch_xlif_fr, text='XLIF folder *').grid(
            row=0, column=0, sticky='e', padx=6, pady=3)
        ttk.Entry(self.b._batch_xlif_fr, textvariable=self.b.sv_batch_xlif_dir, width=45).grid(
            row=0, column=1, sticky='ew', padx=4, pady=3)
        ttk.Button(self.b._batch_xlif_fr, text='Browse...',
                   command=lambda: _browse_dir(self.b.sv_batch_xlif_dir,
                                               'Folder of XLIF files')).grid(
            row=0, column=2, padx=4, pady=3)

        _row(ff, 'PTU folder *',      self.b.sv_batch_ptu_dir,  1,
             lambda: _browse_dir(self.b.sv_batch_ptu_dir,  'PTU tile directory'))
        _row(ff, 'Output base dir *', self.b.sv_batch_out_dir,  2,
             lambda: _browse_dir(self.b.sv_batch_out_dir,  'Base output directory'))
        self.b._batch_io_help = ttk.Label(
            ff, text='One sub-folder per ROI created inside the output base dir.',
            foreground='grey')
        self.b._batch_io_help.grid(row=3, column=1, columnspan=2, sticky='w', padx=4)

        fi = _section(tab, 'IRF')
        fi.grid(row=2, column=0, sticky='ew', pady=(0, 6))
        fi.columnconfigure(1, weight=1)
        self.b.state.sv_batch_mirf = tk.StringVar(value=str(_C()['MACHINE_IRF_DEFAULT_PATH']))
        _row(fi, 'Machine IRF (.npy) *', self.b.sv_batch_mirf, 0,
             lambda: _browse_file(self.b.sv_batch_mirf, 'Machine IRF',
                                  [('NumPy', '*.npy'), ('All', '*.*')]))

        fp = _section(tab, 'Fitting Parameters')
        fp.grid(row=3, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(fp, text='Exponential components:').grid(row=0, column=0, sticky='w', **PAD)
        self.b.state.iv_nexp_batch = tk.IntVar(value=2)
        for n in (1, 2, 3):
            ttk.Radiobutton(fp, text=str(n), variable=self.b.iv_nexp_batch,
                            value=n).grid(row=0, column=n, sticky='w', padx=4)
        ttk.Label(fp, text='Fit window (ns):').grid(row=1, column=0, sticky='w', **PAD)
        self.b.state.sv_batch_tau_min = tk.StringVar(value=str(_C()['Tau_min']))
        self.b.state.sv_batch_tau_max = tk.StringVar(value=str(_C()['Tau_max']))
        ttk.Entry(fp, textvariable=self.b.sv_batch_tau_min, width=7).grid(row=1, column=1, padx=4)
        ttk.Label(fp, text='to').grid(row=1, column=2)
        ttk.Entry(fp, textvariable=self.b.sv_batch_tau_max, width=7).grid(row=1, column=3, padx=4)
        ttk.Label(fp, text='ns', foreground='grey').grid(row=1, column=4, padx=4)
        ttk.Label(fp, text='Colour scale (ns):').grid(row=2, column=0, sticky='w', **PAD)
        self.b.state.sv_batch_tau_lo = tk.StringVar(
            value='' if _C()['TAU_DISPLAY_MIN'] is None else str(_C()['TAU_DISPLAY_MIN']))
        self.b.state.sv_batch_tau_hi = tk.StringVar(
            value='' if _C()['TAU_DISPLAY_MAX'] is None else str(_C()['TAU_DISPLAY_MAX']))
        ttk.Entry(fp, textvariable=self.b.sv_batch_tau_lo, width=7).grid(row=2, column=1, padx=4)
        ttk.Label(fp, text='to').grid(row=2, column=2)
        ttk.Entry(fp, textvariable=self.b.sv_batch_tau_hi, width=7).grid(row=2, column=3, padx=4)
        ttk.Label(fp, text='ns  (display only)', foreground='grey').grid(row=2, column=4, padx=4)

        freg = _section(tab, 'Tile Registration')
        freg.grid(row=4, column=0, sticky='ew', pady=(0, 6))
        self.b._batch_freg = freg  # reference for show/hide on mode change
        self.b.state.bv_batch_register = tk.BooleanVar(value=True)
        ttk.Checkbutton(freg, text='Phase-correlation registration (fixes stage Y/X drift)',
                        variable=self.b.bv_batch_register).grid(
            row=0, column=0, columnspan=3, sticky='w', **PAD)
        ttk.Label(freg, text='Max shift (px):').grid(row=1, column=0, sticky='w', **PAD)
        self.b.state.sv_batch_reg_shift = tk.StringVar(value='120')
        ttk.Entry(freg, textvariable=self.b.sv_batch_reg_shift, width=6).grid(
            row=1, column=1, sticky='w', padx=4)
        ttk.Label(freg, text='(increase if drift > 120px)',
                  foreground='grey').grid(row=1, column=2, sticky='w')

        fm = _section(tab, 'Masking')
        fm.grid(row=5, column=0, sticky='ew', pady=(0, 6))
        self.b.state.bv_batch_thr = tk.BooleanVar(value=False)
        self.b.state.sv_batch_thr = tk.StringVar()
        ttk.Checkbutton(fm, text='Intensity threshold (min photons/px):',
                        variable=self.b.bv_batch_thr,
                        command=lambda: _tog(self.b.bv_batch_thr, self.b._batch_thr_e)).grid(
            row=0, column=0, sticky='w', **PAD)
        self.b._batch_thr_e = ttk.Entry(fm, textvariable=self.b.sv_batch_thr,
                                      width=8, state='disabled')
        self.b._batch_thr_e.grid(row=0, column=1, sticky='w', padx=4)
        self.b.state.bv_batch_correct_pileup = tk.BooleanVar(value=False)
        ttk.Checkbutton(fm, text='Apply Coates pile-up correction (recommended if pile-up > 5%)',
                        variable=self.b.bv_batch_correct_pileup).grid(
            row=1, column=0, columnspan=3, sticky='w', **PAD)

        fexp = _section(tab, 'Image Export')
        fexp.grid(row=6, column=0, sticky='ew', pady=(0, 6))
        self.b.state.bv_batch_save_lifetime  = tk.BooleanVar(value=True)
        self.b.state.bv_batch_save_rgb       = tk.BooleanVar(value=True)
        self.b.state.bv_batch_save_intensity = tk.BooleanVar(value=True)
        self.b.state.bv_batch_save_npy       = tk.BooleanVar(value=True)
        self.b.state.bv_batch_save_ind       = tk.BooleanVar(value=False)
        ttk.Checkbutton(fexp, text='Lifetime image (uint16 TIFF)',
                        variable=self.b.bv_batch_save_lifetime).grid(row=0, column=0, sticky='w', **PAD)
        ttk.Checkbutton(fexp, text='Component RGB TIFF',
                        variable=self.b.bv_batch_save_rgb).grid(row=0, column=1, sticky='w', **PAD)
        ttk.Checkbutton(fexp, text='Intensity TIFF',
                        variable=self.b.bv_batch_save_intensity).grid(row=0, column=2, sticky='w', **PAD)
        ttk.Checkbutton(fexp, text='Raw maps (.npy)',
                        variable=self.b.bv_batch_save_npy).grid(row=1, column=0, sticky='w', **PAD)
        ttk.Checkbutton(fexp, text='Individual component maps (τ₁, a₁, τ₂...)',
                        variable=self.b.bv_batch_save_ind).grid(row=1, column=1, columnspan=2, sticky='w', **PAD)
        ttk.Label(fexp, text='Lifetime colour scale (ns):').grid(row=2, column=0, sticky='w', **PAD)
        ttk.Entry(fexp, textvariable=self.b.sv_batch_tau_lo, width=7).grid(row=2, column=1, sticky='w', padx=4)
        ttk.Label(fexp, text='to').grid(row=2, column=2)
        ttk.Entry(fexp, textvariable=self.b.sv_batch_tau_hi, width=7).grid(row=2, column=3, sticky='w', padx=4)
        ttk.Label(fexp, text='ns  (blank = auto)', foreground='grey').grid(row=2, column=4, sticky='w', padx=4)
        ttk.Label(fexp, text='Gamma (lifetime image):').grid(row=3, column=0, sticky='w', **PAD)
        self.b.state.sv_batch_gamma = tk.StringVar(value='0.4')
        ttk.Entry(fexp, textvariable=self.b.sv_batch_gamma, width=5).grid(row=3, column=1, sticky='w', padx=4)
        ttk.Label(fexp, text='(0.4 = boost dim tissue; 1.0 = linear)',
                  foreground='grey').grid(row=3, column=2, columnspan=3, sticky='w')
        ttk.Label(fexp, text='Intensity display max:').grid(row=4, column=0, sticky='w', **PAD)
        self.b.state.sv_batch_int_max = tk.StringVar()
        ttk.Entry(fexp, textvariable=self.b.sv_batch_int_max, width=8).grid(row=4, column=1, sticky='w', padx=4)
        ttk.Label(fexp, text='(blank = auto 99th percentile)',
                  foreground='grey').grid(row=4, column=2, columnspan=3, sticky='w')

        # Expert settings banner
        self.b._expert_banner_batch = ttk.Label(
            tab, text='⚙  Custom expert settings active',
            foreground='#e8a838', font=('TkDefaultFont', 9, 'bold'))
        self.b._expert_banner_batch.grid(row=7, column=0, sticky='w', padx=8)
        self.b._expert_banner_batch.grid_remove()

        # Bottom row: Expert Settings + Run button
        btn_row_batch = ttk.Frame(tab)
        btn_row_batch.grid(row=8, column=0, pady=8)
        ttk.Button(btn_row_batch, text='⚙  Expert Settings',
                   command=self.b._open_expert_settings).pack(side='left', padx=4)
        self.b._btn_batch = ttk.Button(btn_row_batch, text='▶  Run Batch ROI Fit',
                                     command=self.b._dispatch_batch)
        self.b._btn_batch.pack(side='left', padx=4, ipadx=20, ipady=4)
