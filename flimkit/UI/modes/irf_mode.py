import tkinter as tk
from tkinter import ttk

from flimkit.UI.modes.base import BaseMode
from flimkit.UI.utils import PAD, _C, _section, _row, _browse_dir


class IrfMode(BaseMode):
    def build(self):
        outer, tab = self.b._form_inner_frames['irf']
        tab.columnconfigure(0, weight=1)

        cfg = _C()

        ff = _section(tab, 'Source Data')
        ff.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ff.columnconfigure(1, weight=1)
        self.b.state.sv_mirf_src = tk.StringVar()
        _row(
            ff,
            'PTU/XLSX folder *',
            self.b.sv_mirf_src,
            0,
            lambda: _browse_dir(self.b.sv_mirf_src, 'Folder with paired .ptu and .xlsx'),
        )
        ttk.Label(
            ff,
            text='Builder uses matching <name>.ptu + <name>.xlsx pairs.',
            foreground='grey',
        ).grid(row=1, column=1, columnspan=2, sticky='w', padx=4)

        fp = _section(tab, 'Build Settings')
        fp.grid(row=1, column=0, sticky='ew', pady=(0, 6))
        fp.columnconfigure(1, weight=1)

        self.b.state.sv_mirf_anchor = tk.StringVar(value=cfg['MACHINE_IRF_ALIGN_ANCHOR'])
        self.b.state.sv_mirf_reducer = tk.StringVar(value=cfg['MACHINE_IRF_REDUCER'])

        ttk.Label(fp, text='Align anchor:').grid(row=0, column=0, sticky='w', **PAD)
        ttk.Combobox(
            fp,
            textvariable=self.b.sv_mirf_anchor,
            values=['peak', 'halfmax', 'onset10', 'slope'],
            state='readonly',
            width=12,
        ).grid(row=0, column=1, sticky='w', padx=4)

        ttk.Label(fp, text='Reducer:').grid(row=1, column=0, sticky='w', **PAD)
        ttk.Combobox(
            fp,
            textvariable=self.b.sv_mirf_reducer,
            values=['median', 'mean'],
            state='readonly',
            width=12,
        ).grid(row=1, column=1, sticky='w', padx=4)

        fo = _section(tab, 'Output')
        fo.grid(row=2, column=0, sticky='ew', pady=(0, 6))
        fo.columnconfigure(1, weight=1)
        self.b.state.sv_mirf_out_dir = tk.StringVar(value=str(cfg['MACHINE_IRF_DIR']))
        self.b.state.sv_mirf_name = tk.StringVar(value='machine_irf_default')

        _row(
            fo,
            'Output directory *',
            self.b.sv_mirf_out_dir,
            0,
            lambda: _browse_dir(self.b.sv_mirf_out_dir, 'Machine IRF output directory'),
        )
        ttk.Label(fo, text='Base filename:').grid(row=1, column=0, sticky='w', **PAD)
        ttk.Entry(fo, textvariable=self.b.sv_mirf_name, width=35).grid(
            row=1, column=1, columnspan=2, sticky='ew', padx=4
        )

        self.b._btn_mirf = ttk.Button(
            tab,
            text='▶  Build Machine IRF',
            command=self.b._run_build_machine_irf,
        )
        self.b._btn_mirf.grid(row=3, column=0, pady=8, ipadx=20, ipady=4)
