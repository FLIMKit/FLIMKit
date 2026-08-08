from __future__ import annotations
import re
import sys
import time
import inspect
import threading
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import os
import sys
from pathlib import Path
from typing import Optional
import matplotlib
from flimkit.mpl_backend import select_backend
select_backend()
matplotlib.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.titlecolor': 'white',
})
import matplotlib.image as mpimg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from flimkit.UI.progress_window import ProgressWindow
from flimkit.UI.phasor_panel import PhasorViewPanel
from flimkit.UI import flim_display
from flimkit.UI.roi_tools import RoiManager, RoiAnalysisPanel
from flimkit.UI.project_panel import ProjectBrowserPanel

try:
    import TKinterModernThemes as TKMT
    HAS_TKMT = True
except ImportError:
    HAS_TKMT = False

try:
    from tkinterdnd2 import DND_FILES, DND_TEXT
    HAS_DND = True
except ImportError:
    HAS_DND = False

GUI_MODE = False

from flimkit.UI.utils import (
    PAD,
    _C,
    _reconstruct_dict_from_session,
    _safe_array_from_json,
    _parse_summary,
    _Redirect,
    _FileRedirect,
    _FileTailer,
    _browse_file,
    _browse_dir,
    _row,
    _section,
    _tog,
    _flt,
    _thresh,
    _enable_dnd,
)
from flimkit.UI.progress_window import ProgressWindowManager
from flimkit.UI.irf_widget import IRFWidget
from flimkit.UI.expert_settings import ExpertSettingsDialog, _EXPERT_DEFAULTS
from flimkit.UI.fov_preview import FOVPreviewPanel
from flimkit.UI.results_panel import ResultsPanel
from flimkit.UI.app_state import AppState
from flimkit.UI.mode_controller import ModeController
from flimkit.UI.controller import FLIMKitController

class _UIBuilder:

    def __getattr__(self, nam):
        state = self.__dict__.get('state')
        if state is not None and nam in state.__dict__:
            return state.__dict__[nam]
        raise AttributeError(nam)

    def _make_scroll_frame(self, parent: ttk.Frame) -> tuple:
        outer = ttk.Frame(parent)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        vbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        inner = ttk.Frame(canvas, padding=10)
        window_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_inner_configure(_evt=None):
            canvas.configure(scrollregion=canvas.bbox('all'))
            inner_height = inner.winfo_reqheight()
            canvas_height = canvas.winfo_height()
            if canvas_height > 1:
                canvas.itemconfigure(window_id, height=max(inner_height, canvas_height))
            canvas.itemconfigure(window_id, width=canvas.winfo_width() if canvas.winfo_width() > 1 else None)

        def _on_canvas_configure(evt):
            if evt.width > 1:
                canvas.itemconfigure(window_id, width=evt.width)

        def _on_mousewheel(evt):
            if evt.num == 5 or evt.delta < 0:
                canvas.yview_scroll(3, 'units')
            elif evt.num == 4 or evt.delta > 0:
                canvas.yview_scroll(-3, 'units')
        inner.bind('<Configure>', _on_inner_configure)
        canvas.bind('<Configure>', _on_canvas_configure)
        canvas.bind('<MouseWheel>', _on_mousewheel)
        canvas.bind('<Button-4>', _on_mousewheel)
        canvas.bind('<Button-5>', _on_mousewheel)
        inner.bind('<MouseWheel>', _on_mousewheel)
        inner.bind('<Button-4>', _on_mousewheel)
        inner.bind('<Button-5>', _on_mousewheel)
        inner._canvas = canvas
        inner._window_id = window_id
        outer._canvas = canvas
        outer._window_id = window_id
        return outer, inner

    def _fit_window_to_screen(self):
        self.root.update_idletasks()
        sw = max(1, int(self.root.winfo_screenwidth()))
        sh = max(1, int(self.root.winfo_screenheight()))
        max_w = max(1200, sw - 40)
        max_h = max(700, sh - 40)
        req_w = int(self.root.winfo_reqwidth())
        req_h = int(self.root.winfo_reqheight())
        cur_w = int(self.root.winfo_width())
        cur_h = int(self.root.winfo_height())
        target_w = min(max(cur_w, req_w, 1200), max_w)
        target_h = min(max(cur_h, req_h, 800), max_h)
        x = int(self.root.winfo_x())
        y = int(self.root.winfo_y())
        x = min(max(0, x), max(0, sw - target_w))
        y = min(max(0, y), max(0, sh - target_h))
        self.root.maxsize(sw, sh)
        self.root.geometry(f'{target_w}x{target_h}+{x}+{y}')

    def _set_pane_positions(self):
        try:
            self.root.update_idletasks()
            main_width = self._main_paned.winfo_width()
            if main_width < 100:
                return
            left_width = int(main_width * 0.6)
            self._main_paned.sashpos(0, left_width)
            project_width = int(main_width * 0.15)
            self._left_paned.sashpos(0, project_width)
        except Exception as e:
            print(f'[Layout] Could not set pane positions: {e}')

    def run_with_progress(self, task_fn, task_name='Working...', on_done=None, output_dir=None):
        from flimkit.utils.crash_handler import log_event
        log_event(f'Task started: {task_name}')
        win = ProgressWindow(self.root, task_name=task_name)
        cancel_event = win.cancelled

        def progress_callback(i, total):
            win.set_progress(i, maximum=total)
            if cancel_event.is_set():
                win.set_status('Cancelling...')

        def worker():
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            redir = _Redirect(self._res.log, self._buf, root=self.root)
            redir_err = _Redirect(self._res.log, self._buf, root=self.root, is_stderr=True)
            sys.stdout = redir
            sys.stderr = redir_err
            try:
                result = task_fn(progress_callback, cancel_event)
                self.root.after(0, lambda: win.close())
                if on_done:
                    self.root.after(0, lambda: on_done(result))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                from flimkit.utils.crash_handler import log_exception
                log_exception(f'run_with_progress: {task_name}')
                self.root.after(0, lambda e=exc: win.set_status(f'Error: {e}'))
                self.root.after(0, lambda: win.btn_cancel.config(text='Close', command=win.close))
            finally:
                if hasattr(redir, 'close'):
                    redir.close()
                else:
                    redir.flush()
                if hasattr(redir_err, 'close'):
                    redir_err.close()
                else:
                    redir_err.flush()
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
        threading.Thread(target=worker, daemon=True).start()

    def _build_menu_bar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Restore NPZ...', command=self._menu_restore_npz)
        file_menu.add_command(label='Save NPZ', command=self._menu_save_npz)
        file_menu.add_command(label='Save NPZ As...', command=self._menu_save_npz_as)
        file_menu.add_separator()
        file_menu.add_command(label='Open Project Folder...', command=self._menu_open_project_folder)
        file_menu.add_separator()
        export_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label='Export', menu=export_menu)
        export_menu.add_command(label='Export Summed Fit CSV', command=self._menu_export_fit_csv)
        export_menu.add_command(label='Export ROI Table CSV', command=self._menu_export_roi_csv)
        export_menu.add_command(label='Export ROI as GeoJSON', command=self._menu_export_roi_geojson)
        export_menu.add_command(label='Export All ROIs as GeoJSON', command=self._menu_export_all_rois_geojson)
        file_menu.add_command(label='Import GeoJSON...', command=self._menu_import_geojson)
        file_menu.add_separator()
        self._recent_files_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label='Recent Files', menu=self._recent_files_menu)
        self._recent_files = self._load_recent_list()
        self._update_recent_files_menu()
        file_menu.add_separator()
        file_menu.add_command(label='Preferences...', command=self._menu_preferences)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.root.quit)
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Edit', menu=edit_menu)
        edit_menu.add_command(label='Undo', command=self._menu_undo, accelerator='Ctrl+Z')
        edit_menu.add_command(label='Redo', command=self._menu_redo, accelerator='Ctrl+Shift+Z')
        edit_menu.add_separator()
        edit_menu.add_command(label='Reset', command=self._menu_reset)
        self.root.bind('<Control-z>', lambda e: self._menu_undo())
        self.root.bind('<Control-Shift-Z>', lambda e: self._menu_redo())
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Tools', menu=tools_menu)
        tools_menu.add_command(label='Machine IRF Builder', command=self._menu_irf_builder)
        tools_menu.add_command(label='Time-Resolved Anisotropy...',
                               command=self._menu_anisotropy)
        tools_menu.add_command(label='Generate Synthetic PTU...', command=self._menu_synth_generator)
        batch_menu = tk.Menu(tools_menu, tearoff=0)
        tools_menu.add_cascade(label='Batch Processing', menu=batch_menu)
        batch_menu.add_command(label='Multi-Tile ROI Fit',
                               command=lambda: self._menu_batch_processing('tiled'))
        batch_menu.add_command(label='Single FOV Fit',
                               command=lambda: self._menu_batch_processing('fov'))
        batch_menu.add_command(label='Timelapse Fit',
                               command=lambda: self._menu_batch_processing('timelapse'))
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Help', menu=help_menu)
        help_menu.add_command(label='About', command=self._menu_about)
        help_menu.add_command(label='Documentation', command=self._menu_documentation)
        help_menu.add_command(label='Check for Updates', command=self._menu_check_updates)
        help_menu.add_separator()
        help_menu.add_command(label='View Error Logs', command=self._menu_view_error_logs)
        help_menu.add_command(label='Export Error Logs', command=self._menu_export_error_logs)
    _RECENT_FILE = os.path.join(os.path.expanduser('~'), '.flimkit', 'recent.json')
    _MAX_RECENT = 10

    def _load_recent_list(self):
        try:
            with open(self._RECENT_FILE, 'r') as f:
                import json
                data = json.load(f)
            return [e for e in data if isinstance(e, dict) and 'path' in e and 'type' in e]
        except (FileNotFoundError, ValueError):
            return []

    def _save_recent_list(self):
        import json
        os.makedirs(os.path.dirname(self._RECENT_FILE), exist_ok=True)
        with open(self._RECENT_FILE, 'w') as f:
            json.dump(self._recent_files, f, indent=2)

    def _update_recent_files_menu(self):
        self._recent_files_menu.delete(0, tk.END)
        if self._recent_files:
            for entry in self._recent_files:
                path = entry['path']
                kind = entry.get('type', 'file')
                prefix = '[Project] ' if kind == 'project' else ''
                label = f'{prefix}{path}'
                self._recent_files_menu.add_command(
                    label=label,
                    command=lambda e=entry: self._load_recent_item(e)
                )
            self._recent_files_menu.add_separator()
            self._recent_files_menu.add_command(label='Clear Recent', command=self._clear_recent_files)
        else:
            self._recent_files_menu.add_command(label='(No recent items)', state='disabled')

    def _load_recent_item(self, entry):
        path = entry['path']
        kind = entry.get('type', 'file')
        if kind == 'project':
            if os.path.isdir(path):
                print(f'[Menu] Opening recent project: {path}')
                if hasattr(self, '_proj_browser') and self._proj_browser:
                    self._proj_browser.load_folder(path)
                self._add_to_recent(path, 'project')
            else:
                print(f'[Menu] Project folder not found: {path}')
        else:
            if os.path.isfile(path):
                print(f'[Menu] Loading recent file: {path}')
                self.sv_ptu.set(path)
                self._add_to_recent(path, 'file')
            else:
                print(f'[Menu] File not found: {path}')

    def _add_to_recent(self, filepath, kind='file'):
        path_str = str(filepath)
        self._recent_files = [e for e in self._recent_files if e['path'] != path_str]
        self._recent_files.insert(0, {'path': path_str, 'type': kind})
        if len(self._recent_files) > self._MAX_RECENT:
            self._recent_files = self._recent_files[:self._MAX_RECENT]
        self._update_recent_files_menu()
        self._save_recent_list()

    def _clear_recent_files(self):
        self._recent_files = []
        self._update_recent_files_menu()
        self._save_recent_list()

    def _current_scan_stem(self) -> str:
        if hasattr(self, 'sv_ptu'):
            p = self.sv_ptu.get().strip()
            if p:
                return Path(p).stem
        return ''

    def _menu_restore_npz(self):
        if self._res and hasattr(self._res, '_load_fitted_data'):
            self._res._load_fitted_data()

    def _menu_save_npz(self):
        if self._res and hasattr(self._res, '_on_save_npz_clicked'):
            self._res._on_save_npz_clicked()

    def _menu_save_npz_as(self):
        import shutil
        from tkinter import filedialog, messagebox
        scan = self._current_scan_stem()
        src = getattr(self._res, '_current_npz_path', None) if self._res else None
        if not src or not Path(src).exists():
            messagebox.showwarning('Save NPZ As', 'No session file to save. Run a fit first.')
            return
        npz_file = filedialog.asksaveasfilename(
            title='Save NPZ As',
            initialfile=f'{scan}.roi_session.npz' if scan else '',
            defaultextension='.npz',
            filetypes=[('NPZ files', '*.npz'), ('All files', '*.*')])
        if npz_file:
            try:
                shutil.copy2(src, npz_file)
                self._res._current_npz_path = npz_file
                messagebox.showinfo('Saved', f'Session saved to:\n{Path(npz_file).name}')
            except Exception as e:
                messagebox.showerror('Save Error', f'Could not save: {e}')

    def _menu_open_project_folder(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title='Select Project Folder')
        if folder:
            print(f'[Menu] Opening project folder: {folder}')
            if hasattr(self, '_proj_browser') and self._proj_browser:
                self._proj_browser.load_folder(folder)
            self._add_to_recent(folder, 'project')

    def _menu_export_fit_csv(self):
        print('[Menu] Export Fit CSV')
        if hasattr(self, '_res') and self._res:
            self._res._export_summed_csv()

    def _menu_export_roi_csv(self):
        print('[Menu] Export ROI CSV')
        if self._roi_analysis_panel:
            self._roi_analysis_panel._export_all_rois_csv()

    def _menu_export_roi_geojson(self):
        print('[Menu] Export ROI GeoJSON')
        if self._roi_analysis_panel:
            self._roi_analysis_panel._export_selected_region()

    def _menu_export_all_rois_geojson(self):
        print('[Menu] Export All ROIs GeoJSON')
        if self._roi_analysis_panel:
            self._roi_analysis_panel._export_all_rois_geojson()

    def _menu_import_geojson(self):
        print('[Menu] Import GeoJSON')
        if self._roi_analysis_panel:
            self._roi_analysis_panel._import_rois_geojson()

    def _menu_preferences(self):
        from flimkit.utils.config_manager import cfg
        prefs = cfg.get_section('preferences')
        pref_win = tk.Toplevel(self.root)
        pref_win.title('Preferences')
        pref_win.geometry('500x400')
        pref_win.resizable(False, False)
        main_frame = ttk.Frame(pref_win, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text='FLIMKit Preferences', font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        note = ttk.Notebook(main_frame)
        note.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        disp_frame = ttk.Frame(note, padding=10)
        note.add(disp_frame, text='Display')
        ttk.Label(disp_frame, text='Colormap:', font=('TkDefaultFont', 10)).pack(anchor='w', pady=(5, 0))
        cmap_var = tk.StringVar(value=prefs.get('colormap', 'viridis'))
        ttk.Combobox(disp_frame, textvariable=cmap_var,
                     values=['viridis', 'plasma', 'gray', 'jet'], state='readonly').pack(anchor='w', pady=(0, 10))
        ttk.Label(disp_frame, text='Font Size:', font=('TkDefaultFont', 10)).pack(anchor='w', pady=(5, 0))
        font_var = tk.IntVar(value=prefs.get('font_size', 9))
        ttk.Spinbox(disp_frame, from_=8, to=14, textvariable=font_var, width=10).pack(anchor='w', pady=(0, 10))
        anal_frame = ttk.Frame(note, padding=10)
        note.add(anal_frame, text='Analysis')
        ttk.Label(anal_frame, text='Default Number of Exponents:', font=('TkDefaultFont', 10)).pack(anchor='w', pady=(5, 0))
        exp_var = tk.IntVar(value=prefs.get('default_nexp', 2))
        ttk.Spinbox(anal_frame, from_=1, to=5, textvariable=exp_var, width=10).pack(anchor='w', pady=(0, 10))
        ttk.Label(anal_frame, text='Export Format:', font=('TkDefaultFont', 10)).pack(anchor='w', pady=(5, 0))
        fmt_var = tk.StringVar(value=prefs.get('export_format', 'CSV'))
        ttk.Combobox(anal_frame, textvariable=fmt_var,
                     values=['CSV', 'Excel', 'NumPy'], state='readonly').pack(anchor='w', pady=(0, 10))
        files_frame = ttk.Frame(note, padding=10)
        note.add(files_frame, text='Files')
        ttk.Label(files_frame, text='Output Directory:', font=('TkDefaultFont', 10)).pack(anchor='w', pady=(5, 0))
        saved_outdir = prefs.get('output_directory', '') or os.path.expanduser('~/FLIMKit/output')
        output_var = tk.StringVar(value=saved_outdir)
        ttk.Entry(files_frame, textvariable=output_var, width=40).pack(anchor='w', pady=(0, 5))
        ttk.Button(files_frame, text='Browse...', width=10,
                   command=lambda: output_var.set(filedialog.askdirectory())).pack(anchor='w', pady=(0, 10))
        ttk.Label(files_frame, text='Auto-save NPZ:', font=('TkDefaultFont', 10)).pack(anchor='w', pady=(5, 0))
        autosave_var = tk.BooleanVar(value=prefs.get('auto_save_npz', True))
        ttk.Checkbutton(files_frame, text='Enable auto-save', variable=autosave_var).pack(anchor='w', pady=(0, 10))
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 0))

        def save_prefs():
            cfg.update_section('preferences', {
                'colormap': cmap_var.get(),
                'font_size': font_var.get(),
                'default_nexp': exp_var.get(),
                'export_format': fmt_var.get(),
                'output_directory': output_var.get(),
                'auto_save_npz': autosave_var.get(),
            })
            print(f'[Preferences] Saved to {cfg._CONFIG_FILE if hasattr(cfg, '_CONFIG_FILE') else '~/.flimkit/config.yaml'}')
            pref_win.destroy()
        ttk.Button(btn_frame, text='Save', command=save_prefs).pack(side='right', padx=5)
        ttk.Button(btn_frame, text='Cancel', command=pref_win.destroy).pack(side='right', padx=5)

    def _menu_undo(self):
        print('[Menu] Undo')

    def _menu_redo(self):
        print('[Menu] Redo')

    def _menu_reset(self):
        from tkinter import messagebox
        if messagebox.askyesno('Reset', 'Clear all regions and results? This cannot be undone.'):
            print('[Menu] Resetting analysis...')
            if self._roi_analysis_panel and self._roi_analysis_panel.fov_preview:
                roi_manager = self._roi_analysis_panel.fov_preview._roi_manager
                region_ids = [r['id'] for r in roi_manager.get_all_regions()]
                for region_id in region_ids:
                    roi_manager.remove_region(region_id)
                self._roi_analysis_panel._refresh_region_list()
                self._roi_analysis_panel.fov_preview._redraw_region_overlays()
            if self._res:
                self._res._tv.delete(*self._res._tv.get_children())
            messagebox.showinfo('Reset Complete', 'Analysis cleared.')
            print('[Menu] Reset complete.')

    def _menu_irf_builder(self):
        print('[Menu] Machine IRF Builder')
        if hasattr(self, '_switch_form'):
            self._switch_form('irf')

    def _menu_anisotropy(self):
        print('[Menu] Time-Resolved Anisotropy')
        from flimkit.UI.anisotropy_tool import show_anisotropy_tool
        show_anisotropy_tool(self.root)

    def _menu_synth_generator(self):
        print('[Menu] Generate Synthetic PTU')
        from flimkit import synth
        dlg = tk.Toplevel(self.root)
        dlg.title('Generate Synthetic PTU')
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        fields = [
            ('Lifetime(s) τ, ns (comma for multi-exp)', 'tau', '4.1'),
            ('Amplitudes (comma, blank = equal)', 'amps', ''),
            ('Photons (comma = a series)', 'photons', '1e5'),
            ('Laser period, ns', 'period', '50'),
            ('TCSPC bin width, ps', 'res', '25'),
            ('IRF FWHM, ns', 'irf_fwhm', '0.15'),
            ('IRF centre, ns', 'irf_center', '2.0'),
            ('Reflection at, ns (blank = none)', 'refl_ns', ''),
            ('Reflection fraction', 'refl_frac', '0.02'),
            ('Pile-up, photons/pulse (blank = none)', 'pileup', ''),
            ('Image side, px', 'image', '16'),
            ('Base name', 'name', 'synth'),
        ]
        vals = {}
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill='both', expand=True)
        for i, (label, key, default) in enumerate(fields):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky='w', pady=2, padx=(0, 8))
            v = tk.StringVar(value=default)
            ttk.Entry(frm, textvariable=v, width=22).grid(row=i, column=1, sticky='ew', pady=2)
            vals[key] = v
        also_sdt = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='Also write Becker & Hickl .sdt', variable=also_sdt).grid(
            row=len(fields), column=0, columnspan=2, sticky='w', pady=(6, 0))
        status = tk.StringVar(value='')
        ttk.Label(frm, textvariable=status, foreground='grey').grid(
            row=len(fields) + 1, column=0, columnspan=2, sticky='w', pady=(8, 0))

        def _floats(s):
            return [float(x) for x in str(s).split(',') if x.strip()]

        def do_generate():
            from tkinter import filedialog
            out_dir = filedialog.askdirectory(title='Choose output folder for the PTUs')
            if not out_dir:
                return
            try:
                taus = _floats(vals['tau'].get())
                tau_arg = taus[0] if len(taus) == 1 else taus
                amps = _floats(vals['amps'].get()) or None
                res_ns = float(vals['res'].get()) / 1000.0
                n_bins = int(round(float(vals['period'].get()) / res_ns))
                refl = None
                if vals['refl_ns'].get().strip():
                    refl = dict(center_ns=float(vals['refl_ns'].get()),
                                frac=float(vals['refl_frac'].get()), width_ns=0.15)
                pileup = float(vals['pileup'].get()) if vals['pileup'].get().strip() else None
                side = int(vals['image'].get())
                common = dict(tau_ns=tau_arg, amps=amps, n_bins=n_bins, tcspc_res_ns=res_ns,
                              irf_fwhm_ns=float(vals['irf_fwhm'].get()),
                              irf_center_ns=float(vals['irf_center'].get()), pileup_pp=pileup)
                photons = _floats(vals['photons'].get())
                if len(photons) == 1:
                    synth.generate(out_dir, name=vals['name'].get(), ny=side, nx=side,
                                   n_photons=photons[0], reflection=refl,
                                   sdt=also_sdt.get(), **common)
                    n_files = 1
                else:
                    synth.generate_series(out_dir, photons, name=vals['name'].get(),
                                          with_reflection=refl is not None, reflection=refl,
                                          ny=side, nx=side, sdt=also_sdt.get(), **common)
                    n_files = len(photons)
                fmt = 'PTU + SDT' if also_sdt.get() else 'PTU'
                messagebox.showinfo('Synthetic data',
                                    f'Wrote {n_files} sample(s) ({fmt}) + IRF + truth JSON to\n{out_dir}')
                dlg.destroy()
            except Exception as e:
                import traceback
                traceback.print_exc()
                status.set(f'Error: {e}')

        btns = ttk.Frame(dlg)
        btns.pack(fill='x', padx=12, pady=(0, 12))
        ttk.Button(btns, text='Generate...', command=do_generate).pack(side='left', padx=4)
        ttk.Button(btns, text='Cancel', command=dlg.destroy).pack(side='left', padx=4)
        dlg.update_idletasks()

    def _menu_batch_processing(self, mode: str = 'tiled'):
        print(f'[Menu] Batch Processing → {mode}')
        self.sv_batch_mode.set(mode)
        self._batch_mode_changed()
        if hasattr(self, '_switch_form'):
            self._switch_form('batch')

    def _menu_about(self):
        from flimkit._version import __version__
        about_text = f'''FLIMKit Analysis GUI

Version: {__version__}

A comprehensive FLIM data analysis platform with:
• Single FOV & tile stitching
• ROI-based lifetime analysis
• Machine IRF calibration
• Batch processing
• GeoJSON & CSV export

Built with Python, Tkinter, NumPy, and SciPy.

Designed, developed, and maintained by Alex Hunt.
Anthropic's Claude AI assisted with parts of the GUI implementation.
        '''
        messagebox.showinfo('About FLIMKit', about_text)

    def _menu_documentation(self):
        import webbrowser
        import os
        doc_file = os.path.join(os.path.dirname(__file__), '../../documentation.md')
        if os.path.exists(doc_file):
            webbrowser.open('file://' + os.path.realpath(doc_file))
        else:
            messagebox.showinfo('Documentation', 'See README.md in the project root for documentation.')

    def _menu_check_updates(self):
        status_win = tk.Toplevel(self.root)
        status_win.title('Checking for Updates')
        status_win.geometry('380x120')
        status_win.resizable(False, False)
        ttk.Label(
            status_win,
            text='Checking git status and latest available release...',
            wraplength=340,
            justify='left',
        ).pack(padx=16, pady=(18, 10), anchor='w')
        pb = ttk.Progressbar(status_win, mode='indeterminate')
        pb.pack(fill='x', padx=16, pady=(0, 14))
        pb.start(12)

        def _worker():
            try:
                from flimkit.utils.update_check import (
                    check_installation_freshness,
                    format_update_report,
                )
                report = format_update_report(
                    check_installation_freshness(timeout=3.0, do_fetch=True)
                )
                err = None
            except Exception as exc:
                report = ''
                err = str(exc)

            def _done():
                try:
                    pb.stop()
                    status_win.destroy()
                except Exception:
                    pass
                if err is not None:
                    messagebox.showerror('Update Check', f'Update check failed: {err}')
                    return
                win = tk.Toplevel(self.root)
                win.title('Update Check')
                win.geometry('760x420')
                text_widget = scrolledtext.ScrolledText(win, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
                text_widget.insert(tk.END, report)
                text_widget.config(state=tk.DISABLED)
            self.root.after(0, _done)
        threading.Thread(target=_worker, daemon=True).start()

    def _menu_view_error_logs(self):
        from flimkit.utils.crash_handler import build_export_report, get_log_dir
        import glob
        log_dir = get_log_dir()
        log_files = glob.glob(os.path.join(log_dir, '*.log')) if os.path.exists(log_dir) else []
        if log_files:
            try:
                report = build_export_report(include_all_sessions=False)
                from tkinter.scrolledtext import ScrolledText
                win = tk.Toplevel(self.root)
                win.title('Error Logs - Current Session')
                win.geometry('700x500')
                text_widget = ScrolledText(win, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
                text_widget.insert(tk.END, report)
                text_widget.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror('Error', f'Could not read log file: {e}')
        else:
            messagebox.showinfo('No Logs', 'No error logs found.')

    def _menu_export_error_logs(self):
        from flimkit.utils.crash_handler import build_export_report, get_log_dir
        import glob
        log_dir = get_log_dir()
        log_files = glob.glob(os.path.join(log_dir, '*.log')) if os.path.exists(log_dir) else []
        if not log_files:
            messagebox.showwarning('No Logs', 'No error logs found to export.')
            return
        export_file = filedialog.asksaveasfilename(
            title='Save Error Report',
            defaultextension='.log',
            filetypes=[('Log files', '*.log'), ('Text files', '*.txt'), ('All files', '*.*')]
        )
        if export_file:
            try:
                report = build_export_report(include_all_sessions=True)
                with open(export_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(report)
                messagebox.showinfo('Export Success', f'Error report exported to:\n{export_file}')
                print(f'[Menu] Error report exported to: {export_file}')
            except Exception as e:
                messagebox.showerror('Export Error', f'Failed to export logs: {e}')

    def _find_scroll_canvas(self, widget):
        w = widget
        for _ in range(30):
            try:
                if hasattr(w, '_canvas'):
                    return w._canvas
                w = w.master
                if w is None:
                    break
            except Exception:
                break
        return None

    def _setup_global_scroll(self):
        def _scroll(evt):
            try:
                widget = self.root.winfo_containing(evt.x_root, evt.y_root)
                if widget is None:
                    return
                canvas = self._find_scroll_canvas(widget)
                if canvas is None:
                    return
                sr = canvas.cget('scrollregion')
                if not sr:
                    return
                delta = evt.delta if hasattr(evt, 'delta') else 0
                if evt.num == 5 or delta < 0:
                    canvas.yview_scroll(3, 'units')
                elif evt.num == 4 or delta > 0:
                    canvas.yview_scroll(-3, 'units')
            except Exception:
                pass
        for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.root.bind_all(seq, _scroll, add='+')

    def _setup_global_dnd(self):
        if not HAS_DND:
            return
        try:
            from tkinterdnd2 import DND_FILES, DND_TEXT
        except ImportError:
            return

        def _clean(data: str) -> str:
            data = data.strip()
            if data.startswith('{'):
                end = data.find('}')
                if end != -1:
                    return data[1:end].strip()
            parts = data.split()
            if parts:
                return parts[0]
            return data

        def _register(widget):
            try:
                widget.drop_target_register(DND_FILES, DND_TEXT)
                tv_name = widget.cget('textvariable')
                if tv_name:
                    def _drop(evt, w=widget):
                        path = _clean(evt.data)
                        try:
                            w.delete(0, 'end')
                            w.insert(0, path)
                            w.event_generate('<<Modified>>')
                        except Exception:
                            pass
                    widget.dnd_bind('<<Drop>>', _drop)
            except Exception:
                pass

        def _walk(widget):
            try:
                cls = widget.winfo_class()
                if cls in ('Entry', 'TEntry'):
                    _register(widget)
                for child in widget.winfo_children():
                    _walk(child)
            except Exception:
                pass
        _walk(self.root)
        print('[DnD] Drop targets registered on all Entry widgets')

    def _init_ui(self):
        from flimkit.utils.crash_handler import install_tk_error_handler
        install_tk_error_handler(self.root)
        _enable_dnd(self.root)
        self.state = AppState()
        self._mode_controller = ModeController(self)
        self._controller = FLIMKitController(self)
        self._buf: list = []
        self._current_session_file = None
        self._current_npz_path = None
        self._last_loaded_ptu = None
        self._last_loaded_xlif = None
        self._ptu_after_id = None
        self._xlif_after_id = None
        self._form_buttons = {}
        self._form_frames = {}
        self._form_inner_frames = {}
        self._build_menu_bar()
        self._mode_toolbar = ttk.Frame(self.root)
        self._mode_toolbar.grid(row=0, column=0, sticky='ew', padx=10, pady=(2, 0))
        self._mode_toolbar.columnconfigure(0, weight=1)
        ttk.Label(self._mode_toolbar, text='Mode:', font=('TkDefaultFont', 9, 'bold')).pack(side='left', padx=(0, 10))
        self.state.current_mode = tk.StringVar(value='fov')
        btn_fov = ttk.Button(self._mode_toolbar, text='Single FOV Fit', width=16,
                             command=lambda: self._switch_form('fov'))
        btn_fov.pack(side='left', padx=2)
        self._form_buttons['fov'] = btn_fov
        btn_stitch = ttk.Button(self._mode_toolbar, text='Tile Stitch/Fit', width=16,
                                command=lambda: self._switch_form('stitch'))
        btn_stitch.pack(side='left', padx=2)
        self._form_buttons['stitch'] = btn_stitch
        btn_phasor = ttk.Button(self._mode_toolbar, text='Phasor Analysis', width=16,
                                command=lambda: self._switch_form('phasor'))
        btn_phasor.pack(side='left', padx=2)
        self._form_buttons['phasor'] = btn_phasor
        ttk.Separator(self._mode_toolbar, orient='vertical').pack(side='left', fill='y', padx=10, pady=2)
        self.state.mode_status = tk.StringVar(value='Current: Single FOV Fit')
        ttk.Label(self._mode_toolbar, textvariable=self.mode_status, foreground='grey').pack(side='left', padx=10)
        self._form_labels = {'fov': 'Single FOV Fit', 'stitch': 'Tile Stitch/Fit', 'phasor': 'Phasor Analysis'}
        ttk.Separator(self.root, orient='horizontal').grid(row=1, column=0, sticky='ew', pady=(2, 0))
        self._main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        self._main_paned.grid(row=2, column=0, sticky='nsew', padx=4, pady=(2, 4))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0, minsize=0)
        self.root.rowconfigure(1, weight=0, minsize=0)
        self.root.rowconfigure(2, weight=1)
        self._left_paned = ttk.PanedWindow(self._main_paned, orient='horizontal')
        self._main_paned.add(self._left_paned, weight=3)
        btn_frame = ttk.Frame(self._left_paned)
        self._left_paned.add(btn_frame, weight=1)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.rowconfigure(0, weight=1)
        self._proj_browser = ProjectBrowserPanel(btn_frame, app=self, width=170)
        self._proj_browser.frame.grid(row=0, column=0, sticky='nsew')
        content_paned = ttk.PanedWindow(self._left_paned, orient='vertical')
        self._left_paned.add(content_paned, weight=2)
        form_wrapper = ttk.Frame(content_paned)
        content_paned.add(form_wrapper, weight=3)
        form_wrapper.columnconfigure(0, weight=1)
        form_wrapper.rowconfigure(0, weight=1)
        self._analysis_tabs = ttk.Notebook(form_wrapper)
        self._analysis_tabs.grid(row=0, column=0, sticky='nsew')
        self._analysis_tabs.grid_remove()
        fit_settings_outer = ttk.Frame(self._analysis_tabs)
        self._analysis_tabs.add(fit_settings_outer, text='  Fit Settings  ')
        fit_settings_outer.columnconfigure(0, weight=1)
        fit_settings_outer.rowconfigure(0, weight=1)
        self._fit_settings_tab = fit_settings_outer
        roi_frame = ttk.Frame(self._analysis_tabs, padding=4)
        self._analysis_tabs.add(roi_frame, text='  ROI Analysis  ')
        roi_frame.columnconfigure(0, weight=1)
        roi_frame.rowconfigure(0, weight=1)
        self._roi_analysis_panel = RoiAnalysisPanel(roi_frame)
        self._roi_analysis_panel.grid(row=0, column=0, sticky='nsew')
        self._roi_analysis_frame = roi_frame
        self._stitch_tabs = ttk.Notebook(form_wrapper)
        self._stitch_tabs.grid(row=0, column=0, sticky='nsew')
        self._stitch_tabs.grid_remove()
        stitch_settings_outer = ttk.Frame(self._stitch_tabs)
        self._stitch_tabs.add(stitch_settings_outer, text='  Fit Settings  ')
        stitch_settings_outer.columnconfigure(0, weight=1)
        stitch_settings_outer.rowconfigure(0, weight=1)
        self._stitch_settings_tab = stitch_settings_outer
        stitch_roi_frame = ttk.Frame(self._stitch_tabs, padding=4)
        self._stitch_tabs.add(stitch_roi_frame, text='  ROI Analysis  ')
        stitch_roi_frame.columnconfigure(0, weight=1)
        stitch_roi_frame.rowconfigure(0, weight=1)
        self._stitch_roi_analysis_frame = stitch_roi_frame
        self._form_content_frame = form_wrapper
        for _fid in ('batch', 'irf'):
            _outer, _inner = self._make_scroll_frame(form_wrapper)
            _outer.grid(row=0, column=0, sticky='nsew')
            _outer.grid_remove()
            self._form_inner_frames[_fid] = (_outer, _inner)
            self._form_frames[_fid] = (_outer, _inner)
        form_list = [
            ('fov', 'Single FOV Fit'),
            ('stitch', 'Tile Stitch/Fit'),
            ('phasor', 'Phasor Analysis'),
            ('batch', 'Batch Processing'),
            ('irf', 'Machine IRF Builder'),
        ]
        for form_id, form_label in form_list:
            if form_id == 'fov':
                outer, inner = self._make_scroll_frame(self._fit_settings_tab)
            elif form_id == 'stitch':
                outer, inner = self._make_scroll_frame(self._stitch_settings_tab)
            else:
                outer, inner = self._make_scroll_frame(form_wrapper)
            outer.grid(row=0, column=0, sticky='nsew')
            outer.grid_remove()
            self._form_inner_frames[form_id] = (outer, inner)
            self._form_frames[form_id] = (outer, inner)
        results_frame = ttk.Frame(content_paned)
        content_paned.add(results_frame, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self._res = ResultsPanel(results_frame, root=self.root)
        self._res.grid(row=0, column=0, sticky='nsew')
        self._res.set_export_callback(self._show_export_dialog)
        self._res.set_load_callback(self._load_fitted_data_from_file)
        self._res.set_save_npz_callback(self._save_npz_quick)
        from flimkit.utils.config_manager import cfg as _cfg_mgr
        _saved_expert = _cfg_mgr.get_section('expert')
        _is_default = all(_saved_expert.get(k) == v for k, v in _EXPERT_DEFAULTS.items())
        self._expert_overrides: dict = {} if _is_default else _saved_expert
        self._build_fov_tab()
        self._build_stitch_tab()
        self._build_phasor_tab()
        self._build_batch_tab()
        self._build_machine_irf_tab()
        preview_frame = ttk.LabelFrame(self._main_paned, text='  FOV Preview  ', padding=4)
        self._main_paned.add(preview_frame, weight=2)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self._fov_preview = FOVPreviewPanel(preview_frame)
        self._fov_preview.grid(row=0, column=0, sticky='nsew')
        self._roi_analysis_panel.fov_preview = self._fov_preview
        self._fov_preview._roi_analysis_panel = self._roi_analysis_panel
        self._roi_analysis_panel.run_with_progress = self.run_with_progress
        self._roi_analysis_panel.get_fit_params = self._get_roi_fit_params
        self._phasor_panel = PhasorViewPanel(preview_frame, max_cursors=6)
        self._phasor_panel.on_change = self._on_phasor_change
        self._phasor_panel.frame.grid(row=0, column=0, sticky='nsew')
        self._phasor_panel.frame.grid_remove()
        self._phasor_panel.run_with_progress = self.run_with_progress
        self._phasor_panel.get_fit_params = self._get_roi_fit_params
        self._preview_frame_label = preview_frame
        self._switch_form('fov')
        redir = _Redirect(self._res.log, self._buf, root=self.root)
        sys.stdout = redir
        sys.stderr = redir
        self.root.after_idle(self._fit_window_to_screen)
        self.root.after_idle(self._set_pane_positions)
        self._setup_global_scroll()
        self.root.after(500, self._setup_global_dnd)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._set_window_icon()

    def _refresh_scrollable_frame(self, form_id: str):
        if form_id not in self._form_inner_frames:
            return
        outer, inner = self._form_inner_frames[form_id]
        if not hasattr(outer, '_canvas'):
            return
        if hasattr(outer, '_refresh_scheduled') and outer._refresh_scheduled:
            return
        outer._refresh_scheduled = True

        def do_refresh():
            try:
                self.root.update_idletasks()
                if form_id in ('fov', 'stitch'):
                    self._fit_settings_tab.update_idletasks()
                    self._analysis_tabs.update_idletasks()
                if not hasattr(outer, '_canvas') or not hasattr(outer, '_window_id'):
                    return
                canvas = outer._canvas
                window_id = outer._window_id
                outer.update()
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                if canvas_width > 1:
                    canvas.itemconfigure(window_id, width=canvas_width)
                inner.update()
                inner_height = inner.winfo_reqheight()
                if inner_height > 1:
                    canvas.itemconfigure(window_id, height=inner_height)
                bbox = canvas.bbox('all')
                if bbox:
                    canvas.configure(scrollregion=bbox)
                else:
                    h = max(canvas_height, inner_height, 300) if inner_height > 1 else 500
                    w = max(canvas_width, 300) if canvas_width > 1 else 400
                    canvas.configure(scrollregion=(0, 0, w, h))
                canvas.yview_moveto(0)
                canvas.update()
            except Exception:
                pass
            finally:
                outer._refresh_scheduled = False
        self.root.after(200, do_refresh)

    def _switch_form(self, form_id: str):
        self._mode_controller.switch(form_id)

    def _set_window_icon(self):
        base_path = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(__file__).parent
        icon_paths = [
            base_path / 'flimkit' / 'icon.png',
            base_path / 'icon.png',
            Path(__file__).parent / 'flimkit' / 'icon.png',
            Path(__file__).parent / 'icon.png',
        ]
        for icon_path in icon_paths:
            if icon_path.exists():
                try:
                    from PIL import Image, ImageTk
                    icon_img = Image.open(str(icon_path))
                    icon_img.thumbnail((32, 32), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(icon_img)
                    self.root.iconphoto(False, photo)
                    self.root._icon_photo = photo
                    break
                except Exception as e:
                    print(f'Warning: Could not load icon from {icon_path}: {e}')
                    continue

    def _on_close(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.root.destroy()

    def _capture_form_state(self) -> dict:
        try:
            state = {
                'active_form': getattr(self, '_current_form', 'fov'),
                'ptu_file': self.sv_ptu.get() if hasattr(self, 'sv_ptu') else '',
                'xlsx_file': self.sv_xlsx.get() if hasattr(self, 'sv_xlsx') else '',
                'irf_method': self._irf_fov.sv_method.get() if hasattr(self, '_irf_fov') else 'irf_xlsx',
                'irf_file': self._irf_fov.sv_path.get() if hasattr(self, '_irf_fov') else '',
                'nexp_fov': self.iv_nexp_fov.get() if hasattr(self, 'iv_nexp_fov') else 3,
                'fit_model_fov': self.sv_fit_model_fov.get() if hasattr(self, 'sv_fit_model_fov') else 'discrete',
                'ncomp_dist_fov': self.iv_ncomp_dist_fov.get() if hasattr(self, 'iv_ncomp_dist_fov') else 1,
                'tau_fit_lo': self.sv_tau_fit_lo.get() if hasattr(self, 'sv_tau_fit_lo') else '0.1',
                'tau_fit_hi': self.sv_tau_fit_hi.get() if hasattr(self, 'sv_tau_fit_hi') else '10.0',
                'nexp_st': self.iv_nexp_st.get() if hasattr(self, 'iv_nexp_st') else 3,
                'fit_model_st': self.sv_fit_model_st.get() if hasattr(self, 'sv_fit_model_st') else 'discrete',
                'ncomp_dist_st': self.iv_ncomp_dist_st.get() if hasattr(self, 'iv_ncomp_dist_st') else 1,
                'nexp_batch': self.iv_nexp_batch.get() if hasattr(self, 'iv_nexp_batch') else 3,
                'register': self.bv_register.get() if hasattr(self, 'bv_register') else False,
                'channel': self.sv_channel_focus.get() if hasattr(self, 'sv_channel_focus') else 'auto',
                'threshold': self.sv_int_threshold.get() if hasattr(self, 'sv_int_threshold') else '5',
                'out_fov': self.sv_out_fov.get() if hasattr(self, 'sv_out_fov') else '',
                'mode_fov': self.sv_mode_fov.get() if hasattr(self, 'sv_mode_fov') else 'both',
                'tau_min_fov': self.sv_tau_min_fov.get() if hasattr(self, 'sv_tau_min_fov') else '',
                'tau_max_fov': self.sv_tau_max_fov.get() if hasattr(self, 'sv_tau_max_fov') else '',
                'thr_fov_en': self.bv_thr_fov.get() if hasattr(self, 'bv_thr_fov') else False,
                'thr_fov_val': self.sv_thr_fov.get() if hasattr(self, 'sv_thr_fov') else '5',
                'cell_mask': self.bv_cell.get() if hasattr(self, 'bv_cell') else False,
                'correct_pileup': self.bv_correct_pileup.get() if hasattr(self, 'bv_correct_pileup') else False,
                'correct_pileup_st': self.bv_correct_pileup_st.get() if hasattr(self, 'bv_correct_pileup_st') else False,
                'tvb_ptu_fov': self.sv_tvb_ptu_fov.get() if hasattr(self, 'sv_tvb_ptu_fov') else '',
                'tvb_ptu_st': self.sv_tvb_ptu_st.get() if hasattr(self, 'sv_tvb_ptu_st') else '',
                'tvb_ptu_batch': self.sv_batch_tvb_ptu.get() if hasattr(self, 'sv_batch_tvb_ptu') else '',
                'xlif_file': self.sv_xlif.get() if hasattr(self, 'sv_xlif') else '',
                'ptu_dir': self.sv_ptu_dir.get() if hasattr(self, 'sv_ptu_dir') else '',
                'out_st': self.sv_out_st.get() if hasattr(self, 'sv_out_st') else '',
                'pipeline': self.sv_pipeline.get() if hasattr(self, 'sv_pipeline') else 'stitch_only',
                'bv_rotate': self.bv_rotate.get() if hasattr(self, 'bv_rotate') else True,
                'bv_perpix': self.bv_perpix.get() if hasattr(self, 'bv_perpix') else False,
                'tau_lo': self.sv_tau_lo.get() if hasattr(self, 'sv_tau_lo') else '',
                'tau_hi': self.sv_tau_hi.get() if hasattr(self, 'sv_tau_hi') else '',
                'int_lo': self.sv_int_lo.get() if hasattr(self, 'sv_int_lo') else '',
                'int_hi': self.sv_int_hi.get() if hasattr(self, 'sv_int_hi') else '',
                'thr_st_en': self.bv_thr_st.get() if hasattr(self, 'bv_thr_st') else False,
                'thr_st_val': self.sv_thr_st.get() if hasattr(self, 'sv_thr_st') else '',
                'bv_register': self.bv_register.get() if hasattr(self, 'bv_register') else True,
                'reg_max_shift': self.sv_reg_max_shift.get() if hasattr(self, 'sv_reg_max_shift') else '120',
                'irf_st_method': self._irf_st.sv_method.get() if hasattr(self, '_irf_st') else 'irf_xlsx',
                'irf_st_path': self._irf_st.sv_path.get() if hasattr(self, '_irf_st') else '',
                'tile_irf_dir': self.sv_tile_irf_dir.get() if hasattr(self, 'sv_tile_irf_dir') else '',
                'expert_overrides': self._expert_overrides if hasattr(self, '_expert_overrides') else {},
            }
            print(f'[Session] Captured form state: active_form={state.get('active_form')}')
            return state
        except Exception as e:
            print(f'[Session] Could not capture form state: {e}')
            return {}

    def _restore_form_state(self, state: dict):
        try:
            if 'irf_method' in state and hasattr(self, '_irf_fov'):
                self._irf_fov.sv_method.set(state['irf_method'])
            if 'irf_file' in state and hasattr(self, '_irf_fov'):
                self._irf_fov.sv_path.set(state['irf_file'])
            if 'nexp_fov' in state and hasattr(self, 'iv_nexp_fov'):
                self.iv_nexp_fov.set(state['nexp_fov'])
            if 'fit_model_fov' in state and hasattr(self, 'sv_fit_model_fov'):
                self.sv_fit_model_fov.set(state['fit_model_fov'])
            if 'ncomp_dist_fov' in state and hasattr(self, 'iv_ncomp_dist_fov'):
                self.iv_ncomp_dist_fov.set(state['ncomp_dist_fov'])
            if 'nexp_st' in state and hasattr(self, 'iv_nexp_st'):
                self.iv_nexp_st.set(state['nexp_st'])
            if 'fit_model_st' in state and hasattr(self, 'sv_fit_model_st'):
                self.sv_fit_model_st.set(state['fit_model_st'])
            if 'ncomp_dist_st' in state and hasattr(self, 'iv_ncomp_dist_st'):
                self.iv_ncomp_dist_st.set(state['ncomp_dist_st'])
            if 'nexp_batch' in state and hasattr(self, 'iv_nexp_batch'):
                self.iv_nexp_batch.set(state['nexp_batch'])
            if 'tau_fit_lo' in state and hasattr(self, 'sv_tau_fit_lo'):
                self.sv_tau_fit_lo.set(state['tau_fit_lo'])
            if 'tau_fit_hi' in state and hasattr(self, 'sv_tau_fit_hi'):
                self.sv_tau_fit_hi.set(state['tau_fit_hi'])
            if 'out_fov' in state and hasattr(self, 'sv_out_fov'): self.sv_out_fov.set(state['out_fov'])
            if 'mode_fov' in state and hasattr(self, 'sv_mode_fov'): self.sv_mode_fov.set(state['mode_fov'])
            if 'tau_min_fov' in state and hasattr(self, 'sv_tau_min_fov'): self.sv_tau_min_fov.set(state['tau_min_fov'])
            if 'tau_max_fov' in state and hasattr(self, 'sv_tau_max_fov'): self.sv_tau_max_fov.set(state['tau_max_fov'])
            if 'thr_fov_en' in state and hasattr(self, 'bv_thr_fov'): self.bv_thr_fov.set(state['thr_fov_en'])
            if 'thr_fov_val' in state and hasattr(self, 'sv_thr_fov'): self.sv_thr_fov.set(state['thr_fov_val'])
            if 'cell_mask' in state and hasattr(self, 'bv_cell'): self.bv_cell.set(state['cell_mask'])
            if 'correct_pileup' in state and hasattr(self, 'bv_correct_pileup'): self.bv_correct_pileup.set(state['correct_pileup'])
            if 'correct_pileup_st' in state and hasattr(self, 'bv_correct_pileup_st'): self.bv_correct_pileup_st.set(state['correct_pileup_st'])
            if 'tvb_ptu_fov' in state and hasattr(self, 'sv_tvb_ptu_fov'): self.sv_tvb_ptu_fov.set(state['tvb_ptu_fov'])
            if 'tvb_ptu_st' in state and hasattr(self, 'sv_tvb_ptu_st'): self.sv_tvb_ptu_st.set(state['tvb_ptu_st'])
            if 'tvb_ptu_batch' in state and hasattr(self, 'sv_batch_tvb_ptu'): self.sv_batch_tvb_ptu.set(state['tvb_ptu_batch'])
            if 'register' in state and hasattr(self, 'bv_register'):
                self.bv_register.set(state['register'])
            if 'channel' in state and hasattr(self, 'sv_channel_focus'):
                self.sv_channel_focus.set(state['channel'])
            if 'threshold' in state and hasattr(self, 'sv_int_threshold'):
                self.sv_int_threshold.set(state['threshold'])
            if 'out_st' in state and hasattr(self, 'sv_out_st'):
                self.sv_out_st.set(state['out_st'])
            if 'bv_rotate' in state and hasattr(self, 'bv_rotate'):
                self.bv_rotate.set(state['bv_rotate'])
            if 'tau_lo' in state and hasattr(self, 'sv_tau_lo'):
                self.sv_tau_lo.set(state['tau_lo'])
            if 'tau_hi' in state and hasattr(self, 'sv_tau_hi'):
                self.sv_tau_hi.set(state['tau_hi'])
            if 'int_lo' in state and hasattr(self, 'sv_int_lo'):
                self.sv_int_lo.set(state['int_lo'])
            if 'int_hi' in state and hasattr(self, 'sv_int_hi'):
                self.sv_int_hi.set(state['int_hi'])
            if 'thr_st_en' in state and hasattr(self, 'bv_thr_st'):
                self.bv_thr_st.set(state['thr_st_en'])
            if 'thr_st_val' in state and hasattr(self, 'sv_thr_st'):
                self.sv_thr_st.set(state['thr_st_val'])
            if 'bv_register' in state and hasattr(self, 'bv_register'):
                self.bv_register.set(state['bv_register'])
            if 'reg_max_shift' in state and hasattr(self, 'sv_reg_max_shift'):
                self.sv_reg_max_shift.set(state['reg_max_shift'])
            if 'tile_irf_dir' in state and hasattr(self, 'sv_tile_irf_dir'):
                self.sv_tile_irf_dir.set(state['tile_irf_dir'])
            if 'irf_st_method' in state and hasattr(self, '_irf_st'):
                self._irf_st.sv_method.set(state['irf_st_method'])
                self._irf_st._update()
            if 'irf_st_path' in state and hasattr(self, '_irf_st'):
                self._irf_st.sv_path.set(state['irf_st_path'])
            if hasattr(self, '_irf_fov'):
                self._irf_fov._update()
            if 'pipeline' in state and hasattr(self, 'sv_pipeline'):
                self.sv_pipeline.set(state['pipeline'])
                self._pipeline_changed()
            if 'bv_perpix' in state and hasattr(self, 'bv_perpix'):
                self.bv_perpix.set(state['bv_perpix'])
                self._perpix_toggled()
            if 'expert_overrides' in state and hasattr(self, '_expert_overrides'):
                ex = state['expert_overrides']
                if isinstance(ex, dict):
                    self._expert_overrides = ex
                    self._update_expert_banners()
            if 'active_form' in state:
                _form = state['active_form']
                if isinstance(_form, int):
                    _form = [None, 'fov', 'stitch', 'batch', 'irf', 'phasor'][_form] or 'fov'
                if _form in self._form_buttons:
                    self._switch_form(_form)
            print(f'[Session] Restored form state')
        except Exception as e:
            print(f'[Session] Could not restore form state: {e}')
            import traceback
            traceback.print_exc()

    def _save_roi_progress(self, path: str, fit_result: dict, summary_rows: list):
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            import numpy as np
            base_path = Path(path)
            if base_path.is_file():
                session_file = base_path.parent / f'{base_path.stem}.roi_session.npz'
            else:
                session_file = base_path / 'roi_session.npz'
            form_state = self._capture_form_state()
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'source': str(path),
                'form_state_json': json.dumps(form_state, default=str),
            }
            print(f'[Session] Saving fit_result keys: {list(fit_result.keys())}')
            for key, val in fit_result.items():
                if isinstance(val, np.ndarray):
                    session_data[key] = val
                    print(f'  ✓ Saved array: {key} {val.shape}')
                elif isinstance(val, dict):
                    try:
                        hoisted = {}
                        json_safe = {}
                        for k2, v2 in val.items():
                            if isinstance(v2, np.ndarray):
                                arr_key = f'{key}_arr_{k2}'
                                hoisted[arr_key] = v2
                                print(f'  ✓ Hoisted array from {key}: {k2} {v2.shape}')
                            else:
                                json_safe[k2] = v2
                        session_data[f'{key}_json'] = json.dumps(json_safe, default=str)
                        session_data.update(hoisted)
                        print(f'  ✓ Saved dict: {key} ({len(hoisted)} arrays hoisted)')
                    except Exception as e:
                        print(f'  ✗ Could not save dict {key}: {e}')
                elif isinstance(val, (list, tuple)):
                    try:
                        arr = np.array(val)
                        if arr.dtype != object:
                            session_data[key] = arr
                            print(f'  ✓ Saved list: {key} {arr.shape}')
                    except Exception:
                        pass
                elif val is not None and not callable(val):
                    try:
                        session_data[key] = val
                    except Exception:
                        pass
            if summary_rows:
                params, values, units = zip(*summary_rows)
                session_data['summary_params'] = np.array(params, dtype=object)
                session_data['summary_values'] = np.array(values, dtype=object)
                session_data['summary_units'] = np.array(units, dtype=object)
            if self._fov_preview._ptu_path:
                session_data['fov_ptu_path'] = self._fov_preview._ptu_path
            if self._fov_preview._lifetime_map is not None:
                session_data['fov_lifetime_map'] = self._fov_preview._lifetime_map
            if self._fov_preview._intensity_map is not None:
                session_data['fov_intensity_map'] = self._fov_preview._intensity_map
            session_data['fov_color_scale'] = json.dumps(self._fov_preview._flim_color_scale)
            session_data['fov_n_exp'] = self._fov_preview._n_exp
            session_data['fov_regions'] = self._fov_preview._roi_manager.to_json()
            np.savez_compressed(session_file, **session_data)
            print(f'✓ Session saved: {session_file}')
            print(f'  Saved {len(session_data)} items (fit results + form state + FOV preview)')
            self._current_session_file = str(session_file)
        except Exception as e:
            import traceback
            print(f'[Save Error] {e}')
            traceback.print_exc()

    def _load_roi_session(self, session_path: str) -> dict:
        try:
            import numpy as np
            data = np.load(session_path, allow_pickle=True)
            loaded = {}
            for key in data.files:
                val = data[key]
                if isinstance(val, np.ndarray):
                    if val.ndim == 0:
                        val = val.item()
                    elif val.dtype == object:
                        val = val.tolist()
                loaded[key] = val
            print(f'✓ Loaded session from {session_path}')
            self._current_session_file = session_path
            return loaded
        except Exception as e:
            print(f'[Load Session Error] {e}')
            import traceback
            traceback.print_exc()
            return {}

    def _auto_load_session_for_ptu(self, ptu_path: str):
        try:
            from pathlib import Path
            import json
            import numpy as np
            ptu_path = Path(ptu_path)
            session_file = ptu_path.parent / f'{ptu_path.stem}.roi_session.npz'
            if session_file.exists():
                print(f'[Auto-Load] Found session for PTU: {session_file.name}')
                session_data = self._load_roi_session(str(session_file))
                if not session_data:
                    return
                if 'form_state_json' in session_data:
                    try:
                        form_state_str = session_data['form_state_json']
                        if isinstance(form_state_str, np.ndarray):
                            form_state_str = form_state_str.item()
                        if isinstance(form_state_str, bytes):
                            form_state_str = form_state_str.decode('utf-8')
                        form_state = json.loads(form_state_str)
                        self._restore_form_state(form_state)
                    except Exception as e:
                        print(f'[Auto-Load] Could not restore form state: {e}')
                        import traceback
                        traceback.print_exc()
                try:
                    params = session_data.get('summary_params', [])
                    values = session_data.get('summary_values', [])
                    units = session_data.get('summary_units', [])
                    if isinstance(params, np.ndarray):
                        params = params.tolist()
                    if isinstance(values, np.ndarray):
                        values = values.tolist()
                    if isinstance(units, np.ndarray):
                        units = units.tolist()
                    rows = []
                    for param, val, unit in zip(params, values, units):
                        if isinstance(param, bytes):
                            param = param.decode('utf-8')
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        if isinstance(unit, bytes):
                            unit = unit.decode('utf-8')
                        rows.append((str(param), str(val), str(unit)))
                    if rows:
                        self._res.populate_summary(rows)
                    if 'fov_regions' in session_data:
                        try:
                            import json as _json_roi
                            regions_json = session_data['fov_regions']
                            if isinstance(regions_json, bytes):
                                regions_json = regions_json.decode('utf-8')
                            self._fov_preview._load_regions_from_json(regions_json)
                            if self._roi_analysis_panel is not None:
                                self._roi_analysis_panel._refresh_region_list()
                        except Exception as _e:
                            print(f'[Auto-Load] Could not restore regions: {_e}')
                    if 'fov_intensity_map' in session_data and 'fov_lifetime_map' in session_data:
                        intensity = session_data['fov_intensity_map']
                        lifetime = session_data['fov_lifetime_map']
                        if isinstance(intensity, np.ndarray):
                            self._fov_preview._intensity_map = intensity
                        if isinstance(lifetime, np.ndarray):
                            self._fov_preview._lifetime_map = lifetime
                        if 'fov_color_scale' in session_data:
                            try:
                                import json
                                cs = session_data['fov_color_scale']
                                if isinstance(cs, bytes):
                                    cs = cs.decode('utf-8')
                                self._fov_preview._flim_color_scale = json.loads(cs)
                            except Exception:
                                pass
                        if 'fov_n_exp' in session_data:
                            n_exp = session_data['fov_n_exp']
                            if isinstance(n_exp, (np.integer, int)):
                                self._fov_preview._n_exp = int(n_exp)
                        try:
                            self._fov_preview._update_flim_display()
                        except Exception as _e:
                            print(f'[Auto-Load] Could not render FLIM display: {_e}')
                        try:
                            ax_decay = self._fov_preview._ax_decay
                            ax_decay.clear()
                            if 'decay' in session_data and 'time_ns' in session_data:
                                decay = session_data['decay']
                                time_ns = session_data['time_ns']
                                if isinstance(decay, np.ndarray) and isinstance(time_ns, np.ndarray):
                                    ax_decay.semilogy(time_ns, decay, 'o-', color='steelblue',
                                                    linewidth=1.5, markersize=3, label='Measured', alpha=0.7)
                                    irf = session_data.get('irf_prompt')
                                    if irf is not None and isinstance(irf, np.ndarray) and irf.max() > 0:
                                        self._fov_preview._irf_prompt = irf
                                        irf_scaled = (irf / irf.max()) * decay.max() * 0.2
                                        ax_decay.semilogy(time_ns[:len(irf)], np.maximum(irf_scaled, 1e-2),
                                                        color='orange', linewidth=2.0, label='IRF', alpha=0.8)
                                    gs = _reconstruct_dict_from_session(session_data, 'global_summary')
                                    model = gs.get('model')
                                    if model is not None and isinstance(model, str):
                                        model = _safe_array_from_json(model)
                                    if model is not None and len(model) > 0:
                                        ax_decay.semilogy(time_ns, model, color='red', linewidth=2.0,
                                                        label='Fitted', alpha=0.8)
                                    ax_decay.legend(fontsize=8, loc='upper right', labelcolor='black')
                                    ax_resid = self._fov_preview._ax_resid
                                    ax_resid.clear()
                                    ax_resid.set_facecolor('white')
                                    if model is not None and len(model) == len(decay):
                                        with np.errstate(invalid='ignore', divide='ignore'):
                                            resid = np.where(model > 0,
                                                             (decay - model) / np.sqrt(model),
                                                             0.0)
                                        self._fov_preview._cached_resid_data = (time_ns.copy(), resid)
                                        ax_resid.plot(time_ns, resid, color='steelblue', linewidth=1.0)
                                        ax_resid.axhline(0, color='red', linewidth=1.0,
                                                         linestyle='--', alpha=0.8)
                                        ax_resid.set_ylabel('Resid. (σ)', fontsize=7, color='white')
                                        chi2_r = gs.get('reduced_chi2_tail')
                                        if chi2_r is not None:
                                            ax_resid.annotate(
                                                f'χ²_r = {chi2_r:.3f}',
                                                xy=(0.98, 0.85), xycoords='axes fraction',
                                                ha='right', va='top', fontsize=7,
                                                color='white',
                                                bbox=dict(boxstyle='round,pad=0.2',
                                                          fc='#333333', alpha=0.7),
                                            )
                                    ax_resid.set_xlabel('Time (ns)', color='white')
                                    ax_resid.tick_params(labelsize=7, colors='white')
                                    ax_resid.grid(True, alpha=0.3)
                            ax_decay.set_title('Summed Decay (reloaded)', fontsize=10, fontweight='bold', color='white')
                            ax_decay.set_xlabel('Time (ns)', color='white')
                            ax_decay.set_ylabel('Photon Count', color='white')
                            ax_decay.grid(True, alpha=0.3)
                            ax_decay.tick_params(labelsize=8, colors='white')
                            _SESSION_ONLY_KEYS = {
                                'timestamp', 'source', 'form_state_json', 'fov_ptu_path',
                                'fov_lifetime_map', 'fov_intensity_map', 'fov_color_scale',
                                'fov_n_exp', 'fov_regions', 'summary_params', 'summary_values',
                                'summary_units', 'decay', 'time_ns', 'irf_prompt',
                            }
                            fit_result_for_export = {
                                k: v for k, v in session_data.items()
                                if k not in _SESSION_ONLY_KEYS
                                and not k.endswith('_json')
                                and isinstance(v, np.ndarray)
                                and v.ndim == 2
                            }
                            if not fit_result_for_export:
                                if isinstance(session_data.get('fov_intensity_map'), np.ndarray):
                                    fit_result_for_export['intensity'] = session_data['fov_intensity_map']
                                if isinstance(session_data.get('fov_lifetime_map'), np.ndarray):
                                    fit_result_for_export['lifetime'] = session_data['fov_lifetime_map']
                            self._res.set_fit_result(
                                fit_result_for_export,
                                str(ptu_path.parent),
                                npz_path=str(session_file),
                                scan_name=self._current_scan_stem(),
                            )
                            self._res._status.set('✓ Session restored - ready to export or re-fit')
                            self._fov_preview._ctrl_frame.grid()
                            self._fov_preview._canvas_mpl.draw_idle()
                            print('[Auto-Load] ✓ Session restored')
                        except Exception as e:
                            print(f'[Auto-Load] Could not redraw FOV: {e}')
                except Exception as e:
                    print(f'[Auto-Load] Could not restore results: {e}')
                    import traceback
                    traceback.print_exc()
                finally:
                    try:
                        if self._roi_analysis_panel is not None:
                            self._roi_analysis_panel._refresh_region_list()
                    except Exception:
                        pass
            else:
                print(f'[Auto-Load] No session found for {ptu_path.name}')
        except Exception as e:
            print(f'[Auto-Load Error] {e}')

    def _auto_load_session_for_stitch(self, output_dir: str):
        try:
            from pathlib import Path
            import json
            import numpy as np
            session_file = Path(output_dir) / 'roi_session.npz'
            if not session_file.exists():
                return
            print(f'[Auto-Load] Found tile session: {session_file.name}')
            session_data = np.load(session_file, allow_pickle=True)
            loaded = {key: session_data[key] for key in session_data.files}
            if 'form_state_json' in loaded:
                try:
                    form_state_str = loaded['form_state_json']
                    if isinstance(form_state_str, np.ndarray):
                        form_state_str = form_state_str.item()
                    if isinstance(form_state_str, bytes):
                        form_state_str = form_state_str.decode('utf-8')
                    form_state = json.loads(form_state_str)
                    self._restore_form_state(form_state)
                except Exception as e:
                    print(f'[Auto-Load] Could not restore form state: {e}')
            if 'summary_params' in loaded:
                params = loaded['summary_params']
                values = loaded['summary_values']
                units = loaded['summary_units']
                if isinstance(params, np.ndarray):
                    params = params.tolist()
                if isinstance(values, np.ndarray):
                    values = values.tolist()
                if isinstance(units, np.ndarray):
                    units = units.tolist()
                rows = []
                for p, v, u in zip(params, values, units):
                    if isinstance(p, bytes): p = p.decode('utf-8')
                    if isinstance(v, bytes): v = v.decode('utf-8')
                    if isinstance(u, bytes): u = u.decode('utf-8')
                    rows.append((str(p), str(v), str(u)))
                if rows:
                    self._res.populate_summary(rows)
            if 'fov_color_scale' in loaded:
                try:
                    cs = loaded['fov_color_scale']
                    if isinstance(cs, bytes):
                        cs = cs.decode('utf-8')
                    self._fov_preview._flim_color_scale = json.loads(cs)
                except Exception:
                    pass
            if 'fov_n_exp' in loaded:
                n_exp = loaded['fov_n_exp']
                if isinstance(n_exp, (np.integer, int)):
                    self._fov_preview._n_exp = int(n_exp)
            if 'fov_regions' in loaded:
                regions_json = loaded['fov_regions']
                if isinstance(regions_json, np.ndarray):
                    regions_json = regions_json.item()
                if isinstance(regions_json, bytes):
                    regions_json = regions_json.decode('utf-8')
                if regions_json:
                    self._fov_preview._load_regions_from_json(regions_json)
                    if self._roi_analysis_panel:
                        self._roi_analysis_panel._refresh_region_list()
            fit_result = {}
            for key, val in loaded.items():
                if key in ('summary_params', 'summary_values', 'summary_units',
                        'form_state_json', 'fov_regions', 'fov_color_scale', 'fov_n_exp'):
                    continue
                fit_result[key] = val
            if 'global_summary_json' in fit_result:
                fit_result['global_summary'] = _reconstruct_dict_from_session(fit_result, 'global_summary')
            if self._fov_preview._intensity_map is None:
                self._fov_preview.load_stitched_roi(output_dir)
            self._fov_preview.display_fit_results(None, fit_result)
            self._fov_preview._canvas_mpl.draw_idle()
            self._res.set_fit_result(fit_result, output_dir, npz_path=str(session_file),
                                     scan_name=self._current_scan_stem())
            self._res._status.set('✓ Session restored - ready to export or re-fit')
            self._fov_preview._ctrl_frame.grid()
            self._res._export_btn.configure(state='normal')
            print('[Auto-Load] Tile session fully restored')
        except Exception as e:
            print(f'[Auto-Load] Error loading tile session: {e}')
            import traceback
            traceback.print_exc()

    def load_roi_fit(self, npz_path: str) -> dict:
        try:
            import numpy as np
            data = np.load(npz_path, allow_pickle=True)
            loaded = {}
            for key in data.files:
                val = data[key]
                if hasattr(val, 'dtype') and val.dtype == object:
                    val = val.tolist()
                loaded[key] = val
            print(f'✓ Loaded ROI fit from {npz_path}')
            return loaded
        except Exception as e:
            print(f'✗ Failed to load ROI fit: {e}')
            return {}

    def _save_npz_quick(self, output_dir: str):
        try:
            from pathlib import Path
            import shutil
            import numpy as np
            session_source = None
            if self._current_session_file:
                session_source = Path(self._current_session_file)
                if not session_source.exists():
                    session_source = None
            if not session_source:
                ptu_path = self._fov_preview._ptu_path if self._fov_preview else None
                if ptu_path is not None:
                    if isinstance(ptu_path, np.ndarray):
                        ptu_path = ptu_path.item() if ptu_path.ndim == 0 else str(ptu_path[0])
                    if isinstance(ptu_path, bytes):
                        ptu_path = ptu_path.decode('utf-8')
                    ptu_path = str(ptu_path)
                if ptu_path:
                    base_path = Path(ptu_path)
                    if base_path.is_file():
                        candidate = base_path.parent / f'{base_path.stem}.roi_session.npz'
                        if candidate.exists():
                            session_source = candidate
            if not session_source or not session_source.exists():
                messagebox.showwarning('No session data', 'No saved session (.roi_session.npz) found.\n\nRun a fit first to create a session.')
                return
            output_path = Path(filedialog.askdirectory(
                title='Save Session File',
                initialdir=output_dir))
            if not output_path or output_path == Path():
                return
            session_dest = output_path / session_source.name
            if session_source.samefile(session_dest) if session_dest.exists() else session_source == session_dest:
                messagebox.showinfo('Already Saved',
                    f'Session already saved at:\n{session_source}')
                return
            if session_dest.exists():
                response = messagebox.askyesno('File Exists',
                    f'File already exists:\n{session_dest.name}\n\nOverride?')
                if not response:
                    return
            session_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(session_source, session_dest)
            messagebox.showinfo('Success', f'Session saved:\n{session_dest.name}\nat {output_path}')
            print(f'✓ Session saved: {session_dest}')
        except Exception as e:
            print(f'[Save Session Error] {e}')
            import traceback
            traceback.print_exc()
            messagebox.showerror('Error', f'Failed to save session:\n{str(e)}')

    def _load_fitted_data_from_file(self, npz_path: str, suppress_popups: bool = False):
        try:
            import numpy as np
            import json
            from pathlib import Path
            fit_result = self.load_roi_fit(npz_path)
            if not fit_result:
                if not suppress_popups:
                    messagebox.showerror('Load Error', f'Failed to load fitted data from:\n{npz_path}')
                return
            params = fit_result.get('summary_params', [])
            values = fit_result.get('summary_values', [])
            units = fit_result.get('summary_units', [])
            rows = []
            if isinstance(params, np.ndarray): params = params.tolist()
            if isinstance(values, np.ndarray): values = values.tolist()
            if isinstance(units, np.ndarray): units = units.tolist()
            for param, val, unit in zip(params, values, units):
                if isinstance(param, bytes): param = param.decode('utf-8')
                if isinstance(val, bytes): val = val.decode('utf-8')
                if isinstance(unit, bytes): unit = unit.decode('utf-8')
                rows.append((str(param), str(val), str(unit)))
            try:
                if rows:
                    self._res.populate_summary(rows)
            except Exception:
                import traceback
                traceback.print_exc()
            try:
                if 'fov_ptu_path' in fit_result:
                    ptu_path = fit_result['fov_ptu_path']
                    if isinstance(ptu_path, bytes):
                        ptu_path = ptu_path.decode('utf-8')
                    elif isinstance(ptu_path, np.ndarray):
                        ptu_path = ptu_path.item() if ptu_path.ndim == 0 else str(ptu_path[0])
                        if isinstance(ptu_path, bytes):
                            ptu_path = ptu_path.decode('utf-8')
                    self._fov_preview._ptu_path = str(ptu_path) if ptu_path else None
                if 'fov_lifetime_map' in fit_result:
                    lifetime = fit_result['fov_lifetime_map']
                    if isinstance(lifetime, np.ndarray):
                        self._fov_preview._lifetime_map = lifetime
                if 'fov_intensity_map' in fit_result:
                    intensity = fit_result['fov_intensity_map']
                    if isinstance(intensity, np.ndarray):
                        self._fov_preview._intensity_map = intensity
                if 'fov_color_scale' in fit_result:
                    try:
                        cs = fit_result['fov_color_scale']
                        if isinstance(cs, bytes):
                            cs = cs.decode('utf-8')
                        self._fov_preview._flim_color_scale = json.loads(cs)
                    except Exception:
                        pass
            except Exception:
                import traceback
                traceback.print_exc()
            import sys
            sys.stdout.flush()
            if 'fov_regions' in fit_result:
                try:
                    regions_json = fit_result['fov_regions']
                    if isinstance(regions_json, np.ndarray):
                        regions_json = regions_json.item() if regions_json.ndim == 0 else regions_json[0]
                    if isinstance(regions_json, bytes):
                        regions_json = regions_json.decode('utf-8')
                    self._fov_preview._load_regions_from_json(regions_json)
                    if self._roi_analysis_panel is not None:
                        self._roi_analysis_panel._refresh_region_list()
                except Exception:
                    import traceback
                    traceback.print_exc()
            if 'fov_n_exp' in fit_result:
                n_exp = fit_result['fov_n_exp']
                if isinstance(n_exp, (np.integer, int)):
                    self._fov_preview._n_exp = int(n_exp)
            try:
                if self._fov_preview._lifetime_map is not None and self._fov_preview._intensity_map is not None:
                    from flimkit.UI import flim_display
                    intensity = self._fov_preview._intensity_map
                    lifetime = self._fov_preview._lifetime_map
                    ax_img = self._fov_preview._ax_img
                    ax_flim = self._fov_preview._ax_flim
                    ax_cbar = self._fov_preview._ax_cbar
                    fig = self._fov_preview._fig
                    ax_img.clear()
                    intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
                    ax_img.imshow(intensity_clipped, cmap='inferno', origin='upper')
                    ax_img.set_title('Intensity Image', fontsize=10, fontweight='bold')
                    ax_img.set_xlabel('X (pixels)')
                    ax_img.set_ylabel('Y (pixels)')
                    cs = self._fov_preview._flim_color_scale
                    scaled = flim_display.apply_color_scale(
                        lifetime,
                        vmin=cs.get('vmin'),
                        vmax=cs.get('vmax'),
                        gamma=cs.get('gamma', 1.0),
                    )
                    cmap_obj = flim_display.get_colormap(cs.get('cmap', 'viridis'))
                    cmap_obj.set_bad(color='black')
                    ax_flim.clear()
                    ax_cbar.clear()
                    im = ax_flim.imshow(scaled, cmap=cmap_obj, origin='upper', vmin=0, vmax=1)
                    ax_flim.set_title('FLIM Lifetime (τ-weighted)', fontsize=10, fontweight='bold')
                    ax_flim.set_xlabel('X (pixels)')
                    ax_flim.set_ylabel('Y (pixels)')
                    valid = lifetime[~np.isnan(lifetime)]
                    if valid.size > 0:
                        vmin_cb = cs.get('vmin') or float(np.nanmin(valid))
                        vmax_cb = cs.get('vmax') or float(np.nanmax(valid))
                        cbar = fig.colorbar(im, cax=ax_cbar)
                        cbar.set_label('τ (ns)', fontsize=8)
                        ticks = np.linspace(0, 1, 5)
                        cbar.set_ticks(ticks)
                        cbar.set_ticklabels([f'{vmin_cb + t*(vmax_cb - vmin_cb):.2f}' for t in ticks], fontsize=7)
                    ax_decay = self._fov_preview._ax_decay
                    ax_decay.clear()
                    decay = fit_result.get('decay')
                    time_ns = fit_result.get('time_ns')
                    if decay is not None and time_ns is not None:
                        ax_decay.semilogy(time_ns, decay, 'o-', color='steelblue',
                                        linewidth=1.5, markersize=3, label='Measured', alpha=0.7)
                        irf = fit_result.get('irf_prompt')
                        if irf is not None and len(irf) > 0 and irf.max() > 0:
                            irf_scaled = (irf / irf.max()) * decay.max() * 0.2
                            ax_decay.semilogy(time_ns[:len(irf)], np.maximum(irf_scaled, 1e-2),
                                            color='orange', linewidth=2.0, label='IRF', alpha=0.8)
                        gs = _reconstruct_dict_from_session(fit_result, 'global_summary')
                        model = gs.get('model')
                        if model is not None and isinstance(model, str):
                            model = _safe_array_from_json(model)
                        if model is not None and len(model) > 0:
                            ax_decay.semilogy(time_ns, model, color='red', linewidth=2.0,
                                            label='Fitted', alpha=0.8)
                        ax_decay.legend(fontsize=8, loc='upper right', labelcolor='black')
                    ax_decay.set_title('Summed Decay', fontsize=10, fontweight='bold', color='white')
                    ax_decay.set_xlabel('Time (ns)', color='white')
                    ax_decay.set_ylabel('Photon Count', color='white')
                    ax_decay.grid(True, alpha=0.3)
                    ax_decay.tick_params(labelsize=8, colors='white')
                    self._fov_preview._redraw_region_overlays()
                    self._fov_preview._ctrl_frame.grid()
                    self._fov_preview._canvas_mpl.draw_idle()
                    print(f'[Load] Restored FOV preview from cached data')
            except Exception as e:
                import traceback
                print(f'[Load] Could not redraw FOV preview: {e}')
                traceback.print_exc()
            if 'form_state_json' in fit_result:
                try:
                    fs = fit_result['form_state_json']
                    if isinstance(fs, np.ndarray): fs = fs.item()
                    if isinstance(fs, bytes): fs = fs.decode()
                    form_state = json.loads(fs)
                    ptu_in_session = form_state.get('ptu_file', '').strip()
                    self._last_loaded_ptu = (
                        ptu_in_session
                        if ptu_in_session
                        else (self.sv_ptu.get().strip() if hasattr(self, 'sv_ptu') else '')
                    )
                    self._restore_form_state(form_state)
                except Exception as e:
                    print(f'[Load] Could not restore form state: {e}')
            output_dir = str(Path(npz_path).parent)
            self._res.set_fit_result(fit_result, output_dir, npz_path=npz_path,
                                     scan_name=self._current_scan_stem())
            if not suppress_popups:
                messagebox.showinfo('Success', f'Loaded fitted data from:\n{Path(npz_path).name}')
        except Exception as e:
            import traceback
            traceback.print_exc()
            if not suppress_popups:
                messagebox.showerror('Error', f'Failed to load fitted data:\n{str(e)}')

    def _show_export_dialog(self, image_dict: dict, output_dir: str):
        try:
            print(f'[Export Dialog] Starting with {len(image_dict)} items')
            if not image_dict or not any(isinstance(v, np.ndarray) for v in image_dict.values()):
                messagebox.showinfo('No images', 'No fit images available to export.')
                return
            available_images = {}
            for k, v in image_dict.items():
                if isinstance(v, np.ndarray):
                    if (v.ndim == 2) or (v.ndim == 3 and v.shape[2] == 3):
                        available_images[k] = v
                        print(f'[Export] Found image: {k} shape={v.shape}')
            if not available_images:
                messagebox.showinfo('No images', 'No valid FLIM result images found to export.')
                return
            print(f'[Export] {len(available_images)} images available: {list(available_images.keys())}')
            dlg = tk.Toplevel(self.root)
            dlg.title('Export Results')
            dlg.resizable(True, True)
            dlg.minsize(560, 400)
            dlg.transient(self.root)
            dlg.grab_set()
            bv_scalebar = tk.BooleanVar(value=True)
            bv_annotations = tk.BooleanVar(value=True)
            image_vars = {key: tk.BooleanVar(value=True) for key in available_images}
            ttk.Label(dlg, text='Export Results', font=('TkDefaultFont', 11, 'bold')).pack(pady=10)
            img_frame = ttk.LabelFrame(dlg, text='Images to Export', padding=10)
            img_frame.pack(fill='both', expand=True, padx=20, pady=5)
            sorted_images = sorted(available_images.keys())
            n_cols = 3
            n_rows = (len(sorted_images) + n_cols - 1) // n_cols
            for col in range(n_cols):
                img_frame.columnconfigure(col, weight=1)
            for idx, key in enumerate(sorted_images):
                row = idx % n_rows
                col = idx // n_rows
                ttk.Checkbutton(img_frame, text=key.replace('_', ' ').title(),
                               variable=image_vars[key]).grid(row=row, column=col, sticky='w', padx=5, pady=2)
            sel_btn_frame = ttk.Frame(img_frame)
            sel_btn_frame.grid(row=n_rows, column=0, columnspan=n_cols, sticky='w', pady=(10, 0))
            def select_all():
                for v in image_vars.values():
                    v.set(True)
            def select_none():
                for v in image_vars.values():
                    v.set(False)
            ttk.Button(sel_btn_frame, text='All', command=select_all, width=8).pack(side='left', padx=2)
            ttk.Button(sel_btn_frame, text='None', command=select_none, width=8).pack(side='left', padx=2)
            opt_frame = ttk.LabelFrame(dlg, text='Rendering Options', padding=10)
            opt_frame.pack(fill='x', padx=20, pady=5)
            ttk.Checkbutton(opt_frame, text='Include scale bar', variable=bv_scalebar).pack(anchor='w', pady=3)
            ttk.Checkbutton(opt_frame, text='Include ROI annotations', variable=bv_annotations).pack(anchor='w', pady=3)
            fmt_frame = ttk.LabelFrame(dlg, text='Image Format', padding=10)
            fmt_frame.pack(fill='x', padx=20, pady=5)
            bv_format = tk.StringVar(value='png')
            ttk.Radiobutton(fmt_frame, text='PNG (smaller file size, web-friendly)',
                           variable=bv_format, value='png').pack(anchor='w', pady=3)
            ttk.Radiobutton(fmt_frame, text='OME-TIFF (lossless, metadata-rich)',
                           variable=bv_format, value='ometiff').pack(anchor='w', pady=3)
            loc_frame = ttk.LabelFrame(dlg, text='Save Location', padding=10)
            loc_frame.pack(fill='x', padx=20, pady=5)
            export_path = tk.StringVar(value=output_dir)
            ttk.Label(loc_frame, text='Path:').pack(side='left')
            ttk.Entry(loc_frame, textvariable=export_path, width=40).pack(side='left', padx=5, fill='x', expand=True)

            def browse_folder():
                from tkinter import filedialog
                folder = filedialog.askdirectory(initialdir=output_dir, title='Select export folder')
                if folder:
                    export_path.set(folder)
                    print(f'[Export] Save location changed to: {folder}')
            ttk.Button(loc_frame, text='Browse', command=browse_folder, width=8).pack(side='left', padx=2)

            def do_export():
                try:
                    selected_images = {k: v for k, v in available_images.items()
                                     if image_vars[k].get()}
                    if not selected_images:
                        messagebox.showwarning('No selection', 'Please select at least one image to export.')
                        return
                    export_dir = export_path.get()
                    if not export_dir.strip():
                        messagebox.showwarning('No path', 'Please select an export directory.')
                        return
                    fmt = bv_format.get()
                    print(f'[Export] Exporting {len(selected_images)} images in {fmt.upper()} format to {export_dir}')
                    self._export_images(selected_images, export_dir,
                                       with_scalebar=bv_scalebar.get(),
                                       with_annotations=bv_annotations.get(),
                                       format=fmt)
                    dlg.destroy()
                    messagebox.showinfo('Success', f'Results exported to\n{export_dir}')
                except Exception as e:
                    print(f'[Export Error] {e}')
                    import traceback
                    traceback.print_exc()
                    messagebox.showerror('Export Error', f'Failed to export:\n{str(e)}')
            btn_frame = ttk.Frame(dlg)
            btn_frame.pack(pady=15, fill='x', padx=20)
            ttk.Button(btn_frame, text='💾 Export', command=do_export).pack(side='left', padx=5, fill='x', expand=True)
            ttk.Button(btn_frame, text='Cancel', command=dlg.destroy).pack(side='left', padx=5)
            dlg.update_idletasks()
            req_w = max(560, dlg.winfo_reqwidth() + 24)
            req_h = min(dlg.winfo_reqheight() + 24, int(dlg.winfo_screenheight() * 0.85))
            dlg.geometry(f'{req_w}x{req_h}')
        except Exception as e:
            print(f'[Export Dialog Error] {e}')
            import traceback
            traceback.print_exc()
            messagebox.showerror('Dialog Error', f'Failed to create export dialog:\n{str(e)}')

    def _export_images(self, image_dict: dict, output_dir: str,
                      with_scalebar: bool = True, with_annotations: bool = True,
                      format: str = 'png'):
        try:
            from pathlib import Path
            import numpy as np
            import matplotlib.pyplot as plt
            fmt = format.lower()
            print(f'[Export Images] Exporting {len(image_dict)} images in {fmt.upper()} format to {output_dir}')
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            exported_count = 0
            pixel_size_um = self._get_pixel_size_um()
            if with_scalebar and pixel_size_um is None:
                print('[Export] No pixel size available - scale bar will be omitted')
                with_scalebar = False
            if fmt == 'ometiff':
                try:
                    import tifffile
                    if 'intensity' in image_dict and isinstance(image_dict['intensity'], np.ndarray):
                        try:
                            intensity = image_dict['intensity']
                            intensity_16bit = (intensity / intensity.max() * 65535).astype(np.uint16) if intensity.max() > 0 else intensity.astype(np.uint16)
                            output_file = output_path / 'intensity.ome.tiff'
                            tifffile.imwrite(output_file, intensity_16bit, photometric='minisblack',
                                           metadata={'description': 'FLIM Intensity Image'})
                            print(f'✓ Exported OME-TIFF intensity: {output_file.name} ({intensity.shape})')
                            exported_count += 1
                        except Exception as e:
                            print(f'[Export] Error exporting intensity TIFF: {e}')
                    if 'lifetime' in image_dict and isinstance(image_dict['lifetime'], np.ndarray):
                        try:
                            lifetime = image_dict['lifetime']
                            lifetime = np.nan_to_num(lifetime, nan=0.0)
                            lifetime_32bit = lifetime.astype(np.float32)
                            output_file = output_path / 'lifetime.ome.tiff'
                            tifffile.imwrite(output_file, lifetime_32bit, photometric='minisblack',
                                           metadata={'description': 'FLIM Lifetime Map (ns)'})
                            print(f'✓ Exported OME-TIFF lifetime: {output_file.name} ({lifetime.shape})')
                            exported_count += 1
                        except Exception as e:
                            print(f'[Export] Error exporting lifetime TIFF: {e}')
                except ImportError:
                    print(f'[Export] tifffile not installed, falling back to PNG')
                    fmt = 'png'
            if fmt == 'png':
                if 'intensity' in image_dict and isinstance(image_dict['intensity'], np.ndarray):
                    try:
                        intensity = image_dict['intensity']
                        print(f'[Export] Intensity shape: {intensity.shape}')
                        h, w = intensity.shape
                        fig = plt.figure(figsize=(w/100, h/100), dpi=100, facecolor='black', edgecolor='black')
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.set_facecolor('black')
                        intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
                        im = ax.imshow(intensity_clipped, cmap='inferno', origin='upper', aspect='auto')
                        ax.axis('off')
                        if with_scalebar:
                            self._draw_scale_bar(ax, w, h, pixel_size_um)
                        output_file = output_path / 'intensity.png'
                        fig.savefig(output_file, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
                        plt.close(fig)
                        print(f'✓ Exported PNG intensity: {output_file.name} ({intensity.shape})')
                        exported_count += 1
                    except Exception as e:
                        print(f'[Export] Error exporting intensity PNG: {e}')
                        import traceback
                        traceback.print_exc()
                if 'lifetime' in image_dict and isinstance(image_dict['lifetime'], np.ndarray):
                    try:
                        lifetime = image_dict['lifetime']
                        print(f'[Export] Lifetime shape: {lifetime.shape}')
                        h, w = lifetime.shape[:2]
                        fig = plt.figure(figsize=(w/100, h/100), dpi=100, facecolor='black', edgecolor='black')
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.set_facecolor('black')
                        im = ax.imshow(lifetime, cmap='viridis', origin='upper', aspect='auto')
                        ax.axis('off')
                        if with_scalebar:
                            self._draw_scale_bar(ax, w, h, pixel_size_um)
                        if with_annotations and self._fov_preview._roi_manager.get_all_regions():
                            from flimkit.UI.roi_tools import get_rectangle_patch, get_ellipse_patch, get_polygon_patch
                            for region in self._fov_preview._roi_manager.get_all_regions():
                                region_id = region['id']
                                tool_type = region['tool']
                                coords = region['coords']
                                color = self._fov_preview._roi_manager.get_color(region_id)
                                try:
                                    if tool_type == 'rect':
                                        patch = get_rectangle_patch(coords, edgecolor=color, linewidth=2)
                                    elif tool_type == 'ellipse':
                                        patch = get_ellipse_patch(coords, edgecolor=color, linewidth=2)
                                    elif tool_type in ('polygon', 'freehand'):
                                        patch = get_polygon_patch(coords, edgecolor=color, linewidth=2)
                                    else:
                                        continue
                                    ax.add_patch(patch)
                                except Exception as e:
                                    print(f'[Export] Could not add ROI {region_id}: {e}')
                        output_file = output_path / 'lifetime.png'
                        fig.savefig(output_file, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
                        plt.close(fig)
                        print(f'✓ Exported PNG lifetime: {output_file.name} ({lifetime.shape})')
                        exported_count += 1
                    except Exception as e:
                        print(f'[Export] Error exporting lifetime PNG: {e}')
                        import traceback
                        traceback.print_exc()
            try:
                if self._fov_preview is not None and hasattr(self._fov_preview, '_ax_decay'):
                    ax_decay = self._fov_preview._ax_decay
                    fig, ax = plt.subplots(figsize=(14, 8), dpi=150, facecolor='white', edgecolor='white')
                    ax.set_facecolor('white')
                    for line in ax_decay.get_lines():
                        ax.plot(line.get_xdata(), line.get_ydata(),
                               color=line.get_color(), linewidth=2.5,
                               marker=line.get_marker(), markersize=line.get_markersize(),
                               label=line.get_label(), alpha=line.get_alpha())
                    ax.set_yscale('log')
                    ax.set_title('Summed Decay - Measured, IRF, and Fitted', fontsize=16, fontweight='bold', color='black')
                    ax.set_xlabel('Time (ns)', fontsize=13, color='black')
                    ax.set_ylabel('Photon Count', fontsize=13, color='black')
                    ax.tick_params(colors='black')
                    ax.legend(fontsize=12, loc='upper right', framealpha=0.9, labelcolor='black')
                    ax.grid(True, alpha=0.3, color='gray')
                    output_file = output_path / 'summed_decay.png'
                    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    print(f'✓ Exported decay plot: {output_file.name}')
                    exported_count += 1
            except Exception as e:
                print(f'[Export] Error exporting decay: {e}')
                import traceback
                traceback.print_exc()
            print(f'✓ Export complete: {exported_count} high-resolution images to {output_path}')
            try:
                import subprocess
                subprocess.Popen(['open', str(output_path)])
            except Exception as e:
                print(f'[Export] Could not open folder: {e}')
        except Exception as e:
            print(f'✗ Export images error: {e}')
            import traceback
            traceback.print_exc()

    def _get_pixel_size_um(self) -> 'float | None':
        try:
            from pathlib import Path
            if getattr(self, '_current_form', None) == 'stitch':
                xlif = self.sv_xlif.get().strip() if hasattr(self, 'sv_xlif') else ''
                if xlif:
                    from flimkit.utils.xml_utils import get_pixel_size
                    basename = getattr(self, 'sv_ptu_basename', None)
                    basename = basename.get().strip() if basename is not None else Path(xlif).stem
                    pixel_size_m, _ = get_pixel_size(Path(xlif), basename or Path(xlif).stem)
                    if pixel_size_m and pixel_size_m > 0:
                        return pixel_size_m * 1e6
            ptu_path = getattr(self._fov_preview, '_ptu_path', None)
            if ptu_path and Path(ptu_path).exists():
                from flimkit.formats import FLIMFile
                ptu = FLIMFile(str(ptu_path), verbose=False)
                pix_res = ptu.tags.get('ImgHdr_PixRes', 0)
                if pix_res and float(pix_res) > 0:
                    return float(pix_res) * 1e6
        except Exception as e:
            print(f'[Export] Could not determine pixel size: {e}')
        return None

    @staticmethod
    def _draw_scale_bar(ax, img_w_px: int, img_h_px: int, pixel_size_um: float):
        from matplotlib.patches import FancyBboxPatch
        fov_um = img_w_px * pixel_size_um
        target = fov_um * 0.20
        nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        bar_um = nice[0]
        for n in nice:
            if n <= target:
                bar_um = n
            else:
                break
        bar_px = bar_um / pixel_size_um
        bar_h = max(3, img_h_px * 0.015)
        margin_x = img_w_px * 0.03
        margin_y = img_h_px * 0.03
        x0 = img_w_px - margin_x - bar_px
        y0 = img_h_px - margin_y - bar_h
        label = f'{bar_um} µm'
        fontsize = max(7, min(14, img_h_px * 0.035))
        pad_x = bar_px * 0.08
        pad_y = fontsize * 1.8
        bg = FancyBboxPatch(
            (x0 - pad_x, y0 - pad_y),
            bar_px + 2 * pad_x,
            bar_h + pad_y + margin_y * 0.5,
            boxstyle='round,pad=4',
            facecolor='black', edgecolor='none', alpha=0.55,
            zorder=9,
        )
        ax.add_patch(bg)
        from matplotlib.patches import Rectangle
        bar = Rectangle((x0, y0), bar_px, bar_h,
                         facecolor='white', edgecolor='none', zorder=10)
        ax.add_patch(bar)
        ax.text(x0 + bar_px / 2, y0 - fontsize * 0.35,
                label, color='white', fontsize=fontsize,
                ha='center', va='bottom', zorder=10)

    def _export_npz_fit(self, fit_result: dict, output_dir: str, ptu_path: str = None):
        try:
            from pathlib import Path
            import shutil
            npz_source = None
            if ptu_path:
                base_path = Path(ptu_path)
                if base_path.is_file():
                    npz_source = base_path.parent / f'{base_path.stem}.roi_fit.npz'
            if not npz_source or not npz_source.exists():
                source = fit_result.get('source')
                if isinstance(source, bytes):
                    source = source.decode('utf-8')
                if source:
                    base_path = Path(source)
                    if base_path.is_file():
                        npz_source = base_path.parent / f'{base_path.stem}.roi_fit.npz'
                    else:
                        npz_source = base_path / 'roi_fit.npz'
            if npz_source and npz_source.exists():
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                npz_dest = output_path / 'fit_result.npz'
                shutil.copy2(npz_source, npz_dest)
                print(f'✓ NPZ fit data copied to export: {npz_dest.name}')
            else:
                print(f'[Export] Could not find existing NPZ file to copy')
        except Exception as e:
            print(f'✗ NPZ export error: {e}')
            import traceback
            traceback.print_exc()

    def _build_fov_tab(self):
        from flimkit.UI.modes.fov_mode import FovMode
        FovMode(self).build()

    def _build_stitch_tab(self):
        from flimkit.UI.modes.stitch_mode import StitchMode
        StitchMode(self).build()

    def _build_stitch_fit(self, parent):
        from flimkit.UI.modes.stitch_mode import StitchMode
        StitchMode(self).build_fit(parent)

    def _apply_expert_overrides(self, a):
        ex = self._expert_overrides
        if not ex:
            return
        if 'optimizer' in ex:
            a.optimizer = ex['optimizer']
        if 'de_population' in ex:
            a.de_population = ex['de_population']
        if 'de_maxiter' in ex:
            a.de_maxiter = ex['de_maxiter']
        if 'lm_restarts' in ex:
            a.restarts = ex['lm_restarts']
        if 'binning_factor' in ex:
            a.binning = ex['binning_factor']
        if 'n_workers' in ex:
            a.workers = ex['n_workers']
        if 'min_photons' in ex:
            a.min_photons = ex['min_photons']
        if 'cost_function' in ex:
            a.cost_function = ex['cost_function']
        if 'channels' in ex:
            a.channel = ex['channels']
        if 'irf_fwhm' in ex and ex['irf_fwhm'] is not None:
            a.irf_fwhm = ex['irf_fwhm']
        if 'irf_align' in ex:
            a.irf_align = ex['irf_align']
        if 'irf_shift_bins' in ex:
            a.irf_shift_bins = ex['irf_shift_bins']
        if 'free_tau_perpixel' in ex:
            a.free_tau_perpixel = ex['free_tau_perpixel']
        if 'align_irf' in ex:
            a.align_irf = ex['align_irf']
        if 'fit_start_ns' in ex:
            a.fit_start_ns = ex['fit_start_ns']
        if 'fit_end_ns' in ex:
            a.fit_end_ns = ex['fit_end_ns']
        if 'exclude_ns' in ex:
            a.exclude_ns = ex['exclude_ns'] or None
        if 'fit_t0' in ex:
            a.fit_t0 = ex['fit_t0']

    def _open_expert_settings(self):
        from flimkit.utils.config_manager import cfg
        saved = cfg.get_section('expert')
        merged = dict(saved)
        merged.update(self._expert_overrides)
        dlg = ExpertSettingsDialog(self.root, merged)
        self.root.wait_window(dlg)
        if dlg.result is not None:
            is_default = all(
                dlg.result.get(k) == v for k, v in _EXPERT_DEFAULTS.items()
            )
            if is_default:
                self._expert_overrides = {}
            else:
                self._expert_overrides = dlg.result
            cfg.update_section('expert', dlg.result)
            if hasattr(self, '_proj_browser') and self._proj_browser and self._proj_browser._project:
                self._proj_browser._project.config['expert'] = dlg.result
                self._proj_browser._project.save()
                cfg.load_project_overrides(self._proj_browser._project.config)
            self._update_expert_banners()

    def _update_expert_banners(self):
        active = bool(self._expert_overrides)
        for banner in (self._expert_banner_fov, self._expert_banner_st,
                       self._expert_banner_batch):
            if active:
                banner.grid()
            else:
                banner.grid_remove()

    def _pipeline_changed(self):
        mode = self.sv_pipeline.get()
        if mode == 'stitch_only':
            self._fit_frame.grid_remove()
            self._btn_st.configure(text='▶  Run Tile Stitch')
            self._btn_expert_st.pack_forget()
            self._expert_banner_st.grid_remove()
        elif mode == 'stitch_fit':
            self._fit_frame.grid()
            self._tile_extras_frame.grid_remove()
            self._btn_st.configure(text='▶  Run Stitch + Fit')
            self._btn_expert_st.pack(side='left', padx=4, before=self._btn_st)
            self._update_expert_banners()
        elif mode == 'series_fit':
            self._fit_frame.grid()
            self._tile_extras_frame.grid_remove()
            if hasattr(self, '_series_frame'):
                self._series_frame.grid()
            self._btn_st.configure(text='▶  Run Series Fit')
            self._btn_expert_st.pack(side='left', padx=4, before=self._btn_st)
            self._update_expert_banners()
        else:
            self._fit_frame.grid()
            self._tile_extras_frame.grid()
            self._btn_st.configure(text='▶  Run Per-Tile Fit')
            self._btn_expert_st.pack(side='left', padx=4, before=self._btn_st)
            self._update_expert_banners()
        if mode != 'series_fit' and hasattr(self, '_series_frame'):
            self._series_frame.grid_remove()
        self._update_form_scrollbar('stitch')
        self.root.after_idle(self._fit_window_to_screen)

    def _perpix_toggled(self):
        if self.bv_perpix.get():
            self._pxf.grid()
        else:
            self._pxf.grid_remove()
        self._update_form_scrollbar('stitch')
        self.root.after_idle(self._fit_window_to_screen)

    def _update_form_scrollbar(self, form_id: str):
        if form_id not in self._form_inner_frames:
            return
        try:
            outer, inner = self._form_inner_frames[form_id]
            if not hasattr(outer, '_canvas'):
                return
            canvas = outer._canvas
            window_id = outer._window_id

            def _refresh():
                try:
                    inner.update()
                except Exception:
                    pass
                bbox = canvas.bbox('all')
                if bbox:
                    canvas.configure(scrollregion=bbox)
                new_h = inner.winfo_reqheight()
                canvas_h = canvas.winfo_height()
                target_h = max(new_h, canvas_h if canvas_h > 1 else 0)
                if target_h > 0:
                    canvas.itemconfigure(window_id, height=target_h)
                cw = canvas.winfo_width()
                if cw > 1:
                    canvas.itemconfigure(window_id, width=cw)
            self.root.after_idle(lambda: self.root.after(80, _refresh))
        except Exception as e:
            print(f'[Scrollbar] {form_id}: {e}')

    def _build_batch_tab(self):
        from flimkit.UI.modes.batch_mode import BatchMode
        BatchMode(self).build()

    def _batch_mode_changed(self):
        mode = self.sv_batch_mode.get()
        labels = {
            'tiled': 'Mode: Multi-Tile ROI Fit',
            'fov': 'Mode: Single FOV Fit',
            'timelapse': 'Mode: Timelapse Fit',
        }
        self._batch_mode_label.configure(text=labels.get(mode, ''))
        if mode == 'tiled':
            self._batch_xlif_fr.grid()
            self._batch_freg.grid()
        else:
            self._batch_xlif_fr.grid_remove()
            self._batch_freg.grid_remove()
        if mode == 'timelapse':
            self._batch_tl_fr.grid()
        else:
            self._batch_tl_fr.grid_remove()
        btn_texts = {
            'tiled': '▶  Run Batch ROI Fit',
            'fov': '▶  Run Batch FOV Fit',
            'timelapse': '▶  Run Timelapse Fit',
        }
        self._btn_batch.configure(text=btn_texts.get(mode, '▶  Run'))
        help_texts = {
            'tiled': 'One sub-folder per ROI created inside the output base dir.',
            'fov': 'One sub-folder per PTU file created inside the output base dir.',
            'timelapse': 'One sub-folder per (region, series, z) group inside the output base dir.',
        }
        self._batch_io_help.configure(text=help_texts.get(mode, ''))

    def _dispatch_batch(self):
        mode = self.sv_batch_mode.get()
        if mode == 'tiled':
            self._run_batch()
        elif mode == 'timelapse':
            self._run_timelapse_batch()
        else:
            self._run_batch_fov()

    def _resolve_batch_out_dir(self, ptu_dir: str) -> str:
        if self.bv_batch_save_beside.get():
            out = str(Path(ptu_dir) / 'save')
            Path(out).mkdir(parents=True, exist_ok=True)
            return out
        return self.sv_batch_out_dir.get().strip()

    def _run_timelapse_batch(self):
        ptu_dir = self.sv_batch_ptu_dir.get().strip()
        if not ptu_dir or not Path(ptu_dir).is_dir():
            messagebox.showerror('Missing input', 'Please select a valid PTU folder.')
            return
        out_dir = self._resolve_batch_out_dir(ptu_dir)
        if not out_dir:
            messagebox.showerror('Missing input', 'Please specify an output directory.')
            return
        from flimkit.utils.batch_fit import group_timelapse_files
        groups = group_timelapse_files(ptu_dir)
        if not groups:
            messagebox.showerror(
                'No timelapse PTUs',
                f'No files matching region_tX[_sY][_zZ].ptu found in:\n{ptu_dir}'
            )
            return
        cfg = _C()
        mirf = self.sv_batch_mirf.get().strip() or str(cfg['MACHINE_IRF_DEFAULT_PATH'])
        n_exp = self.iv_nexp_batch.get()
        tau_min = float(self.sv_batch_tau_min.get() or cfg['Tau_min'])
        tau_max = float(self.sv_batch_tau_max.get() or cfg['Tau_max'])
        thr = _thresh(self.bv_batch_thr, self.sv_batch_thr)
        correct_pileup = self.bv_batch_correct_pileup.get()
        save_stack = self.bv_batch_save_stack.get()
        save_lifetime = self.bv_batch_save_lifetime.get()
        save_rgb = self.bv_batch_save_rgb.get()
        save_intensity = self.bv_batch_save_intensity.get()
        save_npy = self.bv_batch_save_npy.get()
        save_ind = self.bv_batch_save_ind.get()
        tau_lo = _flt(self.sv_batch_tau_lo) or cfg['TAU_DISPLAY_MIN']
        tau_hi = _flt(self.sv_batch_tau_hi) or cfg['TAU_DISPLAY_MAX']
        gamma = float(self.sv_batch_gamma.get() or 0.4)
        int_max = _flt(self.sv_batch_int_max) or None
        pool_positions = self.bv_tl_pool_positions.get()
        compute_bound_fraction = self.bv_tl_bound_fraction.get()
        expert_overrides = dict(self._expert_overrides)
        ref_tau1 = ref_tau2 = ref_tau3 = None
        if self.bv_tl_fix_tau.get():
            _raw = [self.sv_tl_tau1.get().strip(),
                    self.sv_tl_tau2.get().strip(),
                    self.sv_tl_tau3.get().strip()][:n_exp]
            _parsed = []
            for i, _t in enumerate(_raw):
                if not _t:
                    _parsed.append(None)
                    continue
                try:
                    _parsed.append(float(_t))
                except ValueError:
                    messagebox.showerror('Bad value', f"τ{i+1} must be a number (got '{_t}').")
                    return
            if any(v is None for v in _parsed):
                messagebox.showerror(
                    'Missing τ',
                    f'Fix reference τ is on - please enter all {n_exp} τ value(s), '
                    'or untick it to fit τ from the pooled decay.')
                return
            ref_tau1, ref_tau2, ref_tau3 = (_parsed + [None, None, None])[:3]
        n_groups = len(groups)
        n_frames = sum(len(v) for v in groups.values())
        from flimkit.FLIM.batch import fit_timelapse

        def task(progress_callback, cancel_event):
            a = argparse.Namespace(
                nexp=n_exp,
                tau_min=tau_min,
                tau_max=tau_max,
                estimate_irf='machine_irf',
                machine_irf=mirf,
                irf_fwhm=expert_overrides.get('irf_fwhm', cfg['IRF_FWHM']),
                irf_bins=cfg['IRF_BINS'],
                irf_fit_width=cfg['IRF_FIT_WIDTH'],
                optimizer=expert_overrides.get('optimizer', 'de'),
                restarts=expert_overrides.get('lm_restarts', cfg['lm_restarts']),
                de_population=expert_overrides.get('de_population', cfg['de_population']),
                de_maxiter=expert_overrides.get('de_maxiter', cfg['de_maxiter']),
                workers=expert_overrides.get('n_workers', cfg['n_workers']),
                no_polish=False,
                channel=expert_overrides.get('channels', cfg['channels']),
                min_photons=expert_overrides.get('min_photons', cfg['MIN_PHOTONS_PERPIX']),
                correct_pileup=correct_pileup,
                cost_function='poisson',
                save_stack=save_stack,
                save_lifetime=save_lifetime,
                save_rgb=save_rgb,
                save_intensity=save_intensity,
                save_npy=save_npy,
                save_ind=save_ind,
                tau_display_min=tau_lo,
                tau_display_max=tau_hi,
                gamma=gamma,
                intensity_display_max=int_max,
                no_plots=False,
                intensity_threshold=thr,
            )
            return fit_timelapse(
                ptu_dir=ptu_dir,
                output_dir=out_dir,
                args=a,
                ref_tau1_ns=ref_tau1,
                ref_tau2_ns=ref_tau2,
                ref_tau3_ns=ref_tau3,
                channel=expert_overrides.get('channels', cfg['channels']),
                pool_positions=pool_positions,
                compute_bound_fraction=compute_bound_fraction,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

        def on_done(result):
            self._set_buttons('normal')
            self._res.set_status(f'✓  Timelapse complete - {n_groups} group(s), {n_frames} frames.')
            self._res.load_images(out_dir)
        self._set_buttons('disabled')
        self.run_with_progress(
            task,
            task_name=f'Timelapse Fit ({n_groups} group(s), {n_frames} frames)',
            on_done=on_done,
            output_dir=out_dir,
        )

    def _run_batch_fov(self):
        ptu_dir = self.sv_batch_ptu_dir.get().strip()
        if not ptu_dir or not Path(ptu_dir).is_dir():
            messagebox.showerror('Missing input', 'Please select a valid PTU folder.')
            return
        out_dir = self._resolve_batch_out_dir(ptu_dir)
        if not out_dir:
            messagebox.showerror('Missing input', 'Please specify an output directory.')
            return
        from flimkit.formats import supported_extensions
        _globs = tuple('*' + e for e in supported_extensions())
        ptu_files = sorted(p for ext in _globs for p in Path(ptu_dir).glob(ext))
        if not ptu_files:
            messagebox.showerror('No FLIM files', f'No FLIM files found in:\n{ptu_dir}')
            return
        cfg = _C()
        mirf = self.sv_batch_mirf.get().strip() or str(cfg['MACHINE_IRF_DEFAULT_PATH'])
        _batch_model = self.sv_fit_model_batch.get()
        n_exp = self.iv_nexp_batch.get() if _batch_model in ('discrete', 'tail') else 2
        tau_min = float(self.sv_batch_tau_min.get() or cfg['Tau_min'])
        tau_max = float(self.sv_batch_tau_max.get() or cfg['Tau_max'])
        tau_lo = _flt(self.sv_batch_tau_lo) or cfg['TAU_DISPLAY_MIN'] or 0.0
        tau_hi = _flt(self.sv_batch_tau_hi) or cfg['TAU_DISPLAY_MAX'] or 10.0
        save_npy = self.bv_batch_save_npy.get()
        thr = _thresh(self.bv_batch_thr, self.sv_batch_thr)
        correct_pileup = self.bv_batch_correct_pileup.get()
        tvb_ptu_batch = (self.sv_batch_tvb_ptu.get().strip() or None) if hasattr(self, 'sv_batch_tvb_ptu') else None
        expert_overrides = dict(self._expert_overrides)
        from flimkit.interactive import _run_flim_fit
        import gc, csv as csv_mod

        def task(progress_callback, cancel_event):
            csv_path = Path(out_dir) / 'batch_fov_fit_summary.csv'
            header_written = False
            n_total = len(ptu_files)
            for idx, ptu_path in enumerate(ptu_files):
                if cancel_event.is_set():
                    print('\nBatch cancelled.')
                    break
                progress_callback(idx, n_total)
                stem = ptu_path.stem
                fov_out = Path(out_dir) / stem.replace(' ', '_')
                fov_out.mkdir(parents=True, exist_ok=True)
                print(f'\n{'='*50}\n  [{idx+1}/{n_total}] {stem}\n{'='*50}')
                try:
                    a = argparse.Namespace(
                        ptu=str(ptu_path),
                        xlsx=None,
                        debug_xlsx=False,
                        print_config=False,
                        irf=None,
                        irf_xlsx=None,
                        estimate_irf='machine_irf',
                        no_xlsx_irf=True,
                        machine_irf=mirf,
                        irf_bins=cfg['IRF_BINS'],
                        irf_fit_width=cfg['IRF_FIT_WIDTH'],
                        irf_fwhm=expert_overrides.get('irf_fwhm', cfg['IRF_FWHM']),
                        nexp=n_exp,
                        dist_type=_batch_model,
                        dist_n_components=(self.iv_ncomp_dist_batch.get() if _batch_model not in ('discrete', 'tail') else 1),
                        fit_t0=expert_overrides.get('fit_t0', False),
                        tau_min=tau_min,
                        tau_max=tau_max,
                        mode='both',
                        binning=expert_overrides.get('binning_factor', cfg['binning_factor']),
                        min_photons=expert_overrides.get('min_photons', cfg['MIN_PHOTONS_PERPIX']),
                        optimizer=expert_overrides.get('optimizer', cfg['Optimizer']),
                        restarts=expert_overrides.get('lm_restarts', cfg['lm_restarts']),
                        de_population=expert_overrides.get('de_population', cfg['de_population']),
                        de_maxiter=expert_overrides.get('de_maxiter', cfg['de_maxiter']),
                        workers=expert_overrides.get('n_workers', cfg['n_workers']),
                        no_polish=False,
                        channel=expert_overrides.get('channels', cfg['channels']),
                        out=str(fov_out / stem),
                        no_plots=True,
                        cell_mask=False,
                        correct_pileup=correct_pileup,
                        intensity_threshold=thr,
                        irf_align=expert_overrides.get('irf_align', 'steepest_rise'),
                        irf_shift_bins=expert_overrides.get('irf_shift_bins', 2),
                        tvb_ptu=tvb_ptu_batch,
                        tvb_channel=None,
                    )
                    result = _run_flim_fit(a)
                    summary = result.get('global_summary', {})
                    pu_pct = result.get('pileup_pct')
                    cr_mhz = result.get('count_rate_mhz')
                    if not save_npy:
                        for f_ in fov_out.glob('*.npy'):
                            if not f_.name.endswith('_time_axis_ns.npy'):
                                f_.unlink(missing_ok=True)
                    gc.collect()
                    row = {'fov': stem, 'status': 'OK',
                           'pileup_pct': pu_pct if pu_pct is not None else '',
                           'count_rate_mhz': cr_mhz if cr_mhz is not None else '',
                           **summary}
                    print(f'  OK: {stem}')
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    row = {'fov': stem,
                           'status': f'ERROR: {type(exc).__name__}: {str(exc)[:80]}'}
                with open(csv_path, 'a', newline='') as fh:
                    writer = csv_mod.DictWriter(fh, fieldnames=list(row.keys()))
                    if not header_written:
                        writer.writeheader(); header_written = True
                    writer.writerow(row)
            progress_callback(n_total, n_total)
            print(f'\nBatch FOV complete. CSV: {csv_path}')

        def on_done(result):
            self._set_buttons('normal')
            self._res.set_status('✓  Batch FOV complete.')
            self._res.load_images(out_dir)
        self._set_buttons('disabled')
        self.run_with_progress(
            task, task_name=f'Batch FOV Fit ({len(ptu_files)} PTUs)',
            on_done=on_done, output_dir=out_dir)

    def _run_batch(self):
        xlif_dir = self.sv_batch_xlif_dir.get().strip()
        ptu_dir = self.sv_batch_ptu_dir.get().strip()
        for val, name in [(xlif_dir, 'XLIF folder'), (ptu_dir, 'PTU folder')]:
            if not val or not Path(val).is_dir():
                messagebox.showerror('Missing input', f'Please select a valid {name}.')
                return
        out_dir = self._resolve_batch_out_dir(ptu_dir)
        if not out_dir:
            messagebox.showerror('Missing input', 'Please specify an output directory.')
            return
        xlif_files = sorted(Path(xlif_dir).glob('*.xlif'))
        if not xlif_files:
            messagebox.showerror('No XLIF files', f'No .xlif files found in:\n{xlif_dir}')
            return
        cfg = _C()
        mirf = self.sv_batch_mirf.get().strip() or str(cfg['MACHINE_IRF_DEFAULT_PATH'])
        _batch_model = self.sv_fit_model_batch.get()
        n_exp = self.iv_nexp_batch.get() if _batch_model in ('discrete', 'tail') else 2
        tau_min = float(self.sv_batch_tau_min.get() or cfg['Tau_min'])
        tau_max = float(self.sv_batch_tau_max.get() or cfg['Tau_max'])
        tau_lo = _flt(self.sv_batch_tau_lo) or cfg['TAU_DISPLAY_MIN'] or 0.0
        tau_hi = _flt(self.sv_batch_tau_hi) or cfg['TAU_DISPLAY_MAX'] or 10.0
        save_lifetime = self.bv_batch_save_lifetime.get()
        save_rgb = self.bv_batch_save_rgb.get()
        save_npy = self.bv_batch_save_npy.get()
        save_ind = self.bv_batch_save_ind.get()
        gamma = float(self.sv_batch_gamma.get() or 0.4)
        int_max = _flt(self.sv_batch_int_max) or None
        tau_weighting = self.sv_batch_tau_weighting.get()
        tau_key = 'tau_mean_int' if tau_weighting == 'intensity' else 'tau_mean_amp'
        register = self.bv_batch_register.get()
        reg_shift = int(self.sv_batch_reg_shift.get() or 120)
        thr = _thresh(self.bv_batch_thr, self.sv_batch_thr)
        correct_pileup = self.bv_batch_correct_pileup.get()
        tvb_ptu_batch = (self.sv_batch_tvb_ptu.get().strip() or None) if hasattr(self, 'sv_batch_tvb_ptu') else None
        expert_overrides = dict(self._expert_overrides)
        from flimkit.formats.PTU.stitch import fit_flim_tiles
        from flimkit.FLIM.assemble import (derive_global_tau, save_assembled_maps,
                                           assemble_tile_maps)
        from flimkit.utils.lifetime_image import make_lifetime_image, make_component_rgb_tiff
        import gc, csv as csv_mod

        def task(progress_callback, cancel_event):
            csv_path = Path(out_dir) / 'batch_roi_fit_summary.csv'
            header_written = False
            n_total = len(xlif_files)
            for idx, xlif_path in enumerate(xlif_files):
                if cancel_event.is_set():
                    print('\nBatch cancelled.')
                    break
                progress_callback(idx, n_total)
                ptu_basename = xlif_path.stem
                roi_clean = ptu_basename.replace(' ', '_')
                roi_out = Path(out_dir) / roi_clean
                roi_out.mkdir(parents=True, exist_ok=True)
                print(f'\n{'='*50}\n  [{idx+1}/{n_total}] {ptu_basename}\n{'='*50}')
                try:
                    fit_args = argparse.Namespace(
                        nexp=n_exp, dist_type=_batch_model,
                        dist_n_components=(self.iv_ncomp_dist_batch.get() if _batch_model not in ('discrete', 'tail') else 1),
                        fit_t0=expert_overrides.get('fit_t0', False),
                        tau_min=tau_min, tau_max=tau_max,
                        optimizer=expert_overrides.get('optimizer', 'de'),
                        restarts=expert_overrides.get('lm_restarts', 1),
                        de_population=expert_overrides.get('de_population', cfg['de_population']),
                        de_maxiter=expert_overrides.get('de_maxiter', cfg['de_maxiter']),
                        workers=expert_overrides.get('n_workers', cfg['n_workers']),
                        binning=expert_overrides.get('binning_factor', 1),
                        min_photons=expert_overrides.get('min_photons', cfg['MIN_PHOTONS_PERPIX']),
                        channel=expert_overrides.get('channels', cfg['channels']),
                        irf_fwhm=expert_overrides.get('irf_fwhm', cfg['IRF_FWHM']),
                        intensity_threshold=thr,
                        correct_pileup=correct_pileup,
                        register_tiles=register,
                        reg_max_shift_px=reg_shift,
                        machine_irf=mirf,
                        tau_display_min=tau_lo,
                        tau_display_max=tau_hi,
                        intensity_display_min=0.0,
                        intensity_display_max=None,
                        irf_xlsx_dir=None, irf_xlsx_map=None,
                        ptu_basename=ptu_basename,
                        xlif=str(xlif_path),
                        ptu_dir=ptu_dir,
                        output_dir=str(roi_out),
                        no_plots=True, cell_mask=False,
                        debug_xlsx=False, print_config=False,
                        irf=None, irf_xlsx=None,
                        estimate_irf='machine_irf',
                        no_xlsx_irf=True,
                        tvb_ptu=tvb_ptu_batch,
                        tvb_channel=None,
                    )
                    (tile_results, canvas_h, canvas_w, _, _, _, _, _, _) = fit_flim_tiles(
                        xlif_path=xlif_path, ptu_dir=Path(ptu_dir),
                        output_dir=roi_out, args=fit_args,
                        ptu_basename=ptu_basename, rotate_tiles=True, verbose=True,
                    )
                    if not tile_results:
                        row = {'roi': ptu_basename, 'status': 'No tiles fitted'}
                    else:
                        canvas = assemble_tile_maps(tile_results, canvas_h, canvas_w, n_exp)
                        del tile_results; gc.collect()
                        summary = derive_global_tau(canvas, n_exp=n_exp)
                        save_assembled_maps(
                            canvas=canvas, global_summary=summary,
                            output_dir=roi_out, roi_name=roi_clean, n_exp=n_exp,
                            tau_display_min=tau_lo, tau_display_max=tau_hi,
                            intensity_display_max=int_max,
                            tau_weighting=('int' if tau_weighting == 'intensity' else 'amp'),
                        )
                        if save_lifetime:
                            make_lifetime_image(
                                canvas=canvas, output_dir=roi_out, roi_name=roi_clean,
                                tau_min_ns=tau_lo, tau_max_ns=tau_hi,
                                smooth_sigma_px=0.0, gamma=gamma, verbose=False,
                                tau_key=tau_key,
                            )
                        if save_rgb:
                            make_component_rgb_tiff(
                                canvas=canvas, output_dir=roi_out,
                                roi_name=roi_clean, n_exp=n_exp, verbose=False,
                            )
                        if not save_npy:
                            for f_ in roi_out.glob('*.npy'):
                                if not f_.name.endswith('_time_axis_ns.npy'):
                                    f_.unlink(missing_ok=True)
                        del canvas; gc.collect()
                        row = {'roi': ptu_basename, 'status': 'OK', **summary}
                        print(f'  OK: {ptu_basename}')
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    row = {'roi': ptu_basename,
                           'status': f'ERROR: {type(exc).__name__}: {str(exc)[:80]}'}
                with open(csv_path, 'a', newline='') as fh:
                    writer = csv_mod.DictWriter(fh, fieldnames=list(row.keys()))
                    if not header_written:
                        writer.writeheader(); header_written = True
                    writer.writerow(row)
            progress_callback(n_total, n_total)
            print(f'\nBatch complete. CSV: {csv_path}')

        def on_done(result):
            self._set_buttons('normal')
            self._res.set_status('✓  Batch complete.')
            self._res.load_images(out_dir)
        self._set_buttons('disabled')
        self.run_with_progress(
            task, task_name=f'Batch ROI Fit ({len(xlif_files)} ROIs)', on_done=on_done, output_dir=out_dir)

    def _build_machine_irf_tab(self):
        from flimkit.UI.modes.irf_mode import IrfMode
        IrfMode(self).build()

    def _build_phasor_tab(self):
        from flimkit.UI.modes.phasor_mode import PhasorMode
        PhasorMode(self).build()

    def _ph_mode_changed(self):
        if self.sv_ph_mode.get() == 'new':
            self._ph_new.grid()
            self._ph_sess.grid_remove()
        else:
            self._ph_new.grid_remove()
            self._ph_sess.grid()

    def _run_ph_find_peaks(self):
        panel = self._phasor_panel
        if panel._real is None:
            messagebox.showwarning('No data', 'Load a PTU file first.')
            return
        try:
            sigma = float(self.sv_ph_pk_sigma.get() or 3.0)
            thresh = float(self.sv_ph_pk_thresh.get() or 0.10)
            min_ph = float(self.sv_ph_minph.get() or 0.01)
        except ValueError:
            messagebox.showerror('Invalid input',
                                 'Sigma and threshold must be numeric.')
            return
        from flimkit.phasor.peaks import find_phasor_peaks, print_peaks
        peaks = find_phasor_peaks(
            panel._real, panel._imag, panel._mean, panel._freq,
            min_photons=min_ph, sigma=sigma, threshold_frac=thresh)
        print_peaks(peaks)
        panel.overlay_peaks(peaks)
        self._res.set_status(
            f'✓  Found {peaks['n_peaks']} peak(s) in phasor histogram.')

    def _run_ph_fret_overlay(self):
        panel = self._phasor_panel
        if panel._real is None:
            messagebox.showwarning('No data', 'Load a PTU file first.')
            return
        try:
            tau_d = float(self.sv_ph_fret_taud.get())
            taua_str = self.sv_ph_fret_taua.get().strip()
            tau_a = float(taua_str) if taua_str else None
            fretting = float(self.sv_ph_fret_fretting.get() or 1.0)
        except ValueError:
            messagebox.showerror('Invalid input', 'Lifetimes must be numeric.')
            return
        from flimkit.phasor.fret import predict_fret_trajectory
        traj = predict_fret_trajectory(
            panel._freq, tau_d,
            acceptor_lifetime=tau_a,
            donor_fretting=fretting)
        panel.overlay_fret_trajectory(traj)
        label = f'τ_D={tau_d} ns'
        if tau_a:
            label += f', τ_A={tau_a} ns'
        self._res.set_status(f'✓  FRET trajectory overlaid  ({label}).')

    def _run_ph_fit_fret(self):
        panel = self._phasor_panel
        if panel._real is None:
            messagebox.showwarning('No data', 'Load a PTU file first.')
            return
        try:
            tau_d = float(self.sv_ph_fret_taud.get())
            fretting = float(self.sv_ph_fret_fretting.get() or 1.0)
            min_ph = float(self.sv_ph_minph.get() or 0.01)
        except ValueError:
            messagebox.showerror('Invalid input', 'Lifetimes must be numeric.')
            return
        from flimkit.phasor.fret import (
            FRETChannelData, FRETModelParameters, fit_donor_fret)
        donor = FRETChannelData(
            panel._real, panel._imag, panel._mean,
            panel._freq, min_photons=min_ph)
        params = FRETModelParameters(
            donor_lifetime=tau_d,
            donor_fretting=fretting)
        try:
            result = fit_donor_fret(donor, params)
            result.print_summary()
            self._res.set_status(
                f'✓  FRET fit: E={result.fret_efficiency:.3f}  '
                f'fretting={result.donor_fretting:.3f}')
        except Exception as exc:
            messagebox.showerror('FRET fit failed', str(exc))

    def _on_fov_ptu_changed(self, var, index, mode):
        ptu_path = self.sv_ptu.get().strip()
        if not ptu_path:
            return
        if ptu_path == self._last_loaded_ptu:
            return
        if getattr(self, '_loading_ptu', False):
            return
        self._last_loaded_ptu = ptu_path
        import os
        from flimkit.formats import file_modality
        if os.path.exists(ptu_path):
            modality = file_modality(ptu_path)
            if modality != 'time':
                msgs = {
                    'frequency': 'ISS .ifli is frequency-domain (FD-FLIM) data. It has no decay to fit; open it in the Phasor tab instead.',
                    'intensity': 'ISS .ifi is an intensity image with no lifetime data; FOV loading for it is not wired up yet.',
                }
                messagebox.showinfo('Not supported yet',
                                    msgs.get(modality, f'Unsupported file for FOV fitting:\n{ptu_path}'))
                return
        self._loading_ptu = True
        self._add_to_recent(ptu_path, 'file')
        if hasattr(self, 'sv_out_fov'):
            self.sv_out_fov.set(Path(ptu_path).stem)

        def load():
            try:
                self._fov_preview.load_fov(ptu_path)
                self._auto_load_session_for_ptu(ptu_path)
            finally:
                self._loading_ptu = False
        if self._ptu_after_id is not None:
            self.root.after_cancel(self._ptu_after_id)
        self._ptu_after_id = self.root.after(100, load)

    def _cancel_pending_scan_loads(self):
        if self._ptu_after_id is not None:
            self.root.after_cancel(self._ptu_after_id)
            self._ptu_after_id = None
        if self._xlif_after_id is not None:
            self.root.after_cancel(self._xlif_after_id)
            self._xlif_after_id = None
        self._loading_ptu = False
        self._loading_xlif = False

    def _on_xlif_changed(self, var, index, mode):
        xlif_path = self.sv_xlif.get().strip()
        if not xlif_path:
            return
        if xlif_path == self._last_loaded_xlif:
            return
        if getattr(self, '_loading_xlif', False):
            return
        self._last_loaded_xlif = xlif_path
        self._loading_xlif = True
        stem = Path(xlif_path).stem
        output_dir = None
        session_file = None
        if hasattr(self, '_proj_browser') and self._proj_browser._project:
            rec = self._proj_browser._project.scans.get(stem)
            if rec and rec.out_st:
                out_path = Path(rec.out_st)
                roi_name = stem.replace(' ', '_')
                if out_path.name == roi_name:
                    output_dir = str(out_path)
                else:
                    output_dir = str(out_path / roi_name)
                session_file = Path(output_dir) / 'roi_session.npz'
                if not session_file.exists():
                    session_file = None
        if output_dir is None:
            ptu_dir = self.sv_ptu_dir.get().strip()
            out_base = self.sv_out_st.get().strip()
            if ptu_dir and out_base:
                roi_name = stem.replace(' ', '_')
                output_dir = str(Path(out_base) / roi_name)
        if output_dir is None:
            self._loading_xlif = False
            return
        if self._xlif_after_id is not None:
            self.root.after_cancel(self._xlif_after_id)
        _sf = session_file
        def _do_xlif_load():
            try:
                self._fov_preview.load_stitched_roi(output_dir)
                if _sf and _sf.exists():
                    self._auto_load_session_for_stitch(output_dir)
            finally:
                self._loading_xlif = False
        self._xlif_after_id = self.root.after(100, _do_xlif_load)

    def _get_roi_fit_params(self) -> dict:
        cfg = _C()
        irf = self._irf_fov.get_args(xlsx_fallback=self.sv_xlsx.get().strip())
        params = {
            'ptu_path': self.sv_ptu.get().strip() or None,
            'n_exp': self.iv_nexp_fov.get(),
            'tau_min': float(self.sv_tau_min_fov.get() or cfg['Tau_min']),
            'tau_max': float(self.sv_tau_max_fov.get() or cfg['Tau_max']),
            'cost_function': cfg.get('cost_function', 'poisson'),
            'channel': cfg['channels'],
            'irf': irf['irf'],
            'irf_xlsx': irf['irf_xlsx'],
            'estimate_irf': irf['estimate_irf'],
            'machine_irf': irf.get('machine_irf') or str(cfg['MACHINE_IRF_DEFAULT_PATH']),
            'irf_bins': cfg['IRF_BINS'],
            'irf_fit_width': cfg['IRF_FIT_WIDTH'],
            'irf_fwhm': cfg['IRF_FWHM'],
            'irf_align': 'steepest_rise',
            'irf_shift_bins': 2,
        }
        expert = self._expert_overrides
        if expert:
            if 'cost_function' in expert:
                params['cost_function'] = expert['cost_function']
            if 'channels' in expert:
                params['channel'] = expert['channels']
        return params

    def _on_fov_analysis_changed(self):
        mode = self.sv_fov_analysis.get()
        if mode == 'zstack':
            self._fov_single_fr.grid_remove()
            self._fov_zstack_fr.grid()
            self._btn_fov.configure(text='▶  Run Z-stack Fit')
            self._on_zstack_dir_changed()
        else:
            self._fov_zstack_fr.grid_remove()
            self._fov_single_fr.grid()
            self._btn_fov.configure(text='▶  Run Single-FOV Fit')
            if self._fov_preview is not None:
                self._fov_preview._hide_zstack()

    def _on_zstack_dir_changed(self, *_):
        if getattr(self, 'sv_fov_analysis', None) is None \
                or self.sv_fov_analysis.get() != 'zstack':
            return
        d = self.sv_zstack_dir.get().strip()
        if not d or not Path(d).is_dir() or self._fov_preview is None:
            return
        from flimkit.utils.batch_fit import group_zstack_files
        try:
            groups = group_zstack_files(d)
        except Exception:
            return
        if not groups:
            return
        zslices = next(iter(groups.values()))
        descs = [{'z': z, 'ptu_path': str(p)} for z, p in sorted(zslices.items())]
        self._fov_preview.display_zstack(descs)

    def _run_fov(self):
        if getattr(self, 'sv_fov_analysis', None) is not None \
                and self.sv_fov_analysis.get() == 'zstack':
            self._run_zstack_fov()
            return
        ptu = self.sv_ptu.get().strip()
        if not ptu or not Path(ptu).exists():
            messagebox.showerror('Missing input', 'Please select a valid PTU file.')
            return
        from flimkit.interactive import _run_flim_fit
        a = self._controller.fov_args()
        out_dir = str(Path(a.out).parent)
        self._launch(
            lambda progress_callback=None, cancel_event=None: _run_flim_fit(a, progress_callback, cancel_event),
            output_dir=out_dir,
            ptu_path=ptu,
            task_name='Single-FOV Fit'
        )

    def _populate_zstack_summary_from_dir(self, group_dir):
        import json
        import numpy as np
        if self._res is None:
            return
        group_dir = Path(group_dir)
        ref = {}
        for rf in group_dir.glob('*_reference_fit.json'):
            try:
                ref = json.loads(rf.read_text())
            except Exception:
                ref = {}
            break
        rows = []
        for i, tau in enumerate(ref.get('taus_ns', [])):
            rows.append((f'τ{i+1} (reference)', f'{float(tau):.3f}', 'ns'))
        if ref.get('n_slices') is not None:
            rows.append(('Z-slices', str(ref['n_slices']), ''))
        if ref.get('total_pooled_photons') is not None:
            rows.append(('Pooled photons', f"{int(ref['total_pooled_photons']):,}", ''))
        zj = {}
        for zf in group_dir.glob('*_zseries.json'):
            try:
                zj = json.loads(zf.read_text())
            except Exception:
                zj = {}
            break
        def _mean(key):
            vals = [v for v in zj.get(key, []) if isinstance(v, (int, float)) and v == v]
            return float(np.mean(vals)) if vals else None
        tm = _mean('tau_mean_mean')
        if tm is not None:
            rows.append(('Mean τ (slices)', f'{tm:.3f}', 'ns'))
        npx = [v for v in zj.get('n_pixels_fitted', []) if isinstance(v, (int, float))]
        if npx:
            rows.append(('Pixels fitted', f'{int(sum(npx)):,}', ''))
        cm = _mean('chi2_r_mean')
        if cm is not None:
            rows.append(('Mean χ²_r', f'{cm:.3f}', ''))
        if rows:
            self._res.populate_summary(rows)

    def _run_zstack_fov(self):
        ptu_dir = self.sv_zstack_dir.get().strip()
        if not ptu_dir or not Path(ptu_dir).is_dir():
            messagebox.showerror('Missing input', 'Please select a valid z-stack folder.')
            return
        from flimkit.FLIM.batch import fit_zstack
        from flimkit.utils.batch_fit import group_zstack_files
        groups = group_zstack_files(ptu_dir)
        if not groups:
            messagebox.showerror(
                'No z-stack PTUs',
                f'No files matching region_zX.ptu found in:\n{ptu_dir}')
            return
        cfg = _C()
        params = self._get_roi_fit_params()
        _out_raw = self.sv_out_fov.get().strip() or 'flim_zstack_out'
        if Path(_out_raw).parent == Path('.'):
            out_dir = str(Path(ptu_dir) / _out_raw)
        else:
            out_dir = _out_raw
        expert = dict(self._expert_overrides)
        correct_pileup = self.bv_correct_pileup.get()
        n_stacks = len(groups)
        n_slices = sum(len(v) for v in groups.values())

        def task(progress_callback, cancel_event):
            a = argparse.Namespace(
                nexp=params['n_exp'],
                tau_min=params['tau_min'],
                tau_max=params['tau_max'],
                estimate_irf=params['estimate_irf'],
                machine_irf=params['machine_irf'],
                irf_fwhm=params['irf_fwhm'],
                irf_bins=params['irf_bins'],
                irf_fit_width=params['irf_fit_width'],
                optimizer=expert.get('optimizer', 'de'),
                restarts=expert.get('lm_restarts', cfg['lm_restarts']),
                de_population=expert.get('de_population', cfg['de_population']),
                de_maxiter=expert.get('de_maxiter', cfg['de_maxiter']),
                workers=expert.get('n_workers', cfg['n_workers']),
                no_polish=False,
                channel=params['channel'],
                min_photons=expert.get('min_photons', cfg['MIN_PHOTONS_PERPIX']),
                correct_pileup=correct_pileup,
                cost_function=params['cost_function'],
                save_stack=True,
                no_plots=False,
            )
            return fit_zstack(
                ptu_dir=ptu_dir,
                output_dir=out_dir,
                args=a,
                channel=params['channel'],
                compute_bound_fraction=False,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )

        def on_done(result):
            self._set_buttons('normal')
            self._res.set_status(
                f'✓  Z-stack complete - {n_stacks} stack(s), {n_slices} slices.')
            first = next(iter(result.values())) if isinstance(result, dict) and result else None
            if first is not None:
                from flimkit.UI.flim_display import load_zstack_display_slices
                try:
                    slices = load_zstack_display_slices(first['group_dir'], ptu_dir=ptu_dir)
                    if slices:
                        self._fov_preview.display_zstack(slices)
                except Exception as exc:
                    print(f'[Z-stack] Could not load fitted stack into preview: {exc}')
                self._populate_zstack_summary_from_dir(first['group_dir'])
                self._res.load_images(first.get('preview_dir') or out_dir)
            else:
                self._res.load_images(out_dir)
            if hasattr(self, '_proj_browser') and self._proj_browser and self._proj_browser._project:
                first_key = next(iter(result.keys())) if isinstance(result, dict) and result else None
                if first_key is not None:
                    self._proj_browser.on_fit_done(
                        first_key, out_st=out_dir, ptu_dir=ptu_dir)
        self._set_buttons('disabled')
        self.run_with_progress(
            task,
            task_name=f'Z-stack Fit ({n_stacks} stack(s), {n_slices} slices)',
            on_done=on_done,
            output_dir=out_dir,
        )

    def _run_stitch(self):
        xlif = self.sv_xlif.get().strip()
        ptu_dir = self.sv_ptu_dir.get().strip()
        out_base = self.sv_out_st.get().strip()
        pipeline = self.sv_pipeline.get()
        required = [(ptu_dir, 'PTU directory'), (out_base, 'Output directory')]
        if pipeline != 'series_fit':
            required.insert(0, (xlif, 'XLIF file'))
        for val, name in required:
            if not val:
                messagebox.showerror('Missing input', f'Please specify the {name}.')
                return
        pool_stride = 10
        if pipeline == 'series_fit':
            try:
                pool_stride = max(1, int(self.sv_pool_stride.get() or 10))
            except ValueError:
                messagebox.showerror('Invalid input',
                                     'Pool decay every N timepoints must be a whole number.')
                return
        from flimkit.formats.PTU.stitch import stitch_flim_tiles
        a = self._controller.stitch_args()

        def on_done(result):
            self._set_buttons('normal')
            self._res.set_status('✓  Complete.')
            self._res._nb.select(0)
            captured = ''.join(self._buf)
            rows = _parse_summary(captured)
            if pipeline == 'tile_fit' and isinstance(result, dict):
                fit_result = result
                global_summary = fit_result.get('global_summary', {})
                global_popt = fit_result.get('global_popt')
                if global_summary:
                    extracted_rows = self._extract_summary_rows(global_summary, global_popt)
                    if extracted_rows:
                        rows = extracted_rows
                try:
                    self._fov_preview.load_stitched_roi(a.output_dir)
                except Exception as e:
                    print(f'[Warning] Could not load stitched image: {e}')
                try:
                    self._fov_preview.display_fit_results(None, fit_result)
                except Exception as e:
                    import traceback
                    print(f'[Warning] Could not display fit results: {e}')
                    traceback.print_exc()
                try:
                    self._save_roi_progress(str(a.output_dir), fit_result, rows or [])
                    npz_path = Path(a.output_dir) / 'roi_session.npz'
                    self._res.set_fit_result(
                        fit_result, str(a.output_dir),
                        npz_path=str(npz_path) if npz_path.exists() else None,
                        scan_name=self._current_scan_stem(),
                    )
                    if hasattr(self, '_proj_browser'):
                        xlif_stem = Path(a.xlif).stem if hasattr(a, 'xlif') else None
                        if xlif_stem:
                            self._proj_browser.on_fit_done(
                                xlif_stem,
                                out_st = str(Path(a.output_dir).parent),
                                ptu_dir = getattr(a, 'ptu_dir', None),
                            )
                except Exception as e:
                    import traceback
                    print(f'[Warning] Could not save NPZ: {e}')
                    traceback.print_exc()
                try:
                    if hasattr(self, '_stitch_roi_panel'):
                        self._stitch_roi_panel._refresh_region_list()
                except Exception:
                    pass
            elif pipeline == 'series_fit' and isinstance(result, dict):
                planes = result.get('planes', [])
                taus = result.get('consensus_taus_ns', [])
                print(f"\n  {len(planes)} plane(s) written under {a.output_dir}")
                print(f"  Manifest: {result.get('base', '')}_series_index.json")
                if taus:
                    print(f"  Consensus τ = {[f'{t:.3f}' for t in taus]} ns")
                if planes:
                    self._res.set_status(
                        f'✓  {len(planes)} planes fitted - drag the slider to browse them')
                    try:
                        self._fov_preview.display_series(planes, a.output_dir)
                    except Exception as e:
                        print(f'Warning: Could not load series planes: {e}')
            else:
                try:
                    self._fov_preview.load_stitched_roi(a.output_dir)
                except Exception as e:
                    print(f'Warning: Could not load stitched image: {e}')
            if rows:
                self._res.populate_summary(rows)

        def task(progress_callback, cancel_event):
            if pipeline == 'stitch_only':
                return stitch_flim_tiles(
                    xlif_path=a.xlif,
                    ptu_dir=a.ptu_dir,
                    output_dir=a.output_dir,
                    ptu_basename=a.ptu_basename,
                    rotate_tiles=a.rotate_tiles,
                    register_tiles=a.register_tiles,
                    reg_max_shift_px=a.reg_max_shift_px,
                    verbose=True,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            elif pipeline == 'stitch_fit':
                from flimkit.interactive import _run_stitch_and_fit
                return _run_stitch_and_fit(a, progress_callback=progress_callback,
                                           cancel_event=cancel_event)
            elif pipeline == 'series_fit':
                from flimkit.formats.PTU.stitch import fit_flim_series
                return fit_flim_series(
                    ptu_dir=a.ptu_dir,
                    output_dir=a.output_dir,
                    args=a,
                    rotate_tiles=a.rotate_tiles,
                    xlif_path=a.xlif or None,
                    pool_stride=pool_stride,
                    verbose=True,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
            else:
                from flimkit.interactive import _run_tile_fit
                return _run_tile_fit(a, progress_callback=progress_callback,
                                     cancel_event=cancel_event)
        self._set_buttons('disabled')
        task_name = {'stitch_only': 'Stitching', 'stitch_fit': 'Stitch + Fit',
                     'tile_fit': 'Per-Tile Fit', 'series_fit': 'Series Fit'}[pipeline]
        self.run_with_progress(task, task_name=task_name, on_done=on_done, output_dir=a.output_dir)

    def _run_phasor(self):
        try:
            min_ph = float(self.sv_ph_minph.get() or 0.01)
            max_cur = int(self.sv_ph_maxc.get() or 6)
        except ValueError:
            messagebox.showerror('Invalid input',
                                 'Min photons and max cursors must be numeric.')
            return
        self._phasor_panel.max_cursors = max_cur
        if self.sv_ph_mode.get() == 'session':
            sess = self.sv_ph_session.get().strip()
            if not sess or not Path(sess).exists():
                messagebox.showerror('Missing input',
                                     'Please select a valid .npz session file.')
                return

            def _worker():
                from flimkit.phasor_launcher import load_session
                return load_session(sess)

            def _done(result):
                if isinstance(result, Exception):
                    messagebox.showerror('Session load error', str(result))
                    return
                self._phasor_panel.load_session(result, min_photons=min_ph)
                self._res.set_status('✓  Phasor session loaded.')
            self._phasor_thread(_worker, _done, status='  Loading session...')
        else:
            ptu = self.sv_ph_ptu.get().strip()
            if not ptu or not Path(ptu).exists():
                messagebox.showerror('Missing input',
                                     'Please select a valid PTU file.')
                return
            try:
                channel = self._resolve_phasor_channel(ptu)
            except Exception as exc:
                messagebox.showerror('Phasor error', str(exc), parent=self.root)
                return
            if channel is None:
                self._res.set_status('  Phasor load cancelled.')
                return
            xlsx_irf = self.sv_ph_irf.get().strip() or None
            mach_irf = self.sv_ph_mirf.get().strip() or None
            irf_path = xlsx_irf or mach_irf

            from flimkit.formats import file_modality
            modality = file_modality(ptu)

            def _worker():
                if modality == 'frequency':
                    from flimkit.phasor.signal import process_ifli
                    return process_ifli(ptu, channel=channel)
                from flimkit.phasor_launcher import _process_ptu
                return _process_ptu(ptu, irf_path=irf_path, channel=channel)

            def _done(result):
                if isinstance(result, Exception):
                    messagebox.showerror('Phasor error', str(result))
                    return
                frequency = result.get('frequency') or 0.0
                if not frequency:
                    frequency = self._prompt_frequency(ptu) or 0.0
                    result['frequency'] = frequency
                self._phasor_panel.set_data(
                    result['real_cal'],
                    result['imag_cal'],
                    result['mean'],
                    frequency,
                    display_image=result.get('display_image'),
                    min_photons=min_ph,
                )
                self._phasor_panel._ptu_path = ptu
                self._phasor_panel._channel = channel
                self._auto_save_phasor(ptu)
                self._res.set_status(
                    f'✓  Phasor data loaded from channel {channel} - click the phasor to place cursors.')
            self._phasor_thread(_worker, _done,
                                status='  Loading PTU and computing phasors...')

    def _resolve_phasor_channel(self, ptu_path: str) -> Optional[int]:
        from flimkit.phasor_launcher import get_ptu_active_channels
        active_channels = get_ptu_active_channels(ptu_path)
        if not active_channels:
            raise ValueError('No photon channels found in PTU file')
        if len(active_channels) == 1:
            return active_channels[0]
        return self._prompt_phasor_channel(ptu_path, active_channels)

    def _prompt_phasor_channel(
        self,
        ptu_path: str,
        active_channels: list[int],
    ) -> Optional[int]:
        available = ', '.join(str(channel) for channel in active_channels)
        prompt = (
            f'Multiple photon channels were detected in:\n{Path(ptu_path).name}\n\n'
            f'Available channels: {available}\n\n'
            'Enter the channel to use for phasor analysis:'
        )
        while True:
            selected = simpledialog.askinteger(
                'Select PTU Channel',
                prompt,
                parent=self.root,
                minvalue=min(active_channels),
                maxvalue=max(active_channels),
            )
            if selected is None:
                return None
            if selected in active_channels:
                return selected
            messagebox.showerror(
                'Invalid channel',
                f'Channel {selected} is not present in this PTU file.\n\n'
                f'Available channels: {available}',
                parent=self.root,
            )

    def _prompt_frequency(self, path: str) -> Optional[float]:
        prompt = (
            f'No laser repetition frequency is stored in:\n{Path(path).name}\n\n'
            'This format carries no frequency metadata, so the universal circle\n'
            'and lifetime overlays cannot be drawn without it.\n\n'
            'Enter the modulation frequency in MHz (leave blank to skip):'
        )
        return simpledialog.askfloat(
            'Enter Frequency (MHz)',
            prompt,
            parent=self.root,
            minvalue=0.0,
        )

    def _phasor_thread(self, worker_fn, done_cb, *, status='  Working...'):
        self._btn_ph.configure(state='disabled')
        self._res.set_status(status)

        def _run():
            try:
                result = worker_fn()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result = exc
            self.root.after(0, lambda: _finish(result))

        def _finish(result):
            self._btn_ph.configure(state='normal')
            done_cb(result)
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _auto_save_phasor(self, ptu_path: str):
        try:
            sd = self._phasor_panel.get_session_dict()
            if sd.get('real_cal') is None:
                return
            p = Path(ptu_path)
            save_path = str(p.parent / f'{p.stem}_phasor.npz')
            from flimkit.phasor_launcher import save_session
            save_session(
                save_path,
                real_cal=sd['real_cal'],
                imag_cal=sd['imag_cal'],
                mean=sd['mean'],
                frequency=sd['frequency'],
                cursors=sd['cursors'],
                params=sd['params'],
                ptu_file=ptu_path,
                display_image=sd.get('display_image'),
            )
            print(f'[Phasor] Auto-saved session → {save_path}')
            if hasattr(self, '_proj_browser'):
                self._proj_browser.on_phasor_done(p.stem)
        except Exception as e:
            print(f'[Phasor] Auto-save failed: {e}')

    def _on_phasor_change(self, panel):
        ptu = self.sv_ph_ptu.get().strip() if hasattr(self, 'sv_ph_ptu') else ''
        if ptu and Path(ptu).exists():
            self._auto_save_phasor(ptu)

    def _restore_phasor_session(self, npz_path: str):
        try:
            min_ph = float(self.sv_ph_minph.get() or 0.01)
        except ValueError:
            min_ph = 0.01
        def _worker():
            from flimkit.phasor_launcher import load_session
            return load_session(npz_path)
        def _done(result):
            if isinstance(result, Exception):
                print(f'[Phasor] Could not restore session: {result}')
                return
            self._phasor_panel.load_session(result, min_photons=min_ph)
            self._res.set_status('✓  Phasor session restored.')
        self._phasor_thread(_worker, _done, status='  Restoring phasor session...')

    def _run_build_machine_irf(self):
        src_dir = self.sv_mirf_src.get().strip()
        out_dir = self.sv_mirf_out_dir.get().strip()
        out_name = self.sv_mirf_name.get().strip()
        anchor = self.sv_mirf_anchor.get().strip()
        reducer = self.sv_mirf_reducer.get().strip()
        if not src_dir or not Path(src_dir).exists():
            messagebox.showerror('Missing input', 'Please select a valid PTU/XLSX source folder.')
            return
        if not out_dir:
            messagebox.showerror('Missing input', 'Please select an output directory.')
            return
        if not out_name:
            messagebox.showerror('Missing input', 'Please enter an output base filename.')
            return
        from flimkit.FLIM.irf_tools import build_machine_irf_from_folder

        def task_fn():
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return build_machine_irf_from_folder(
                folder=src_dir,
                align_anchor=anchor,
                reducer=reducer,
                save=True,
                confirm_save=True,
                output_name=out_name,
                output_dir=out_dir,
                verbose=True,
            )

        def on_done_irf(result):
            self._set_buttons('normal')
            self._res.set_status('✓  Machine IRF built.')
            if result is None or not isinstance(result, dict):
                return
            irf = result.get('irf')
            meta = result.get('metadata', {})
            if irf is None:
                return
            tcspc_ns = float(meta.get('tcspc_res_ns_mean', 0.05))
            import numpy as np
            time_ns = np.arange(len(irf)) * tcspc_ns
            preview_parent = self._preview_frame_label
            if not hasattr(self, '_irf_plot_frame'):
                import tkinter as _tk
                from tkinter import ttk as _ttk
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                self._irf_plot_frame = _ttk.Frame(preview_parent)
                self._irf_plot_frame.grid(row=0, column=0, sticky='nsew')
                self._irf_fig = Figure(figsize=(6, 4), dpi=100, facecolor='#1e1e1e')
                self._irf_ax = self._irf_fig.add_subplot(111)
                self._irf_canvas_mpl = FigureCanvasTkAgg(self._irf_fig, master=self._irf_plot_frame)
                self._irf_canvas_mpl.get_tk_widget().pack(fill='both', expand=True)
            ax = self._irf_ax
            ax.clear()
            ax.set_facecolor('#1e1e1e')
            self._irf_fig.patch.set_facecolor('#1e1e1e')
            n_pairs = meta.get('n_pairs', '?')
            anchor = meta.get('align_anchor', '')
            reducer = meta.get('reducer', '')
            ax.plot(time_ns, irf, color='#00d4ff', linewidth=2, label=f'Machine IRF ({reducer})')
            ax.set_xlabel('Time (ns)', color='white')
            ax.set_ylabel('Amplitude (normalised)', color='white')
            ax.set_title(f'Machine IRF  |  {n_pairs} pairs  |  anchor={anchor}',
                         color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.spines[:].set_color('#555')
            import numpy as _np
            half_max = irf.max() / 2
            above = _np.where(irf >= half_max)[0]
            if len(above) > 1:
                fwhm_ns = (above[-1] - above[0]) * tcspc_ns
                ax.axhline(half_max, color='#ff9900', linewidth=1,
                           linestyle='', alpha=0.7, label=f'FWHM={fwhm_ns*1000:.0f} ps')
                ax.axvspan(above[0]*tcspc_ns, above[-1]*tcspc_ns,
                           alpha=0.12, color='#ff9900')
            ax.legend(fontsize=8, facecolor='#2a2a2a', edgecolor='#555', labelcolor='white')
            self._irf_canvas_mpl.draw_idle()
            self._irf_plot_frame.grid()
            self._fov_preview.frame.grid_remove()
            self._preview_frame_label.configure(text='  Machine IRF Builder  ')
        self._launch(task_fn, output_dir=out_dir, task_name='Building Machine IRF',
                     _on_done_override=on_done_irf)

    def _launch(self, fn, output_dir=None, ptu_path=None, task_name='Working...',
                _on_done_override=None):
        self._buf.clear()
        self._set_buttons('disabled')
        self._res.set_status('  Running...')
        self._res._nb.select(0)
        win_manager = ProgressWindowManager(self.root)
        win = ProgressWindow(self.root, task_name=task_name)
        cancel_event = win.cancelled

        def progress_callback(i, total):
            win.set_progress(i, maximum=total)
            if cancel_event.is_set():
                win.set_status('Cancelling...')

        def _worker():
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            redir = _Redirect(self._res.log, self._buf, root=self.root)
            redir_err = _Redirect(self._res.log, self._buf, root=self.root, is_stderr=True)
            sys.stdout = redir
            sys.stderr = redir_err
            try:
                sig = inspect.signature(fn)
                if 'progress_callback' in sig.parameters or 'cancel_event' in sig.parameters:
                    if 'progress_window_manager' in sig.parameters:
                        result = fn(progress_callback=progress_callback, cancel_event=cancel_event,
                                   progress_window_manager=win_manager)
                    else:
                        result = fn(progress_callback=progress_callback, cancel_event=cancel_event)
                else:
                    result = fn()
                captured = ''.join(self._buf)
                rows = _parse_summary(captured)
                if _on_done_override is not None:
                    self.root.after(0, lambda r=result: _on_done_override(r))
                else:
                    self.root.after(0, lambda: self._on_done(rows, output_dir, result, ptu_path))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                from flimkit.utils.crash_handler import log_exception
                log_exception(f'_launch: {task_name}')
                self.root.after(0, lambda e=exc: self._res.set_status(f'✗  Error: {e}'))
            finally:
                self.root.after(0, lambda: win.close())
                self.root.after(0, lambda: win_manager.close_all())
                if hasattr(redir, 'close'):
                    redir.close()
                else:
                    redir.flush()
                if hasattr(redir_err, 'close'):
                    redir_err.close()
                else:
                    redir_err.flush()
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                self.root.after(0, lambda: self._set_buttons('normal'))
        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, rows, output_dir, fit_result=None, ptu_path=None):
        self._res.set_status('✓  Finished.')
        if fit_result is not None:
            global_summary = fit_result.get('global_summary')
            global_popt = fit_result.get('global_popt')
            if global_summary is not None:
                extracted_rows = self._extract_summary_rows(global_summary, global_popt)
                if extracted_rows:
                    rows = extracted_rows
        if rows:
            self._res.populate_summary(rows)
        if fit_result is not None:
            try:
                self._fov_preview.display_fit_results(ptu_path, fit_result)
                if hasattr(self, '_roi_analysis_panel'):
                    self._roi_analysis_panel._refresh_region_list()
            except Exception as e:
                import traceback
                traceback.print_exc()
        sug = fit_result.get('suggested_binning') if fit_result else None
        if sug:
            messagebox.showwarning(
                'Photon-starved image',
                'Most pixels have too few photons for a reliable per-pixel fit, so '
                'the FLIM image is mostly empty.\n\n'
                f'Try binning {sug}×{sug} (Expert settings → binning), or use '
                'the summed-fit result / phasor analysis for this image.')
        npz_file_path = None
        if ptu_path or output_dir:
            try:
                self._save_roi_progress(ptu_path or output_dir, fit_result, rows or [])
                from pathlib import Path
                base_path = Path(ptu_path) if ptu_path else Path(output_dir)
                if base_path.is_file():
                    npz_file_path = str(base_path.parent / f'{base_path.stem}.roi_session.npz')
                else:
                    npz_file_path = str(base_path / 'roi_session.npz')
            except Exception as e:
                print(f'[Info] Could not save ROI progress: {e}')
        self._res.set_fit_result(fit_result or {}, output_dir, npz_path=npz_file_path,
                                 scan_name=Path(ptu_path).stem if ptu_path else self._current_scan_stem())
        if hasattr(self, '_proj_browser'):
            stem = Path(ptu_path).stem if ptu_path else None
            prefix = self.sv_out_fov.get().strip() if hasattr(self, 'sv_out_fov') else None
            if stem:
                self._proj_browser.on_fit_done(stem, output_prefix=prefix)

    def _extract_summary_rows(self, global_summary: dict, global_popt=None) -> list:
        if not global_summary:
            return []
        rows = []
        tau_centers = global_summary.get('tau_centers_ns')
        if tau_centers is not None:
            import numpy as _np
            tau_centers = list(_np.atleast_1d(tau_centers))
            widths_ns = list(_np.atleast_1d(global_summary.get('widths_ns', [])))
            fwhms_ns = list(_np.atleast_1d(global_summary.get('fwhms_ns', [])))
            amps_d = list(_np.atleast_1d(global_summary.get('amps', [])))
            fracs_d = list(_np.atleast_1d(global_summary.get('fractions', [])))
            dist_type = global_summary.get('dist_type', 'gaussian')
            dist_label = dist_type.capitalize()
            width_label = 'σ' if dist_type == 'gaussian' else 'Γ (FWHM)'
            for i in range(len(tau_centers)):
                rows.append((f'τ̄{i+1} ({dist_label} center)', f'{tau_centers[i]:.4f}', 'ns'))
                if i < len(widths_ns):
                    rows.append((f'{width_label}{i+1}', f'{widths_ns[i]:.4f}', 'ns'))
                if i < len(fwhms_ns):
                    rows.append((f'FWHM{i+1}', f'{fwhms_ns[i]:.4f}', 'ns'))
                if i < len(amps_d):
                    rows.append((f'g{i+1} (amplitude)', f'{amps_d[i]:.3e}', ''))
                if i < len(fracs_d):
                    rows.append((f'f{i+1} (int. frac.)', f'{fracs_d[i]:.4f}', ''))
        taus = global_summary.get('taus_ns', []) if tau_centers is None else None
        amps = global_summary.get('amps', []) if tau_centers is None else None
        fracs = global_summary.get('fractions', []) if tau_centers is None else None
        if taus is not None and len(taus) > 0:
            import numpy as np
            taus = list(np.atleast_1d(taus))
            amps = list(np.atleast_1d(amps)) if amps is not None else []
            fracs = list(np.atleast_1d(fracs)) if fracs is not None else []
            intens = list(np.atleast_1d(global_summary.get('intensities', [])))
            ifracs = list(np.atleast_1d(global_summary.get('intensity_fractions', [])))
            for i in range(len(taus)):
                rows.append((f'τ{i+1}', f'{taus[i]:.4f}', 'ns'))
                if i < len(amps):
                    rows.append((f'α{i+1}', f'{amps[i]:.3e}', ''))
                if i < len(fracs):
                    rows.append((f'f{i+1} (amp frac)', f'{fracs[i]:.4f}', ''))
                if i < len(intens):
                    rows.append((f'I{i+1} (intensity)', f'{intens[i]:.3e}', 'cts'))
                if i < len(ifracs):
                    rows.append((f'f{i+1} (int frac)', f'{ifracs[i]:.4f}', ''))
        for key, label in [
            ('tau_mean_amp_ns', 'τ_mean (amp-weighted)'),
            ('tau_mean_int_ns', 'τ_mean (int-weighted)'),
        ]:
            v = global_summary.get(key)
            if v is not None:
                rows.append((label, f'{v:.4f}', 'ns'))
        tau_global = global_summary.get('tau_mean_amp_global_ns')
        if tau_global is not None:
            rows.append(('τ_mean amp-wtd (global)', f'{tau_global:.4f}', 'ns'))
        tau_std = global_summary.get('tau_std_amp_global_ns')
        if tau_std is not None:
            rows.append(('τ σ (pixel distrib.)', f'{tau_std:.4f}', 'ns'))
        tau_med = global_summary.get('tau_median_amp_global_ns')
        if tau_med is not None:
            rows.append(('τ_median (amp-wtd)', f'{tau_med:.4f}', 'ns'))
        n_px = global_summary.get('n_pixels_fitted')
        if n_px is not None:
            rows.append(('Pixels fitted', f'{n_px:,}', ''))
        k = 1
        while True:
            tau_k = global_summary.get(f'tau{k}_mean_ns')
            if tau_k is None:
                break
            rows.append((f'τ{k} mean', f'{tau_k:.4f}', 'ns'))
            a_k = global_summary.get(f'a{k}_mean_frac')
            if a_k is not None:
                rows.append((f'f{k} mean (amp frac)', f'{a_k:.4f}', ''))
            k += 1
        i_sum = global_summary.get('i_sum')
        if i_sum is not None:
            rows.append(('I_sum', f'{i_sum:.3e}', 'cts'))
        a_sum = global_summary.get('a_sum')
        if a_sum is not None:
            rows.append(('A_sum', f'{a_sum:.3e}', ''))
        bg_fit = global_summary.get('bg_fit')
        if bg_fit is not None:
            rows.append(('Background (fitted)', f'{bg_fit:.2f}', 'cts/bin'))
        t0_ns = global_summary.get('t0_ns')
        if t0_ns is not None:
            rows.append(('t0 (lifetime offset)', f'{t0_ns:.4f}', 'ns'))
        irf_shift = global_summary.get('irf_shift_bins')
        if irf_shift is not None:
            rows.append(('IRF shift', f'{irf_shift:.3f}', 'bins'))
        irf_sigma = global_summary.get('irf_sigma_bins')
        if irf_sigma is not None:
            rows.append(('IRF σ (broadening)', f'{irf_sigma:.3f}', 'bins'))
        irf_fwhm = global_summary.get('irf_fwhm_eff_ns')
        if irf_fwhm is not None:
            rows.append(('IRF FWHM (eff.)', f'{irf_fwhm:.4f}', 'ns'))
        chi2_r_tail = global_summary.get('reduced_chi2_tail')
        if chi2_r_tail is not None:
            rows.append(('χ²_r(tail) Neyman', f'{chi2_r_tail:.4f}', ''))
        chi2_p_tail = global_summary.get('reduced_chi2_tail_pearson')
        if chi2_p_tail is not None:
            rows.append(('χ²_r(tail) Pearson', f'{chi2_p_tail:.4f}', ''))
        return rows

    def _set_buttons(self, state):
        for btn in (self._btn_fov, self._btn_st, self._btn_ph):
            btn.configure(state=state)

if HAS_TKMT:
    class FLIMKitGUIThemed(TKMT.ThemedTKinterFrame, _UIBuilder):
        def __init__(self, theme='sun-valley', mode='dark'):
            super().__init__('FLIMkit Analysis GUI', theme, mode,
                             usecommandlineargs=True, useconfigfile=True)
            self.root = self.master
            self.root.minsize(760, 700)
            self._init_ui()
            self.run(cleanresize=False)

class FLIMKitGUIFallback(_UIBuilder):
    def __init__(self, root):
        self.root = root
        self.root.title('FLIMkit Analysis GUI')
        self.root.minsize(760, 700)
        self._init_ui()
        self.root.mainloop()

def launch_gui():
    global GUI_MODE
    GUI_MODE = True
    from flimkit.utils.crash_handler import init_crash_handler
    init_crash_handler()
    if HAS_TKMT:
        app = FLIMKitGUIThemed(theme='sun-valley', mode='dark')
    else:
        if HAS_DND:
            from tkinterdnd2 import Tk
            root = Tk()
        else:
            root = tk.Tk()
        style = ttk.Style(root)
        for theme_name in ('clam', 'alt', 'default'):
            if theme_name in style.theme_names():
                style.theme_use(theme_name)
                break
        app = FLIMKitGUIFallback(root)

if __name__ == '__main__':
    launch_gui()
