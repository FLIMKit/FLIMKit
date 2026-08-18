import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

_ICON_FIT_ONLY = '●'
_ICON_PHASOR_ONLY = '◐'
_ICON_BOTH = '◉'
_ICON_NOSESSION = '○'
_ICON_FOV = 'F'
_ICON_XLIF = 'T'
_ICON_ZSTACK = 'Z'

class ProjectBrowserPanel:

    def __init__(self, parent, app, width=170):
        self._app = app
        self._width = width
        self._project = None
        self._stems = []
        self.frame = ttk.Frame(parent, width=width)
        self.frame.grid_propagate(False)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        hdr = ttk.Frame(self.frame)
        hdr.grid(row=0, column=0, sticky='ew', padx=4, pady=(6, 2))
        ttk.Label(hdr, text='Project',
                  font=('TkDefaultFont', 9, 'bold')).pack(side='left')
        lb_outer = ttk.Frame(self.frame, relief='sunken', borderwidth=1)
        lb_outer.grid(row=1, column=0, sticky='nsew', padx=4, pady=2)
        lb_outer.columnconfigure(0, weight=1)
        lb_outer.rowconfigure(0, weight=1)
        sb = ttk.Scrollbar(lb_outer, orient='vertical')
        sb.grid(row=0, column=1, sticky='ns')
        self._lb = tk.Listbox(
            lb_outer,
            yscrollcommand = sb.set,
            selectmode = 'single',
            activestyle = 'none',
            font = ('Courier', 9),
            relief = 'flat',
            borderwidth = 0,
            highlightthickness = 0,
            exportselection = False,
        )
        self._lb.grid(row=0, column=0, sticky='nsew')
        sb.config(command=self._lb.yview)
        self._lb.bind('<<ListboxSelect>>', self._on_select)
        self._sv_status = tk.StringVar(value='No project open')
        ttk.Label(
            self.frame,
            textvariable = self._sv_status,
            foreground = 'grey',
            font = ('TkDefaultFont', 7),
            wraplength = width - 12,
            anchor = 'w',
        ).grid(row=2, column=0, sticky='ew', padx=6, pady=(2, 6))
        self._setup_dnd()

    def grid(self, **kw):
        self.frame.grid(**kw)

    def on_fit_done(
        self,
        stem,
        out_st=None,
        output_prefix=None,
        ptu_dir=None,
    ):
        if self._project is None:
            return
        self._project.update_after_fit(
            stem,
            out_st = out_st,
            output_prefix = output_prefix,
            ptu_dir = ptu_dir,
        )
        self._project.save()
        self._refresh()

    def on_phasor_done(self, stem):
        if self._project is None:
            return
        self._project.update_after_phasor(stem)
        self._project.save()
        self._refresh()

    def _setup_dnd(self):
        try:
            from tkinterdnd2 import DND_FILES, DND_TEXT
        except ImportError:
            return
        try:
            self._lb.drop_target_register(DND_FILES, DND_TEXT)
            self._lb.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass

    def _on_drop(self, event):
        try:
            paths = self._lb.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        for p in paths:
            p = p.strip()
            if not p:
                continue
            path = Path(p)
            if path.is_dir():
                self.load_folder(str(path))
                if hasattr(self._app, '_add_to_recent'):
                    self._app._add_to_recent(str(path), 'project')
                return
            if path.is_file():
                if hasattr(self._app, '_switch_form'):
                    self._app._switch_form('fov')
                if hasattr(self._app, 'sv_ptu'):
                    self._app.sv_ptu.set(str(path))
                return

    def load_folder(self, folder):
        if not folder:
            return
        from flimkit.project import ProjectFile
        self._project = ProjectFile.load_or_create(Path(folder))
        self._project.save()
        if self._project.config:
            from flimkit.utils.config_manager import cfg
            cfg.load_project_overrides(self._project.config)
        self._refresh()

    def _open_project(self):
        folder = filedialog.askdirectory(title='Open project folder')
        if not folder:
            return
        self.load_folder(folder)
        if hasattr(self._app, '_add_to_recent'):
            self._app._add_to_recent(folder, 'project')

    def _refresh(self):
        if self._project is None:
            return
        self._lb.delete(0, tk.END)
        self._stems.clear()
        for stem, rec in self._project.sorted_scans():
            has_fit = rec.has_session
            has_ph = rec.has_phasor_session
            if has_fit and has_ph:
                session_dot = _ICON_BOTH
            elif has_fit:
                session_dot = _ICON_FIT_ONLY
            elif has_ph:
                session_dot = _ICON_PHASOR_ONLY
            else:
                session_dot = _ICON_NOSESSION
            if rec.scan_type == 'xlif':
                type_tag = _ICON_XLIF
            elif rec.scan_type == 'zstack':
                type_tag = _ICON_ZSTACK
            else:
                type_tag = _ICON_FOV
            label = f'{session_dot} {type_tag} {stem}'
            self._lb.insert(tk.END, label)
            self._stems.append(stem)
        n = len(self._stems)
        n_sess = sum(1 for r in self._project.scans.values()
                     if r.has_session or r.has_phasor_session)
        folder_name = self._project.project_dir.name
        self._sv_status.set(
            f"{folder_name}  |  {n} scan{'s' if n != 1 else ''}  |  {n_sess} saved"
        )

    def _on_select(self, _event=None):
        if getattr(self, '_selecting', False):
            return
        sel = self._lb.curselection()
        if not sel or self._project is None:
            return
        stem = self._stems[sel[0]]
        rec = self._project.scans.get(stem)
        if rec is None:
            return
        self._selecting = True
        try:
            self._load_scan(rec)
        finally:
            self._selecting = False

    def _load_scan(self, rec):
        app = self._app
        if hasattr(app, '_cancel_pending_scan_loads'):
            app._cancel_pending_scan_loads()
        if hasattr(app, '_fov_preview'):
            app._fov_preview._roi_manager.clear_all()
            app._fov_preview._roi_patches.clear()
            app._fov_preview._redraw_region_overlays()
        for panel in (getattr(app, '_roi_analysis_panel', None),
                      getattr(app, '_stitch_roi_panel', None)):
            if panel is not None:
                panel._refresh_region_list()
        if rec.scan_type == 'fov':
            current_form = getattr(app, '_current_form', 'fov')
            want_phasor = (current_form == 'phasor')
            if want_phasor:
                app._switch_form('phasor')
            else:
                app._switch_form('fov')
            if hasattr(app, '_last_loaded_ptu'):
                app._last_loaded_ptu = rec.source_path
            app.sv_ptu.set(rec.source_path)
            if hasattr(app, '_fov_preview'):
                app._fov_preview.load_fov(rec.source_path)
            if rec.xlsx_path and hasattr(app, 'sv_xlsx'):
                app.sv_xlsx.set(rec.xlsx_path)
            if hasattr(app, '_irf_fov'):
                if rec.xlsx_path:
                    app._irf_fov.sv_method.set('irf_xlsx')
                else:
                    app._irf_fov.sv_method.set('machine_irf')
            if hasattr(app, 'sv_out_fov'):
                app.sv_out_fov.set(rec.stem)
            if want_phasor:
                if hasattr(app, 'sv_ph_ptu'):
                    app.sv_ph_ptu.set(rec.source_path)
                if rec.phasor_session_path and hasattr(app, '_restore_phasor_session'):
                    app._restore_phasor_session(str(rec.phasor_session_path))
            else:
                if rec.session_path and hasattr(app, '_load_fitted_data_from_file'):
                    app._load_fitted_data_from_file(str(rec.session_path), suppress_popups=True)
        elif rec.scan_type == 'zstack':
            app._switch_form('fov')
            if hasattr(app, 'sv_fov_analysis'):
                app.sv_fov_analysis.set('zstack')
                if hasattr(app, '_on_fov_analysis_changed'):
                    app._on_fov_analysis_changed()
            if hasattr(app, 'sv_zstack_dir'):
                app.sv_zstack_dir.set(rec.ptu_dir or rec.source_path)
            if hasattr(app, 'sv_out_fov'):
                app.sv_out_fov.set(rec.stem)
            if rec.session_path and hasattr(app, '_fov_preview'):
                try:
                    from flimkit.utils.display import load_zstack_display_slices
                    group_dir = Path(rec.session_path).parent
                    slices = load_zstack_display_slices(
                        group_dir, ptu_dir=rec.ptu_dir, region=rec.stem)
                    if slices:
                        app._fov_preview.display_zstack(slices)
                    if hasattr(app, '_populate_zstack_summary_from_dir'):
                        app._populate_zstack_summary_from_dir(group_dir)
                except Exception as exc:
                    print(f'[Project] Could not reload z-stack {rec.stem}: {exc}')
        else:
            app._switch_form('stitch')
            if hasattr(app, '_last_loaded_xlif'):
                app._last_loaded_xlif = rec.source_path
            app.sv_xlif.set(rec.source_path)
            if rec.ptu_dir and hasattr(app, 'sv_ptu_dir'):
                app.sv_ptu_dir.set(rec.ptu_dir)
            out_st = self._project.default_out_st(rec.stem) if self._project else ''
            if hasattr(app, 'sv_out_st'):
                app.sv_out_st.set(out_st)
            if rec.session_path and hasattr(app, '_load_fitted_data_from_file'):
                app._load_fitted_data_from_file(str(rec.session_path), suppress_popups=True)
