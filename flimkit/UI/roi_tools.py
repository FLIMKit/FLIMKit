import json
from typing import List
import numpy as np
from flimkit.utils.roi import (
    RoiManager,
    _COLORS,
    get_rectangle_patch,
    get_ellipse_patch,
    get_polygon_patch,
)

def _show_roi_fit_result_standalone(result: dict):
    _show_fit_result_window(result)

def _show_fit_result_window(result: dict):
    import tkinter as tk
    from tkinter import ttk
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    summary    = result['summary']
    decay      = result['decay']
    time_ns    = result['time_ns']
    irf_prompt = result['irf_prompt']
    model      = summary.get('model')
    taus       = summary.get('taus_ns', [])
    amps       = summary.get('amps',    [])
    chi2       = summary.get('reduced_chi2_tail')
    win = tk.Toplevel()
    win.title(f"ROI Fit - {result['region_name']}")
    win.geometry('660x560')
    win.resizable(True, True)
    fig, (ax_d, ax_r) = plt.subplots(
        2, 1, figsize=(6.2, 4.2),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.patch.set_facecolor('#2b2b2b')
    for ax in (ax_d, ax_r):
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#555555')
    ax_d.semilogy(time_ns, decay, 'o-', color='steelblue',
                  linewidth=1.2, markersize=2, label='Decay', alpha=0.8)
    if irf_prompt is not None and irf_prompt.max() > 0:
        irf_sc = (irf_prompt / irf_prompt.max()) * decay.max() * 0.15
        ax_d.semilogy(time_ns[:len(irf_prompt)], np.maximum(irf_sc, 1e-2),
                      color='orange', linewidth=1.5,
                      label=f'IRF ({result['irf_source']})', alpha=0.7)
    if model is not None and len(model) == len(decay):
        ax_d.semilogy(time_ns, model, color='red', linewidth=2.0,
                      label='Fit', alpha=0.9)
    ax_d.legend(fontsize=7, loc='upper right', labelcolor='white',
                facecolor='#333333', edgecolor='#555555')
    title_bits = [result['region_name']]
    if len(taus) > 0:
        title_bits.append('  '.join(f'τ{i+1}={t:.3f}ns' for i, t in enumerate(taus)))
    if chi2 is not None:
        title_bits.append(f'χ²_r={chi2:.3f}')
    ax_d.set_title('  |  '.join(title_bits), fontsize=8, color='white')
    ax_d.set_ylabel('Photon Count', color='white', fontsize=8)
    if model is not None and len(model) == len(decay):
        with np.errstate(invalid='ignore', divide='ignore'):
            resid = np.where(model > 0, (decay - model) / np.sqrt(model), 0.0)
        ax_r.plot(time_ns, resid, color='steelblue', linewidth=0.9)
        ax_r.axhline(0, color='red', linewidth=1.0, linestyle='--', alpha=0.7)
        if chi2 is not None:
            ax_r.annotate(f'χ²_r = {chi2:.3f}',
                          xy=(0.98, 0.85), xycoords='axes fraction',
                          ha='right', va='top', fontsize=7, color='white',
                          bbox=dict(boxstyle='round,pad=0.2',
                                    fc='#333333', alpha=0.7))
    ax_r.set_ylabel('Resid. (σ)', color='white', fontsize=7)
    ax_r.set_xlabel('Time (ns)', color='white', fontsize=8)
    plt.tight_layout(pad=0.8)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
    tbl = ttk.Frame(win, padding=4)
    tbl.pack(fill='x', padx=6, pady=(0, 4))
    cols = ('Parameter', 'Value', 'Unit')
    n_rows = (len(taus) + len(amps)
              + (1 if chi2 is not None else 0)
              + (1 if len(taus) > 0 and len(amps) > 0 else 0))
    tv = ttk.Treeview(tbl, columns=cols, show='headings',
                      height=min(8, max(1, n_rows)))
    for col, w in zip(cols, (200, 110, 60)):
        tv.heading(col, text=col, anchor='w')
        tv.column(col, width=w, anchor='w')
    rows = []
    for i, tau in enumerate(taus):
        rows.append((f'τ{i+1}', f'{tau:.4f}', 'ns'))
    for i, amp in enumerate(amps):
        rows.append((f'A{i+1} (amplitude)', f'{amp:.4f}', ''))
    if len(taus) > 0 and len(amps) > 0 and np.sum(amps) > 0:
        tau_mean = float(np.dot(taus, amps) / np.sum(amps))
        rows.append(('τ_mean (amplitude-weighted)', f'{tau_mean:.4f}', 'ns'))
    if chi2 is not None:
        rows.append(('χ²_r (tail)', f'{chi2:.4f}', ''))
    for i, row in enumerate(rows):
        tv.insert('', 'end', values=row,
                  tags=('odd' if i % 2 else 'even',))
    tv.tag_configure('odd',  background='#f5f7fa', foreground='#1a1a1a')
    tv.tag_configure('even', background='#ffffff', foreground='#1a1a1a')
    tv.pack(fill='x')
    btn_row = ttk.Frame(win, padding=4)
    btn_row.pack(fill='x', padx=6, pady=(0, 6))
    ttk.Button(btn_row, text='Close',
               command=lambda: (plt.close(fig), win.destroy())).pack(side='right')
    win.protocol('WM_DELETE_WINDOW', lambda: (plt.close(fig), win.destroy()))

def _ask_roi_fit_options(params: dict):
    import tkinter as tk
    from tkinter import ttk
    result = {}
    dlg = tk.Toplevel()
    dlg.title('ROI Fit Options')
    dlg.resizable(False, False)
    dlg.grab_set()
    pad = dict(padx=8, pady=4)
    ttk.Label(dlg, text='Fit parameters for this ROI',
              font=('TkDefaultFont', 10, 'bold')).grid(
        row=0, column=0, columnspan=2, sticky='w', **pad)
    ttk.Label(dlg, text='Components (n_exp):').grid(row=1, column=0, sticky='w', **pad)
    sv_nexp = tk.IntVar(value=int(params.get('n_exp', 1)))
    nexp_frame = ttk.Frame(dlg)
    nexp_frame.grid(row=1, column=1, sticky='w', pady=4)
    for n, lbl in [(1, '1-exp'), (2, '2-exp'), (3, '3-exp')]:
        ttk.Radiobutton(nexp_frame, text=lbl, variable=sv_nexp, value=n).pack(
            side='left', padx=(0, 6))
    ttk.Label(dlg, text='τ_min (ns):').grid(row=2, column=0, sticky='w', **pad)
    sv_tau_min = tk.StringVar(value=str(params.get('tau_min', 0.1)))
    ttk.Entry(dlg, textvariable=sv_tau_min, width=10).grid(
        row=2, column=1, sticky='w', **pad)
    ttk.Label(dlg, text='τ_max (ns):').grid(row=3, column=0, sticky='w', **pad)
    sv_tau_max = tk.StringVar(value=str(params.get('tau_max', 25.0)))
    ttk.Entry(dlg, textvariable=sv_tau_max, width=10).grid(
        row=3, column=1, sticky='w', **pad)
    ttk.Label(dlg, text='Cost function:').grid(row=4, column=0, sticky='w', **pad)
    sv_cost = tk.StringVar(value=params.get('cost_function', 'poisson'))
    cost_frame = ttk.Frame(dlg)
    cost_frame.grid(row=4, column=1, sticky='w', pady=4)
    ttk.Radiobutton(cost_frame, text='Poisson deviance',
                    variable=sv_cost, value='poisson').pack(side='left', padx=(0, 6))
    ttk.Radiobutton(cost_frame, text='Pearson χ²',
                    variable=sv_cost, value='chi2').pack(side='left')
    ttk.Separator(dlg, orient='horizontal').grid(
        row=5, column=0, columnspan=2, sticky='ew', padx=8, pady=6)
    btn_frame = ttk.Frame(dlg)
    btn_frame.grid(row=6, column=0, columnspan=2, sticky='e', padx=8, pady=(0, 8))

    def _ok():
        try:
            tau_min = float(sv_tau_min.get())
            tau_max = float(sv_tau_max.get())
        except ValueError:
            tk.messagebox.showerror('Invalid Input',
                                    'τ_min and τ_max must be numbers.', parent=dlg)
            return
        if tau_min <= 0 or tau_max <= tau_min:
            tk.messagebox.showerror('Invalid Input',
                                    'Need 0 < τ_min < τ_max.', parent=dlg)
            return
        result['ok'] = True
        result['n_exp']          = sv_nexp.get()
        result['tau_min']        = tau_min
        result['tau_max']        = tau_max
        result['cost_function']  = sv_cost.get()
        dlg.destroy()

    def _cancel():
        dlg.destroy()
    ttk.Button(btn_frame, text='Cancel', command=_cancel).pack(side='left', padx=(0, 4))
    ttk.Button(btn_frame, text='Run Fit', command=_ok, style='Accent.TButton').pack(side='left')
    dlg.update_idletasks()
    w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
    sw = dlg.winfo_screenwidth()
    sh = dlg.winfo_screenheight()
    dlg.geometry(f'{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}')
    dlg.wait_window()
    if not result.get('ok'):
        return None
    merged = dict(params)
    merged['n_exp']         = result['n_exp']
    merged['tau_min']       = result['tau_min']
    merged['tau_max']       = result['tau_max']
    merged['cost_function'] = result['cost_function']
    return merged

class RoiAnalysisPanel:

    def __init__(self, parent, fov_preview=None):
        import tkinter as tk
        from tkinter import ttk
        self.frame = ttk.Frame(parent, padding=4)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)
        self.fov_preview = fov_preview
        self.app = None
        self._current_mode = tk.StringVar(value='select')
        self._region_counter = 0
        self._last_fit_results: dict = {}
        toolbar = ttk.LabelFrame(self.frame, text='Drawing Mode', padding=4)
        toolbar.grid(row=0, column=0, sticky='ew', pady=(0, 4))
        for i in range(3):
            toolbar.columnconfigure(i, weight=1)
        self._btn_select = ttk.Button(toolbar, text='◯ Select', width=12,
                                      command=lambda: self._set_mode('select'))
        self._btn_select.grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        self._btn_rect = ttk.Button(toolbar, text='▭ Rectangle', width=12,
                                    command=lambda: self._set_mode('rect'))
        self._btn_rect.grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        self._btn_ellipse = ttk.Button(toolbar, text='○ Ellipse', width=12,
                                       command=lambda: self._set_mode('ellipse'))
        self._btn_ellipse.grid(row=0, column=2, sticky='ew', padx=2, pady=2)
        self._btn_polygon = ttk.Button(toolbar, text='◇ Polygon', width=12,
                                       command=lambda: self._set_mode('polygon'))
        self._btn_polygon.grid(row=1, column=0, sticky='ew', padx=2, pady=2)
        self._btn_freehand = ttk.Button(toolbar, text='✏ Freehand', width=12,
                                        command=lambda: self._set_mode('freehand'))
        self._btn_freehand.grid(row=1, column=1, sticky='ew', padx=2, pady=2)
        ttk.Button(toolbar, text='Clear All', width=12,
                   command=self._clear_all_regions).grid(row=1, column=2, sticky='ew', padx=2, pady=2)
        list_frame = ttk.LabelFrame(self.frame, text='Regions', padding=4)
        list_frame.grid(row=1, column=0, sticky='nsew', pady=(0, 4))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        cols = ('Name', 'Type', 'τ_mean (ns)', 'τ_med (ns)', 'τ_sd (ns)', 'Photons', 'σ_photons')
        self._tree = ttk.Treeview(list_frame, columns=cols, height=6, show='tree headings')
        self._tree.grid(row=0, column=0, sticky='nsew')
        self._tree.column('#0', width=0, stretch=False)
        self._tree.column('Name', anchor='w', width=100)
        self._tree.column('Type', anchor='center', width=55)
        self._tree.column('τ_mean (ns)', anchor='center', width=72)
        self._tree.column('τ_med (ns)', anchor='center', width=72)
        self._tree.column('τ_sd (ns)', anchor='center', width=70)
        self._tree.column('Photons', anchor='center', width=65)
        self._tree.column('σ_photons', anchor='center', width=72)
        self._tree.heading('#0', text='', anchor='w')
        self._tree.heading('Name', text='Name', anchor='w')
        self._tree.heading('Type', text='Type', anchor='center')
        self._tree.heading('τ_mean (ns)', text='τ_mean (ns)', anchor='center')
        self._tree.heading('τ_med (ns)', text='τ_med (ns)', anchor='center')
        self._tree.heading('τ_sd (ns)', text='τ_sd (ns)', anchor='center')
        self._tree.heading('Photons', text='Photons', anchor='center')
        self._tree.heading('σ_photons', text='σ_photons', anchor='center')
        self._tree.bind('<Double-1>', self._on_region_double_click)
        self._tree.bind('<Delete>', self._on_delete_key)
        self._tree.bind('<<TreeviewSelect>>', self._on_region_selection_change)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self._tree.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self._tree.configure(yscroll=scrollbar.set)
        actions_frame = ttk.Frame(self.frame)
        actions_frame.grid(row=2, column=0, sticky='ew', pady=4)
        for i in range(3):
            actions_frame.columnconfigure(i, weight=1)
        ttk.Button(actions_frame, text='Delete Selected', width=16,
                   command=self._delete_selected_region).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(actions_frame, text='Rename...', width=16,
                   command=self._rename_selected_region).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        ttk.Button(actions_frame, text='Import from GeoJSON', width=18,
                   command=self._import_rois_geojson).grid(row=0, column=2, sticky='ew', padx=2, pady=2)
        ttk.Button(actions_frame, text='Export as CSV', width=16,
                   command=self._export_all_rois_csv).grid(row=1, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(actions_frame, text='Export as GeoJSON', width=18,
                   command=self._export_selected_region).grid(row=1, column=1, sticky='ew', padx=2, pady=2)
        ttk.Button(actions_frame, text='Export All as GeoJSON', width=20,
                   command=self._export_all_rois_geojson).grid(row=1, column=2, sticky='ew', padx=2, pady=2)
        ttk.Button(actions_frame, text='⚗ Fit ROI Decay', width=16,
                   command=self._fit_roi_decay).grid(row=2, column=0, columnspan=2,
                   sticky='ew', padx=2, pady=(6, 2))
        ttk.Button(actions_frame, text='View Fit', width=12,
                   command=self._view_last_fit_result).grid(row=2, column=2,
                   sticky='ew', padx=2, pady=(6, 2))
        self._add_plugin_buttons(actions_frame, start_row=3)
        self._status = tk.StringVar(value='Ready - Select drawing mode or click regions to add')
        ttk.Label(self.frame, textvariable=self._status, foreground='grey',
                  font=('Courier', 8)).grid(row=3, column=0, sticky='w', padx=2, pady=2)

    def _add_plugin_buttons(self, parent, start_row):
        from tkinter import ttk
        try:
            from flimkit import plugins
            buttons = plugins.panel_buttons('roi')
        except Exception:
            return
        for index, spec in enumerate(buttons):
            row = start_row + index // 3
            column = index % 3
            ttk.Button(parent, text=spec.label, width=18,
                       command=lambda spec=spec: self._run_plugin_button(spec)).grid(
                           row=row, column=column, sticky='ew', padx=2, pady=2)

    def _run_plugin_button(self, spec):
        from tkinter import messagebox
        if self.app is None:
            messagebox.showerror(
                spec.label,
                f'{spec.label} is not connected to the FLIMKit window yet.')
            return
        try:
            spec.callback(self.app)
        except Exception as exc:
            messagebox.showerror(
                spec.label,
                f'{spec.label} ({spec.source}) raised {type(exc).__name__}: {exc}')

    def _set_mode(self, mode: str):
        self._current_mode.set(mode)
        if self.fov_preview:
            self.fov_preview._drawing_mode.set(mode)
        self._status.set(f'Mode: {mode.upper()} - Draw on FLIM image')
        print(f'[ROI] Drawing mode: {mode}')

    def _clear_all_regions(self):
        if self.fov_preview:
            self.fov_preview._roi_manager.clear_all()
            self.fov_preview._redraw_region_overlays()
            self.fov_preview._save_regions_update()
        self._refresh_region_list()
        self._status.set('All regions cleared')

    def _on_region_double_click(self, event):
        selected = self._tree.selection()
        if selected:
            self._rename_selected_region()

    def _on_delete_key(self, event):
        self._delete_selected_region()

    def _on_region_selection_change(self, event):
        if getattr(self, '_refreshing', False):
            return
        selected = self._tree.selection()
        if selected:
            item = selected[0]
            region_id = int(item)
            if self.fov_preview:
                self.fov_preview._roi_manager.select_region(region_id)
                self.fov_preview._redraw_region_overlays()
        else:
            if self.fov_preview:
                self.fov_preview._roi_manager.select_region(None)
                self.fov_preview._redraw_region_overlays()

    def _delete_selected_region(self):
        selected = self._tree.selection()
        if not selected:
            return
        item = selected[0]
        region_id = int(item)
        if self.fov_preview:
            self.fov_preview._roi_manager.remove_region(region_id)
            self.fov_preview._redraw_region_overlays()
            self.fov_preview._save_regions_update()
        self._refresh_region_list()
        self._status.set(f'Deleted region {region_id}')

    def _rename_selected_region(self):
        import tkinter as tk
        from tkinter import simpledialog
        selected = self._tree.selection()
        if not selected:
            return
        item = selected[0]
        region_id = int(item)
        old_name = self._tree.item(item, 'values')[0]
        new_name = simpledialog.askstring('Rename Region',
                                         f'Enter new name for region:',
                                         initialvalue=old_name)
        if new_name:
            if self.fov_preview:
                self.fov_preview._roi_manager.update_region(region_id, name=new_name)
                self.fov_preview._save_regions_update()
            self._refresh_region_list()
            self._status.set(f"Renamed to '{new_name}'")

    def _get_fov_stem(self) -> str:
        if self.fov_preview and hasattr(self.fov_preview, '_ptu_path'):
            p = self.fov_preview._ptu_path
            if p:
                from pathlib import Path
                return Path(p).stem
        return ''

    def _export_selected_region(self):
        import json
        from pathlib import Path
        from tkinter import filedialog, messagebox
        selected = self._tree.selection()
        if not selected:
            messagebox.showwarning('No Selection', 'Select a region first')
            return
        item = selected[0]
        region_id = int(item)
        if not self.fov_preview or not self.fov_preview._roi_manager:
            return
        regions = self.fov_preview._roi_manager.get_all_regions()
        region = next((r for r in regions if r.get('id') == region_id), None)
        if not region:
            messagebox.showerror('Error', 'Region not found')
            return
        name = region.get('name', '')
        init_name = f'{name}.geojson' if name else None
        geojson_file = filedialog.asksaveasfilename(
            title='Export Region as GeoJSON',
            initialfile=init_name,
            defaultextension='.geojson',
            filetypes=[('GeoJSON files', '*.geojson'), ('JSON files', '*.json'), ('All files', '*.*')])
        if not geojson_file:
            return
        try:
            payload = self.fov_preview._roi_manager.to_geojson([region_id])
            feature = payload['features'][0]
            with open(geojson_file, 'w', encoding='utf-8') as f:
                json.dump(feature, f, indent=2)
            messagebox.showinfo('Export Success', f"Region exported to:\n{Path(geojson_file).name}")
            print(f'[Export] Region GeoJSON: {geojson_file}')
        except Exception as e:
            import traceback
            messagebox.showerror('Export Error', f'Failed to export: {e}')
            traceback.print_exc()

    def _export_all_rois_csv(self):
        import csv
        from pathlib import Path
        from tkinter import filedialog, messagebox
        if not self.fov_preview or not self.fov_preview._roi_manager.regions:
            messagebox.showwarning('No Data', 'No regions to export.')
            return
        fov_stem = self._get_fov_stem()
        init_name = f'{fov_stem}_roi_data.csv' if fov_stem else 'roi_data.csv'
        csv_file = filedialog.asksaveasfilename(
            title='Export ROI Data',
            initialfile=init_name,
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if not csv_file:
            return
        try:
            regions = self.fov_preview._roi_manager.get_all_regions()
            rows = []
            max_exp = 0
            for region in regions:
                taus_fit = region.get('statistics', {}).get('taus_ns_fit', [])
                if taus_fit:
                    max_exp = max(max_exp, len(taus_fit))
            for region in regions:
                region_id = region.get('id', '')
                name = region.get('name', '')
                tool = region.get('tool', '')
                stats = region.get('statistics', {})
                tau_mean    = stats.get('tau_mean',    'N/A')
                tau_median  = stats.get('tau_median',  'N/A')
                tau_stdev   = stats.get('tau_stdev',   'N/A')
                photon_count = stats.get('photon_count', 'N/A')
                photon_stdev = stats.get('photon_stdev', 'N/A')
                tau_mean_fit = stats.get('tau_mean_fit', 'N/A')
                taus_fit     = stats.get('taus_ns_fit',  [])
                amps_fit     = stats.get('amps_fit',     [])
                chi2_r_fit   = stats.get('chi2_r_fit',   'N/A')
                row = [region_id, name, tool,
                       tau_mean, tau_median, tau_stdev, photon_count, photon_stdev,
                       tau_mean_fit, chi2_r_fit]
                for k in range(max_exp):
                    row.append(taus_fit[k] if k < len(taus_fit) else 'N/A')
                    row.append(amps_fit[k] if k < len(amps_fit) else 'N/A')
                rows.append(row)
            if not rows:
                messagebox.showwarning('No Data', 'No regions to export.')
                return
            header = ['ID', 'Name', 'Type',
                      'Tau_mean_ns', 'Tau_median_ns', 'Tau_stdev_ns',
                      'Photon_count', 'Photon_stdev',
                      'Tau_mean_fit_ns', 'Chi2_r_fit']
            for k in range(1, max_exp + 1):
                header += [f'Tau{k}_fit_ns', f'Amp{k}_fit']
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            messagebox.showinfo('Export Success', f"ROI data exported to:\n{Path(csv_file).name}")
            print(f'[Export] ROI CSV: {csv_file}')
        except Exception as e:
            import traceback
            messagebox.showerror('Export Error', f'Failed to export: {e}')
            traceback.print_exc()

    def _export_all_rois_geojson(self):
        import json
        from pathlib import Path
        from tkinter import filedialog, messagebox
        if not self.fov_preview or not self.fov_preview._roi_manager.regions:
            messagebox.showwarning('No Data', 'No regions to export.')
            return
        fov_stem = self._get_fov_stem()
        init_name = f'{fov_stem}_all_rois.geojson' if fov_stem else 'all_rois.geojson'
        geojson_file = filedialog.asksaveasfilename(
            title='Export ROI Data as GeoJSON',
            initialfile=init_name,
            defaultextension='.geojson',
            filetypes=[('GeoJSON files', '*.geojson'), ('JSON files', '*.json'), ('All files', '*.*')])
        if not geojson_file:
            return
        try:
            payload = self.fov_preview._roi_manager.to_geojson()
            features = payload['features']
            if not features:
                messagebox.showwarning('No Data', 'No regions to export.')
                return
            with open(geojson_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            messagebox.showinfo('Export Success', f"ROI data exported to:\n{Path(geojson_file).name}\n({len(features)} regions)")
            print(f'[Export] ROI GeoJSON: {geojson_file}')
        except Exception as e:
            import traceback
            messagebox.showerror('Export Error', f'Failed to export: {e}')
            traceback.print_exc()

    def _import_rois_geojson(self):
        import json
        from tkinter import filedialog, messagebox
        if not self.fov_preview or not self.fov_preview._roi_manager:
            messagebox.showwarning('Not Ready', 'FOV preview not initialized')
            return
        geojson_file = filedialog.askopenfilename(
            title='Import ROI Data from GeoJSON',
            filetypes=[('GeoJSON files', '*.geojson'), ('JSON files', '*.json'), ('All files', '*.*')])
        if not geojson_file:
            return
        try:
            with open(geojson_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            imported_ids = self.fov_preview._roi_manager.add_geojson(data)
            imported_count = len(imported_ids)
            if imported_count > 0:
                self.fov_preview._redraw_region_overlays()
                self.fov_preview._save_regions_update()
                self._refresh_region_list()
                messagebox.showinfo('Import Success', f"Imported {imported_count} region(s) from {data.get('type', 'GeoJSON file')}")
            else:
                messagebox.showwarning('Import Failed', 'No regions could be imported')
        except json.JSONDecodeError as e:
            messagebox.showerror('JSON Error', f'Invalid JSON file: {e}')
        except Exception as e:
            import traceback
            messagebox.showerror('Import Error', f'Failed to import: {e}')
            traceback.print_exc()

    def _refresh_region_list(self):
        import tkinter as tk
        self._refreshing = True
        try:
            selected_id = None
            sel = self._tree.selection()
            if sel:
                selected_id = sel[0]
            for item in self._tree.get_children():
                self._tree.delete(item)
            if not self.fov_preview or not self.fov_preview._roi_manager:
                return
            for region in self.fov_preview._roi_manager.get_all_regions():
                region_id = region['id']
                name = region['name']
                tool_type = region['tool'].upper()
                color = self.fov_preview._roi_manager.get_color(region_id)
                tau_mean = '-'
                tau_med = '-'
                tau_stdev = '-'
                photon_count = '-'
                photon_stdev = '-'
                if self.fov_preview._lifetime_map is not None:
                    try:
                        mask = self.fov_preview._roi_manager.compute_region_mask(
                            region_id, self.fov_preview._lifetime_map.shape
                        )
                        if mask is not None:
                            lifetime_in_region = self.fov_preview._lifetime_map[mask]
                            valid = lifetime_in_region[~np.isnan(lifetime_in_region)]
                            if valid.size > 0:
                                tau_mean_val = float(np.mean(valid))
                                tau_med_val = float(np.median(valid))
                                tau_stdev_val = float(np.std(valid))
                                intensity_map = self.fov_preview._intensity_map
                                if intensity_map is not None and intensity_map.shape == self.fov_preview._lifetime_map.shape:
                                    intensity_in_region = intensity_map[mask]
                                    photon_count_val = int(intensity_in_region.sum())
                                    photon_stdev_val = float(np.std(intensity_in_region))
                                else:
                                    photon_count_val = int(valid.size)
                                    photon_stdev_val = float(np.sqrt(photon_count_val))
                                tau_mean = f'{tau_mean_val:.2f}'
                                tau_med = f'{tau_med_val:.2f}'
                                tau_stdev = f'{tau_stdev_val:.2f}'
                                photon_count = str(photon_count_val)
                                photon_stdev = f'{photon_stdev_val:.1f}'
                                region['statistics'] = {
                                    'tau_mean': tau_mean_val,
                                    'tau_median': tau_med_val,
                                    'tau_stdev': tau_stdev_val,
                                    'photon_count': photon_count_val,
                                    'photon_stdev': photon_stdev_val,
                                }
                    except Exception as e:
                        print(f'[ROI] Could not compute stats: {e}')
                values = (name, tool_type, tau_mean, tau_med, tau_stdev, photon_count, photon_stdev)
                self._tree.insert('', 'end', iid=str(region_id), values=values, tags=(f'color_{region_id}',))
                self._tree.tag_configure(f'color_{region_id}', foreground=color)
                if selected_id is not None and str(region_id) == selected_id:
                    self._tree.selection_set(str(region_id))
                self._tree.update_idletasks()
        finally:
            self._refreshing = False
    def add_region_from_drawing(self, tool_type: str, coords: List[List[float]]):
        if not self.fov_preview:
            return
        self._region_counter += 1
        name = f'{tool_type.capitalize()}-{self._region_counter}'
        region_id = self.fov_preview._roi_manager.add_region(name, tool_type, coords)
        self.fov_preview._redraw_region_overlays()
        self.fov_preview._save_regions_update()
        self._refresh_region_list()
        self._status.set(f'Added region: {name}')

    def _fit_roi_decay(self):
        import tkinter as tk
        from tkinter import messagebox
        from pathlib import Path
        if not self.fov_preview:
            return
        try:
            selected = self._tree.selection()
            if not selected:
                messagebox.showwarning('No Region', 'Select a region in the list first.')
                return
            selected_ids = []
            for iid in selected:
                try:
                    selected_ids.append(int(iid))
                except ValueError:
                    pass
            all_regions = self.fov_preview._roi_manager.get_all_regions()
            regions = [r for r in all_regions if r['id'] in selected_ids]
            if not regions:
                messagebox.showwarning('Region Not Found',
                                       'The selected region could not be found. '
                                       'Try refreshing the region list.')
                return
            region_id   = regions[0]['id']
            region_name = (regions[0]['name'] if len(regions) == 1
                           else f'{len(regions)} regions (merged)')
            if not callable(getattr(self, 'get_fit_params', None)) or \
               not callable(getattr(self, 'run_with_progress', None)):
                messagebox.showwarning('Not Ready',
                                       'Run a whole-FOV fit first - fit parameters '
                                       'are needed to re-fit the ROI decay.')
                return
            params = self.get_fit_params()
            ptu_path = params.get('ptu_path') or getattr(self.fov_preview, '_ptu_path', None)
            if not ptu_path or not Path(ptu_path).exists():
                messagebox.showwarning('No PTU',
                                       'No PTU file loaded - select a PTU file and run '
                                       'a fit before using ROI decay fitting.')
                return
        except Exception as _setup_exc:
            messagebox.showerror('Fit ROI Decay - Setup Error',
                                 f"Could not prepare parameters:\n{_setup_exc}")
            import traceback as _tb
            _tb.print_exc()
            return
        region_name = (regions[0].get('name', f'Region {region_id}')
                       if len(regions) == 1
                       else f'{len(regions)} regions (merged)')
        irf_cached  = getattr(self.fov_preview, '_irf_prompt', None)
        params = _ask_roi_fit_options(params)
        if params is None:
            return

        def task(progress_callback=None, cancel_event=None):
            from flimkit.formats import FLIMFile
            from flimkit.FLIM.fitters import fit_summed
            ptu = FLIMFile(ptu_path, verbose=False)
            n_bins    = ptu.n_bins
            tcspc_res = ptu.tcspc_res
            channel   = params.get('channel')
            if progress_callback:
                progress_callback(1, 4)
            stack = ptu.pixel_stack(channel=channel, binning=1)
            img_shape = (stack.shape[0], stack.shape[1])
            union_mask = np.zeros(img_shape, dtype=bool)
            for r in regions:
                m = self.fov_preview._roi_manager.compute_region_mask(r['id'], img_shape)
                if m is not None:
                    union_mask |= m
            if not union_mask.any():
                raise ValueError('ROI mask is empty - region(s) may be outside the image bounds.')
            roi_decay = stack[union_mask].sum(axis=0).astype(float)
            if roi_decay.max() == 0:
                raise ValueError('ROI contains no photons.')
            if progress_callback:
                progress_callback(2, 4)
            irf_prompt = irf_cached
            if irf_prompt is None or len(irf_prompt) != n_bins:
                from flimkit.FLIM.irf_tools import gaussian_irf
                decay_peak = int(np.argmax(roi_decay))
                fwhm_bins  = max(1.0, 0.2e-9 / tcspc_res)
                irf_prompt = gaussian_irf(n_bins, decay_peak, fwhm_bins)
                irf_source = 'gaussian (no IRF cached)'
            else:
                irf_source = 'from main fit'
            if progress_callback:
                progress_callback(3, 4)
            popt, summary = fit_summed(
                roi_decay, tcspc_res, n_bins, irf_prompt,
                has_tail=False, fit_bg=True, fit_sigma=False,
                n_exp=params['n_exp'],
                tau_min_ns=params['tau_min'],
                tau_max_ns=params['tau_max'],
                cost_function=params['cost_function'],
            )
            if progress_callback:
                progress_callback(4, 4)
            return {
                'region_id':   region_id,
                'region_name': region_name,
                'region_ids':  [r['id'] for r in regions],
                'decay':       roi_decay,
                'time_ns':     ptu.time_ns,
                'irf_prompt':  irf_prompt,
                'irf_source':  irf_source,
                'popt':        popt,
                'summary':     summary,
                'n_exp':       params['n_exp'],
            }

        def on_done(result):
            if result is None:
                return
            summary = result['summary']
            taus = list(summary.get('taus_ns', []))
            amps = list(summary.get('amps',    []))
            tau_mean_fit = (float(np.dot(taus, amps) / np.sum(amps))
                            if len(taus) > 0 and len(amps) > 0 else None)
            for rid in result.get('region_ids', [result['region_id']]):
                region_obj = next((r for r in self.fov_preview._roi_manager.get_all_regions()
                                   if r['id'] == rid), None)
                if region_obj is not None:
                    stats = region_obj.get('statistics', {})
                    stats['tau_mean_fit']  = tau_mean_fit
                    stats['taus_ns_fit']   = taus
                    stats['amps_fit']      = amps
                    stats['chi2_r_fit']    = summary.get('reduced_chi2_tail')
                    region_obj['statistics'] = stats
            key = tuple(sorted(result.get('region_ids', [result['region_id']])))
            self._last_fit_results[key] = result
            self._refresh_region_list()
            self._show_roi_fit_result(result)
        self.run_with_progress(
            task,
            task_name=f'ROI Decay Fit - {region_name}',
            on_done=on_done,
        )

    def _view_last_fit_result(self):
        from tkinter import messagebox
        if not self.fov_preview:
            return
        selected = self._tree.selection()
        if not selected:
            messagebox.showwarning('No Region', 'Select a region in the list first.')
            return
        selected_ids = []
        for iid in selected:
            try:
                selected_ids.append(int(iid))
            except ValueError:
                pass
        key = tuple(sorted(selected_ids))
        result = self._last_fit_results.get(key)
        if result is None:
            messagebox.showinfo(
                'No Fit Cached',
                'No fit result cached for this selection.\n'
                'Run \u22cf Fit ROI Decay first, or reload the session and refit '
                'to regenerate the plot (numeric stats are saved in the .npz).')
            return
        self._show_roi_fit_result(result)

    def _show_roi_fit_result(self, result: dict):
        _show_fit_result_window(result)

    def grid(self, **kw):
        self.frame.grid(**kw)
