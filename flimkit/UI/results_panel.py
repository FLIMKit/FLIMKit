from __future__ import annotations
import os
import re
import json
import time
import inspect
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import numpy as np
import matplotlib
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from flimkit.UI import flim_display


class ResultsPanel:
    def __init__(self, parent, root=None):
        self.parent = parent
        self.root = root  # Reference to main window for dialogs
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self._nb = ttk.Notebook(self.frame)
        self._nb.grid(row=0, column=0, sticky='nsew')

        self._build_progress()
        self._build_summary()
        self._build_images()

        self._imgs = []
        self._folder = None
        self._img_i = 0
        self._fit_result = None
        self._output_dir = None
        self._current_npz_path = None
        self._scan_name = None
        self._export_callback = None
        self._load_callback = None
        self._save_npz_callback = None

        self._status = tk.StringVar(value='Ready.')
        ttk.Label(self.frame, textvariable=self._status, foreground='grey').grid(
            row=1, column=0, sticky='w', padx=4, pady=(2, 4))

    def _build_progress(self):
        f = ttk.Frame(self._nb, padding=4)
        self._nb.add(f, text='  Progress  ')
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(
            f, state='disabled', wrap='word',
            font=('Courier', 9), background='#1e1e1e', foreground='#d4d4d4')
        self.log.grid(row=0, column=0, sticky='nsew')

        btn_bar = ttk.Frame(f)
        btn_bar.grid(row=1, column=0, sticky='ew', pady=(4, 0))
        ttk.Button(btn_bar, text='Save log...', command=self._save_log).pack(side='left',  padx=4)
        ttk.Button(btn_bar, text='Clear log', command=self._clear_log).pack(side='right', padx=4)

    def _clear_log(self):
        self.log.configure(state='normal')
        self.log.delete('1.0', tk.END)
        self.log.configure(state='disabled')

    def _save_log(self):
        text = self.log.get('1.0', tk.END)
        if not text.strip():
            messagebox.showinfo('Nothing to save', 'The log is empty.')
            return
        path = filedialog.asksaveasfilename(
            title='Save log as...',
            initialfile=f"{self._scan_name}_log.txt" if self._scan_name else '',
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('All', '*.*')])
        if path:
            Path(path).write_text(text, encoding='utf-8')
            self._status.set(f"Log saved → {Path(path).name}")

    def _on_export_clicked(self):
        try:
            print(f"[Export Button] Clicked - callback={self._export_callback is not None}, fit_result={self._fit_result is not None}, output_dir={self._output_dir}")
            if self._export_callback and self._fit_result and self._output_dir:
                print(f"[Export Button] Calling callback...")
                self._export_callback(self._fit_result, self._output_dir)
            else:
                print(f"[Export Button] Missing: callback={self._export_callback} fit_result={self._fit_result is not None} output_dir={self._output_dir}")
        except Exception as e:
            print(f"[Export Button Error] {e}")
            import traceback
            traceback.print_exc()
    
    def set_fit_result(self, fit_result: dict, output_dir: str, npz_path: str = None, scan_name: str = None):
        self._fit_result = fit_result
        self._output_dir = output_dir
        if npz_path:
            self._current_npz_path = npz_path
        if scan_name:
            self._scan_name = scan_name
        has_images = any(isinstance(v, np.ndarray) for v in (fit_result or {}).values())
        self._export_btn.configure(state='normal' if has_images else 'disabled')
    
    def set_export_callback(self, callback):
        self._export_callback = callback
    
    def set_load_callback(self, callback):
        self._load_callback = callback
    
    def set_save_npz_callback(self, callback):
        self._save_npz_callback = callback
    
    def _on_save_npz_clicked(self):
        try:
            if self._save_npz_callback and self._output_dir:
                self._save_npz_callback(self._output_dir)
        except Exception as e:
            print(f"[Save NPZ Error] {e}")
            import traceback
            traceback.print_exc()
    
    def _export_summed_csv(self):
        try:
            import csv
            from pathlib import Path
            
            init_name = f"{self._scan_name}_summed_fit.csv" if self._scan_name else None
            csv_file = filedialog.asksaveasfilename(
                title='Export Summed Fit Data',
                initialfile=init_name,
                defaultextension='.csv',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
                initialdir=self._output_dir)
            
            if not csv_file:
                return
            
            rows = []
            for item in self._tv.get_children():
                values = self._tv.item(item)['values']
                rows.append(values)  # (Parameter, Value, Unit)
            
            if not rows:
                messagebox.showwarning('No Data', 'No summary data to export.')
                return
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Parameter', 'Value', 'Unit'])
                writer.writerows(rows)
            
            messagebox.showinfo('Export Success', f"Summed fit data exported to:\n{Path(csv_file).name}")
            self._status.set(f"Exported → {Path(csv_file).name}")
            print(f"[Export] Summed fit CSV: {csv_file}")
            
        except Exception as e:
            messagebox.showerror('Export Error', f"Failed to export CSV:\n{e}")
            import traceback
            traceback.print_exc()
    
    def _load_fitted_data(self):
        npz_file = filedialog.askopenfilename(
            title='Load Fitted Data',
            filetypes=[('NumPy Archives', '*.npz'), ('All', '*.*')],
            defaultextension='.npz')
        if not npz_file:
            return
        
        if self._load_callback:
            self._load_callback(npz_file)

    def _build_summary(self):
        f = ttk.Frame(self._nb, padding=4)
        self._nb.add(f, text='  Fit Summary  ')
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        cols = ('Parameter', 'Value', 'Unit')
        tv = ttk.Treeview(f, columns=cols, show='headings')
        tv.heading('Parameter', text='Parameter', anchor='w')
        tv.heading('Value',     text='Value',     anchor='e')
        tv.heading('Unit',      text='Unit',      anchor='w')
        tv.column('Parameter', width=300, anchor='w', stretch=True)
        tv.column('Value',     width=110, anchor='e', stretch=False)
        tv.column('Unit',      width=70,  anchor='w', stretch=False)

        sb = ttk.Scrollbar(f, orient='vertical', command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.grid(row=0, column=0, sticky='nsew')
        sb.grid(row=0, column=1, sticky='ns')

        tv.tag_configure('odd',  background='#f5f7fa', foreground='#000000')
        tv.tag_configure('even', background='#ffffff', foreground='#000000')
        tv.tag_configure('warn', foreground='#c0550a', background='#fff8f0')
        self._tv = tv
        
        btn_bar = ttk.Frame(f)
        btn_bar.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(4, 0))
        self._export_btn = ttk.Button(btn_bar, text='Export Images...', 
                                     command=self._on_export_clicked, state='disabled')
        self._export_btn.pack(side='left', padx=4)

    def _build_images(self):
        f = ttk.Frame(self._nb, padding=4)
        self._nb.add(f, text='  Images  ')
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        fig = Figure(figsize=(5, 5), facecolor='#2b2b2b')
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor('#2b2b2b')
        self._ax.axis('off')
        self._canvas_mpl = FigureCanvasTkAgg(fig, master=f)
        self._canvas_mpl.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self._img_lbl = tk.StringVar(value='No images loaded')
        ttk.Label(f, textvariable=self._img_lbl, foreground='grey').grid(
            row=1, column=0, sticky='w', padx=4, pady=(2, 0))

        nav = ttk.Frame(f)
        nav.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        ttk.Button(nav, text='◀ Prev', command=self._img_prev).pack(side='left', padx=4)
        ttk.Button(nav, text='Next ▶', command=self._img_next).pack(side='left', padx=4)
        ttk.Button(nav, text='Open folder', command=self._open_folder).pack(side='right', padx=4)
        ttk.Button(nav, text='Save all...', command=self._save_all_imgs).pack(side='right', padx=4)
        ttk.Button(nav, text='Save image...', command=self._save_img).pack(side='right', padx=4)

    def populate_summary(self, rows: list):
        for item in self._tv.get_children():
            self._tv.delete(item)
        
        self._tv.update()
        for i, (param, val, unit) in enumerate(rows):
            tag = 'warn' if param.startswith('⚠') else ('odd' if i % 2 else 'even')
            self._tv.insert('', tk.END, values=(param, val, unit), tags=(tag,))
        self._tv.update_idletasks()
        
        if rows:
            self._nb.select(1)
            self._nb.update_idletasks()

    def set_status(self, msg: str):
        self._status.set(msg)

    def grid(self, **kw):
        self.frame.grid(**kw)

    def load_images(self, folder: Optional[str]):
        self._imgs = []
        if folder and Path(folder).is_dir():
            self._folder = folder
            for pat in ('*.png', '*.tif', '*.tiff'):
                self._imgs += sorted(Path(folder).glob(pat))
        self._img_i = 0
        self._draw_img()
        if self._imgs:
            self._nb.select(2)

    def _draw_img(self):
        self._ax.cla()
        self._ax.set_facecolor('#2b2b2b')
        self._ax.axis('off')
        if not self._imgs:
            self._img_lbl.set('No images found')
            self._ax.text(0.5, 0.5, 'No images found',
                          ha='center', va='center', color='grey', fontsize=11,
                          transform=self._ax.transAxes)
        else:
            path = self._imgs[self._img_i]
            self._img_lbl.set(f"{path.name}  ({self._img_i + 1}/{len(self._imgs)})")
            try:
                img = mpimg.imread(str(path))
                self._ax.imshow(img, aspect='equal')
            except Exception as e:
                self._ax.text(0.5, 0.5, f"Cannot load image:\n{e}",
                              ha='center', va='center', color='red',
                              fontsize=9, transform=self._ax.transAxes)
        self._canvas_mpl.draw_idle()

    def _img_prev(self):
        if self._imgs:
            self._img_i = (self._img_i - 1) % len(self._imgs)
            self._draw_img()

    def _img_next(self):
        if self._imgs:
            self._img_i = (self._img_i + 1) % len(self._imgs)
            self._draw_img()

    def _save_img(self):
        if not self._imgs:
            messagebox.showinfo('No image', 'No image is currently displayed.')
            return
        src = self._imgs[self._img_i]
        path = filedialog.asksaveasfilename(
            title='Save current image as...',
            initialfile=src.name,
            defaultextension=src.suffix,
            filetypes=[
                ('PNG',  '*.png'),
                ('TIFF', '*.tif *.tiff'),
                ('All',  '*.*'),
            ])
        if path:
            import shutil
            shutil.copy2(str(src), path)
            self._status.set(f"Image saved → {Path(path).name}")

    def _save_all_imgs(self):
        if not self._imgs:
            messagebox.showinfo('No images', 'No images are available to save.')
            return
        dest = filedialog.askdirectory(title='Save all images to...')
        if not dest:
            return
        import shutil
        dest_path = Path(dest)
        for img in self._imgs:
            shutil.copy2(str(img), str(dest_path / img.name))
        self._status.set(f"{len(self._imgs)} image(s) saved → {dest_path.name}/")

    def _open_folder(self):
        import subprocess, platform
        if not self._folder or not Path(self._folder).exists():
            messagebox.showinfo('No folder', 'No output folder available yet.')
            return
        s = platform.system()
        if   s == 'Darwin':  subprocess.Popen(['open',     self._folder])
        elif s == 'Windows': subprocess.Popen(['explorer', self._folder])
        else:                subprocess.Popen(['xdg-open', self._folder])
