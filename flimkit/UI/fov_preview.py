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
from flimkit.UI.roi_tools import RoiManager, RoiAnalysisPanel

class FOVPreviewPanel:

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        from matplotlib.gridspec import GridSpec
        self._fig = Figure(figsize=(10, 8), dpi=100, facecolor='black')
        self._decay_visible = True
        self._display_mode = 'flim'
        gs = GridSpec(3, 3, figure=self._fig, height_ratios=[1, 0.6, 0.3], width_ratios=[1, 1, 0.05], hspace=0.38, wspace=0.15)
        
        self._ax_img = self._fig.add_subplot(gs[0, 0])
        self._ax_flim = self._fig.add_subplot(gs[0, 1])
        self._ax_cbar = self._fig.add_subplot(gs[0, 2])
        self._ax_decay = self._fig.add_subplot(gs[1, :])
        self._ax_resid = self._fig.add_subplot(gs[2, :], sharex=self._ax_decay)
        for _ax in (self._ax_img, self._ax_flim):
            _ax.set_facecolor('black')
        self._ax_decay.set_facecolor('white')
        self._ax_decay.tick_params(colors='white')
        self._ax_decay.xaxis.label.set_color('white')
        self._ax_decay.yaxis.label.set_color('white')
        self._ax_decay.title.set_color('white')
        self._ax_resid.set_facecolor('white')
        self._ax_resid.tick_params(colors='white')
        self._ax_resid.xaxis.label.set_color('white')
        self._ax_resid.yaxis.label.set_color('white')
        self._strip_image_axes(self._ax_img)
        self._strip_image_axes(self._ax_flim)
        
        self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=self.frame)
        self._canvas_mpl.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self._status = tk.StringVar(value='No FOV loaded')
        ttk.Label(self.frame, textvariable=self._status, foreground='grey', font=('Courier', 8)).grid(
            row=1, column=0, sticky='w', padx=4, pady=(2, 4))

        ctrl_frame = ttk.LabelFrame(self.frame, text='FLIM Color Scale', padding=4)
        ctrl_frame.grid(row=2, column=0, sticky='ew', padx=4, pady=(0, 4))
        ctrl_frame.columnconfigure(1, weight=1)
        ctrl_frame.grid_remove()
        self._ctrl_frame = ctrl_frame
        
        ttk.Label(ctrl_frame, text='τ range (ns):').grid(row=0, column=0, sticky='w')
        ttk.Label(ctrl_frame, text='Min:').grid(row=0, column=1, sticky='w', padx=(10, 2))
        self._sv_tau_min = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self._sv_tau_min, width=6).grid(row=0, column=2, sticky='w', padx=2)
        ttk.Label(ctrl_frame, text='Max:').grid(row=0, column=3, sticky='w', padx=(10, 2))
        self._sv_tau_max = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self._sv_tau_max, width=6).grid(row=0, column=4, sticky='w', padx=2)
        ttk.Button(ctrl_frame, text='Auto', width=6, command=self._auto_detect_scale).grid(row=0, column=5, sticky='w', padx=2)
        
        ttk.Label(ctrl_frame, text='Γ:').grid(row=1, column=0, sticky='w')
        self._sv_gamma = tk.StringVar(value='1.0')
        ttk.Entry(ctrl_frame, textvariable=self._sv_gamma, width=6).grid(row=1, column=2, sticky='w', padx=2)
        
        ttk.Label(ctrl_frame, text='Colormap:').grid(row=1, column=3, sticky='w', padx=(10, 2))
        self._sv_cmap = tk.StringVar(value='viridis')
        self._cmap_combo = ttk.Combobox(ctrl_frame, textvariable=self._sv_cmap, 
                                 state='readonly', width=10)
        self._cmap_combo.grid(row=1, column=4, sticky='w', padx=2)
        
        self._cmap_combo['values'] = list(flim_display.COLORMAPS.keys())
        
        ttk.Button(ctrl_frame, text='Update', width=8, command=self._update_flim_display).grid(row=1, column=5, sticky='w', padx=2)

        self._bv_show_decay = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text='Show Decay Plot',
                        variable=self._bv_show_decay,
                        command=self._toggle_decay).grid(
            row=2, column=0, columnspan=3, sticky='w', pady=(4, 0))

        self._sv_display_mode = tk.StringVar(value='flim')
        dm_frame = ttk.Frame(ctrl_frame)
        dm_frame.grid(row=2, column=3, columnspan=3, sticky='w', pady=(4, 0))
        ttk.Label(dm_frame, text='View:').pack(side='left', padx=(0, 4))
        ttk.Radiobutton(dm_frame, text='FLIM', variable=self._sv_display_mode,
                        value='flim', command=self._on_display_mode_changed).pack(side='left')
        ttk.Radiobutton(dm_frame, text='Intensity', variable=self._sv_display_mode,
                        value='intensity', command=self._on_display_mode_changed).pack(side='left')

        self._ptu_path = None
        self._lifetime_map = None
        self._intensity_map = None
        self._flim_cbar = None
        self._flim_color_scale = {
            'vmin': None,
            'vmax': None,
            'gamma': 1.0,
            'cmap': 'viridis',
        }
        self._n_exp = 1
        self._irf_prompt = None
        self._roi_manager = RoiManager()
        self._roi_patches = {}
        self._drawing_mode = tk.StringVar(value='select')
        self._is_drawing = False
        self._draw_coords = []
        self._temp_line = None
        self._mouse_press_event = None
        self._roi_analysis_panel = None
        self._roi_drag = None
        self._setup_drawing_events()
        self._cached_decay_lines = []
        self._cached_decay_title = ''
        self._cached_decay_yscale = 'log'
        self._cached_resid_data = None
        self._setup_zoom()

    def load_fov(self, ptu_path: Optional[str]):
        if not ptu_path or not Path(ptu_path).exists():
            self._clear()
            self._status.set('Invalid PTU file')
            return

        try:
            self._ptu_path = ptu_path
            from flimkit.PTU.reader import PTUFile
            import numpy as np

            ptu = PTUFile(ptu_path, verbose=False)
            
            stack = ptu.pixel_stack(channel=None, binning=1)
            intensity = stack.sum(axis=2)
            decay = ptu.summed_decay(channel=None)
            time_ns = ptu.time_ns

            self._ax_img.clear()
            intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
            self._ax_img.imshow(intensity_clipped, cmap='inferno', origin='upper')
            self._ax_img.set_title('Intensity', fontsize=9, fontweight='bold', color='white')
            self._strip_image_axes(self._ax_img)

            self._ax_flim.clear()
            self._ax_flim.text(0.5, 0.5, 'Waiting for fit...', ha='center', va='center',
                              transform=self._ax_flim.transAxes, fontsize=9, color='white')
            self._ax_flim.set_title('FLIM Lifetime', fontsize=10, fontweight='bold', color='white')

            self._ax_decay.clear()
            self._ax_decay.set_facecolor('white')
            self._ax_decay.semilogy(time_ns, decay, color='steelblue', linewidth=1.5)
            self._ax_decay.set_title('Summed Decay', fontsize=10, fontweight='bold', color='white')
            self._ax_decay.set_xlabel('Time (ns)', color='white')
            self._ax_decay.set_ylabel('Photon Count', color='white')
            self._ax_decay.grid(True, alpha=0.3)
            self._ax_decay.tick_params(labelsize=8, colors='white')

            self._ax_resid.clear()
            self._ax_resid.set_facecolor('white')
            self._ax_resid.tick_params(labelsize=7, colors='white')
            self._ax_resid.grid(True, alpha=0.3)
            self._cached_resid_data = None

            self._redraw_region_overlays()
            self._canvas_mpl.draw_idle()

            n_photons = int(decay.sum())
            img_shape = intensity.shape
            self._status.set(f"✓ {Path(ptu_path).name} | {img_shape[0]}×{img_shape[1]}px | {n_photons} photons")

        except Exception as e:
            self._clear()
            self._status.set(f"Error loading FOV: {str(e)[:50]}")

    def display_fit_results(self, ptu_path: str, fit_result: dict):
        try:
            from flimkit.PTU.reader import PTUFile
            import numpy as np

            global_summary = fit_result.get('global_summary', {})
            global_popt = fit_result.get('global_popt')
            irf_prompt = fit_result.get('irf_prompt')
            if irf_prompt is not None:
                self._irf_prompt = irf_prompt
            time_ns_from_result = fit_result.get('time_ns')
            decay_from_result = fit_result.get('decay')
            canvas = fit_result.get('canvas')
            
            
            if decay_from_result is not None and time_ns_from_result is not None:
                decay = decay_from_result
                time_ns = time_ns_from_result
            else:
                if ptu_path and Path(ptu_path).exists():
                    ptu = PTUFile(ptu_path, verbose=False)
                    decay = ptu.summed_decay(channel=None)
                    time_ns = ptu.time_ns
                else:
                    decay = None
                    time_ns = None
            
            # Get intensity image: prefer canvas (from tile fitting), then fit_result, then PTU fallback
            intensity = None
            if canvas is not None and 'intensity' in canvas:
                intensity = canvas['intensity']
            elif 'intensity' in fit_result:
                intensity = fit_result['intensity']
            elif ptu_path and Path(ptu_path).exists():
                ptu = PTUFile(ptu_path, verbose=False)
                stack = ptu.pixel_stack(channel=None, binning=1)
                intensity = stack.sum(axis=2)
            
            if intensity is None:
                intensity = np.ones((512, 512), dtype=np.float32)
            
            from flimkit.UI.flim_display import compute_intensity_weighted_lifetime
            
            pixel_maps = fit_result.get('pixel_maps')
            if pixel_maps is None and canvas is not None:
                # For tile fits, extract pixel_maps from canvas
                pixel_maps = {k: v for k, v in canvas.items() 
                             if k not in ('intensity', 'coverage')}
            
            nexp = global_summary.get('n_exp', len(global_summary.get('taus_ns', [])))
            if nexp == 0:
                # derive_global_tau schema (tile_fit) stores tau1_mean_ns / tau2_mean_ns ...
                # rather than a taus_ns list.  Count how many tau{k}_mean_ns keys exist.
                nexp = sum(1 for k in range(1, 4) if f'tau{k}_mean_ns' in global_summary)
            lifetime_map = None
            
            if pixel_maps and nexp > 0:
                try:
                    lifetime_map = compute_intensity_weighted_lifetime(
                        pixel_maps, intensity, n_exp=nexp
                    )
                except Exception as e:
                    print(f"  - Warning: Could not compute lifetime map: {e}")
                    lifetime_map = None

            # Upsample lifetime map to full-res intensity shape if they differ
            # (happens when per-pixel fitting used binning > 1)
            if (lifetime_map is not None and intensity is not None
                    and lifetime_map.shape != intensity.shape[:2]):
                try:
                    import cv2 as _cv2
                    th, tw = intensity.shape[:2]
                    lifetime_map = _cv2.resize(
                        lifetime_map.astype(np.float32), (tw, th),
                        interpolation=_cv2.INTER_NEAREST)
                except Exception as _upe:
                    print(f"  - Could not upsample lifetime_map: {_upe}")

            self._lifetime_map = lifetime_map
            self._intensity_map = intensity
            self._n_exp = nexp

            if 'intensity' not in fit_result and intensity is not None:
                fit_result['intensity'] = intensity
            if lifetime_map is not None:
                fit_result['lifetime'] = lifetime_map
            if pixel_maps:
                fit_result['pixel_maps'] = pixel_maps
            
            # Extract fit data (taus_ns present in fit_summed schema only;
            # derive_global_tau / tile_fit uses tau1_mean_ns etc.).
            # nexp resolved above - do NOT overwrite it here.
            taus_fit = global_summary.get('taus_ns', [])
            model = global_summary.get('model')
            
            
            self._ax_img.clear()
            intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
            self._ax_img.imshow(intensity_clipped, cmap='inferno', origin='upper')
            self._ax_img.set_title('Intensity', fontsize=9, fontweight='bold', color='white')
            self._strip_image_axes(self._ax_img)

            self._ax_flim.clear()
            if self._lifetime_map is not None and np.any(~np.isnan(self._lifetime_map)):
                # Apply color scaling
                scaled = flim_display.apply_color_scale(
                    self._lifetime_map,
                    vmin=self._flim_color_scale['vmin'],
                    vmax=self._flim_color_scale['vmax'],
                    gamma=self._flim_color_scale['gamma'],
                )
                
                cmap = flim_display.get_colormap(self._flim_color_scale['cmap'])
                cmap.set_bad(color='black')
                im = self._ax_flim.imshow(scaled, cmap=cmap, origin='upper', vmin=0, vmax=1)
                self._ax_flim.set_title('FLIM Lifetime (ns)', fontsize=9, fontweight='bold', color='white')
                self._strip_image_axes(self._ax_flim)
                valid_data = self._lifetime_map[~np.isnan(self._lifetime_map)]
                if valid_data.size > 0:
                    data_min = np.min(valid_data)
                    data_max = np.max(valid_data)
                    self._ax_cbar.clear()
                    cbar = self._fig.colorbar(im, cax=self._ax_cbar)
                    cbar.set_label(f"τ (ns)", fontsize=8, color='white')
                    self._flim_cbar = cbar

                    n_ticks = 5
                    tick_positions = np.linspace(0, 1, n_ticks)
                    tick_values = data_min + tick_positions * (data_max - data_min)
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels([f"{v:.2f}" for v in tick_values], fontsize=7, color='white')
                    cbar.ax.tick_params(colors='white')
                else:
                    self._ax_cbar.clear()
            else:
                self._ax_flim.text(0.5, 0.6, 'No FLIM data', ha='center', va='center',
                                  transform=self._ax_flim.transAxes, fontsize=9, color='white')
                self._ax_flim.text(0.5, 0.35, '(enable per-pixel fitting)', ha='center', va='center',
                                  transform=self._ax_flim.transAxes, fontsize=8, color='white', style='italic')
                self._ax_flim.set_title('FLIM Lifetime', fontsize=10, fontweight='bold', color='white')
            
            self._redraw_region_overlays()

            self._ax_decay.clear()
            self._ax_decay.set_facecolor('white')
            
            if decay is None or len(decay) == 0:
                self._ax_decay.text(0.5, 0.5, 'No decay data', ha='center', va='center',
                                  transform=self._ax_decay.transAxes)
            else:
                self._ax_decay.semilogy(time_ns, decay, 'o-', color='steelblue',
                                        linewidth=1.5, markersize=3, label='Measured', alpha=0.7)
                
                if irf_prompt is not None and len(irf_prompt) > 0:
                    irf_max = irf_prompt.max()
                    if irf_max > 0:
                        # Scale IRF to ~20% of max decay for visibility
                        irf_scaled = (irf_prompt / irf_max) * decay.max() * 0.2
                        irf_time = time_ns[:len(irf_prompt)]
                        self._ax_decay.semilogy(irf_time, np.maximum(irf_scaled, 1e-2), 
                                              linewidth=2.0, color='orange', label='IRF', alpha=0.8)
                
                model = global_summary.get('model')
                if model is not None and len(model) > 0:
                    self._ax_decay.semilogy(time_ns, model, linewidth=2.0, 
                                          color='red', label='Fitted', alpha=0.8)
            
            self._ax_decay.set_title(f"Summed Decay{f' ({nexp}-exp fit)' if nexp > 0 else ''}", 
                                    fontsize=10, fontweight='bold', color='white')
            self._ax_decay.set_xlabel('Time (ns)', color='white')
            self._ax_decay.set_ylabel('Photon Count', color='white')
            if decay is not None and len(decay) > 0:
                self._ax_decay.legend(fontsize=8, loc='upper right', labelcolor='black')
            self._ax_decay.grid(True, alpha=0.3)
            self._ax_decay.tick_params(labelsize=8, colors='white')

            self._ax_resid.clear()
            self._ax_resid.set_facecolor('white')
            model_arr = global_summary.get('model')
            if (decay is not None and len(decay) > 0
                    and model_arr is not None
                    and len(model_arr) == len(decay)):
                with np.errstate(invalid='ignore', divide='ignore'):
                    resid = np.where(model_arr > 0,
                                     (decay - model_arr) / np.sqrt(model_arr),
                                     0.0)
                self._cached_resid_data = (time_ns.copy(), resid)
                self._ax_resid.plot(time_ns, resid, color='steelblue', linewidth=1.0)
                self._ax_resid.axhline(0, color='red', linewidth=1.0,
                                       linestyle='--', alpha=0.8)
                self._ax_resid.set_ylabel('Resid. (σ)', fontsize=7, color='white')
                chi2_r = global_summary.get('reduced_chi2_tail')
                if chi2_r is not None:
                    self._ax_resid.annotate(
                        f"χ²_r = {chi2_r:.3f}",
                        xy=(0.98, 0.85), xycoords='axes fraction',
                        ha='right', va='top', fontsize=7,
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', fc='#333333', alpha=0.7),
                    )
            else:
                self._cached_resid_data = None
            self._ax_resid.set_xlabel('Time (ns)', color='white')
            self._ax_resid.tick_params(labelsize=7, colors='white')
            self._ax_resid.grid(True, alpha=0.3)

            self._ctrl_frame.grid()
            
            self._canvas_mpl.draw_idle()

            status = f"✓ Fit complete"
            chi2_tail = global_summary.get('reduced_chi2_tail')
            if chi2_tail is not None:
                status += f" | χ²_r(tail)={chi2_tail:.3f}"
            if nexp > 0:
                taus = [global_summary.get(f'taus_ns', [])[i] if i < len(global_summary.get('taus_ns', [])) else None 
                        for i in range(nexp)]
                taus_str = ', '.join([f"{t:.3f}" for t in taus if t is not None])
                status += f" | τ=[{taus_str}] ns"
            self._status.set(status)
            print(f"  - Status: {status}")

        except Exception as e:
            import traceback
            print(f"[FOV Preview] Error displaying fit results:")
            traceback.print_exc()
            self._status.set(f"Error: {str(e)[:60]}")
            self._status.set(f"Error displaying fit: {str(e)[:50]}")

    def load_stitched_roi(self, output_dir: str):
        if not output_dir:
            self._clear()
            self._status.set('No output directory')
            return
        
        try:
            from pathlib import Path
            import numpy as np
            import tifffile
            
            out_path = Path(output_dir)
            
            # Find and load the stitched intensity TIFF file.
            # Stitch-pipeline writes  *_stitched_intensity.tif
            # Tile-fit pipeline writes *_intensity.tif (via save_assembled_maps)
            intensity_files = sorted(out_path.glob('*_stitched_intensity.tif'))
            if not intensity_files:
                intensity_files = sorted(out_path.glob('*_intensity.tif'))
            if not intensity_files:
                self._clear()
                self._status.set('No stitched image found')
                return
            
            intensity = tifffile.imread(str(intensity_files[0]))
            
            self._ax_img.clear()
            intensity_clipped = np.clip(intensity, 0, np.percentile(intensity, 99))
            self._ax_img.imshow(intensity_clipped, cmap='inferno', origin='upper')
            self._ax_img.set_title('Stitched ROI', fontsize=9, fontweight='bold', color='white')
            self._strip_image_axes(self._ax_img)
            
            lifetime_data = None
            lifetime_min, lifetime_max = None, None
            
            # Priority 1: Try full-range TIFF (best quality)
            lifetime_full = sorted(out_path.glob('*_tau_intensity_weighted_fullrange.tif'))
            if lifetime_full:
                try:
                    lifetime_data = tifffile.imread(str(lifetime_full[0])).astype(np.float32)
                    valid = np.isfinite(lifetime_data)
                    if valid.any():
                        lifetime_min = float(np.nanmin(lifetime_data[valid]))
                        lifetime_max = float(np.nanpercentile(lifetime_data[valid], 98))
                        print(f"  ✓ Loaded full-range lifetime: {lifetime_min:.2f}-{lifetime_max:.2f} ns")
                except Exception as e:
                    print(f"  - Could not load full-range lifetime: {e}")
            
            # Priority 2: Fall back to display-scaled TIFF
            if lifetime_data is None:
                lifetime_disp = sorted(out_path.glob('*_tau_intensity_weighted.tif'))
                if lifetime_disp:
                    try:
                        lifetime_data = tifffile.imread(str(lifetime_disp[0])).astype(np.float32)
                        # Convert uint16 back to ns (assumes 0-5 ns scale)
                        lifetime_data = lifetime_data / 65535.0 * 5.0
                        lifetime_min, lifetime_max = 0.0, 5.0
                        print(f"  ✓ Loaded display-scaled lifetime: 0-5 ns")
                    except Exception as e:
                        print(f"  - Could not load display-scaled lifetime: {e}")
            
            if lifetime_data is not None:
                self._ax_flim.clear()
                if lifetime_min is None or lifetime_max is None or lifetime_max <= lifetime_min:
                    lifetime_min = 0.0
                    lifetime_max = 5.0
                    if lifetime_max <= lifetime_min:
                        lifetime_max = lifetime_min + 0.1
                
                lifetime_norm = np.clip((lifetime_data - lifetime_min) / (lifetime_max - lifetime_min), 0, 1)
                
                im = self._ax_flim.imshow(lifetime_norm, cmap='viridis', origin='upper', vmin=0, vmax=1)
                self._ax_flim.set_title(f"FLIM Lifetime ({lifetime_min:.2f}-{lifetime_max:.2f} ns)",
                                       fontsize=9, fontweight='bold', color='white')
                self._strip_image_axes(self._ax_flim)
                
                self._ax_cbar.clear()
                cbar = self._fig.colorbar(im, cax=self._ax_cbar, label='τ (ns)')
                _min, _max = lifetime_min, lifetime_max
                def _fmt_ns(x, pos):
                    return f"{_min + x * (_max - _min):.1f}"
                from matplotlib.ticker import FuncFormatter
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ns))
                cbar.ax.tick_params(labelsize=7)
            else:
                self._ax_flim.clear()
                self._ax_flim.text(0.5, 0.5, 'Lifetime map not available', ha='center', va='center',
                                  transform=self._ax_flim.transAxes, fontsize=9, color='white')
                self._ax_flim.set_title('FLIM Lifetime', fontsize=10, fontweight='bold', color='white')
            
            self._ax_decay.clear()
            self._ax_decay.set_facecolor('white')
            self._ax_decay.text(0.5, 0.5, 'Per-tile fit complete ✓', 
                               ha='center', va='center', transform=self._ax_decay.transAxes,
                               fontsize=10, color='forestgreen', fontweight='bold')
            
            self._canvas_mpl.draw_idle()
            img_shape = intensity.shape
            self._status.set(f"✓ Tile fit | {img_shape[0]}×{img_shape[1]}px")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._clear()
            self._status.set(f"Error loading stitched: {str(e)[:50]}")

    def _clear(self):
        self._ax_img.clear()
        self._ax_flim.clear()
        self._ax_decay.clear()
        self._ax_decay.set_facecolor('white')
        self._ax_cbar.clear()
        self._flim_cbar = None
        self._ax_img.set_title('No FOV loaded', color='white')
        self._ax_flim.set_title('FLIM Lifetime', color='white')
        self._ax_decay.text(0.5, 0.5, 'Load a PTU file →', 
                           ha='center', va='center', transform=self._ax_decay.transAxes,
                           fontsize=10, color='#888')
        self._ctrl_frame.grid_remove()
        self._canvas_mpl.draw_idle()

    def _auto_detect_scale(self):
        import numpy as np
        
        if self._lifetime_map is None:
            return
        valid_data = self._lifetime_map[~np.isnan(self._lifetime_map)]
        if valid_data.size > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
            self._sv_tau_min.set(f"{vmin:.2f}")
            self._sv_tau_max.set(f"{vmax:.2f}")
            self._update_flim_display()

    def _update_flim_display(self):
        import numpy as np
        
        if self._lifetime_map is None or not np.any(~np.isnan(self._lifetime_map)):
            return
        
        try:
            try:
                vmin = float(self._sv_tau_min.get()) if self._sv_tau_min.get() else None
            except ValueError:
                vmin = None
            try:
                vmax = float(self._sv_tau_max.get()) if self._sv_tau_max.get() else None
            except ValueError:
                vmax = None
            try:
                gamma = float(self._sv_gamma.get())
                if gamma <= 0:
                    gamma = 1.0
            except ValueError:
                gamma = 1.0
            
            cmap_name = self._sv_cmap.get()
            
            self._flim_color_scale['vmin'] = vmin
            self._flim_color_scale['vmax'] = vmax
            self._flim_color_scale['gamma'] = gamma
            self._flim_color_scale['cmap'] = cmap_name
            
            self._save_color_scale_update()
            scaled = flim_display.apply_color_scale(
                self._lifetime_map, vmin=vmin, vmax=vmax, gamma=gamma
            )
            
            self._ax_flim.clear()
            self._ax_cbar.clear()
            self._flim_cbar = None
            cmap = flim_display.get_colormap(cmap_name)
            cmap.set_bad(color='black')
            
            im = self._ax_flim.imshow(scaled, cmap=cmap, origin='upper', vmin=0, vmax=1)
            self._ax_flim.set_title('FLIM Lifetime (ns)', fontsize=9, fontweight='bold', color='white')
            self._strip_image_axes(self._ax_flim)
            
            valid_data = self._lifetime_map[~np.isnan(self._lifetime_map)]
            if valid_data.size > 0:
                data_min = vmin if vmin is not None else np.min(valid_data)
                data_max = vmax if vmax is not None else np.max(valid_data)
                self._ax_cbar.clear()
                cbar = self._fig.colorbar(im, cax=self._ax_cbar)
                cbar.set_label('τ (ns)', fontsize=8, color='white')
                self._flim_cbar = cbar

                n_ticks = 5
                tick_positions = np.linspace(0, 1, n_ticks)
                tick_values = data_min + tick_positions * (data_max - data_min)
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels([f"{v:.2f}" for v in tick_values], fontsize=7, color='white')
                cbar.ax.tick_params(colors='white')
            else:
                self._ax_cbar.clear()
            
            self._redraw_region_overlays()
            
            self._canvas_mpl.draw_idle()
        except Exception as e:
            print(f"Error updating FLIM display: {e}")
    
    def _save_color_scale_update(self):
        try:
            if not self._ptu_path:
                return
            
            from pathlib import Path
            import json
            import numpy as np
            
            ptu_path = Path(self._ptu_path)
            session_file = ptu_path.parent / f"{ptu_path.stem}.roi_session.npz"
            
            if not session_file.exists():
                return
            existing_data = np.load(session_file, allow_pickle=True)
            session_data = {key: existing_data[key].item() if existing_data[key].ndim == 0 else existing_data[key]
                           for key in existing_data.files}
            session_data['fov_color_scale'] = json.dumps(self._flim_color_scale)
            np.savez_compressed(session_file, **session_data)
            print(f"[Color Scale] ✓ Saved to {session_file.name}")
            
        except Exception as e:
            print(f"[Color Scale] Could not save update: {e}")

    def _save_regions_update(self):
        try:
            if not self._ptu_path:
                return
            
            from pathlib import Path
            import json
            import numpy as np
            from datetime import datetime
            
            ptu_path = Path(self._ptu_path)
            session_file = ptu_path.parent / f"{ptu_path.stem}.roi_session.npz"
            
            if session_file.exists():
                existing_data = np.load(session_file, allow_pickle=True)
                session_data = {key: existing_data[key].item() if existing_data[key].ndim == 0 else existing_data[key]
                               for key in existing_data.files}
            else:
                session_data = {
                    'timestamp': datetime.now().isoformat(),
                    'source': str(self._ptu_path),
                    'form_state_json': json.dumps({}, default=str),
                }
                
                if self._lifetime_map is not None:
                    session_data['fov_lifetime_map'] = self._lifetime_map
                if self._intensity_map is not None:
                    session_data['fov_intensity_map'] = self._intensity_map
                session_data['fov_color_scale'] = json.dumps(self._flim_color_scale)
                session_data['fov_n_exp'] = self._n_exp
                if self._ptu_path:
                    session_data['fov_ptu_path'] = self._ptu_path
            
            session_data['fov_regions'] = self._roi_manager.to_json()
            np.savez_compressed(session_file, **session_data)
            print(f"[ROI Manager] ✓ Saved {len(self._roi_manager.regions)} region(s) to {session_file.name}")
            
        except Exception as e:
            print(f"[ROI Manager] Could not save regions: {e}")
    
    def _load_regions_from_json(self, json_str: str):
        try:
            self._roi_manager = RoiManager.from_json(json_str)
            print(f"[ROI Manager] ✓ Loaded {len(self._roi_manager.regions)} region(s)")
            self._redraw_region_overlays()
        except Exception as e:
            print(f"[ROI Manager] Could not load regions: {e}")
    
    def _redraw_region_overlays(self):
        import matplotlib.patches as mpatches
        from flimkit.UI.roi_tools import get_rectangle_patch, get_ellipse_patch, get_polygon_patch
        
        target_axes = [ax for ax in (self._ax_flim, self._ax_img) if ax.get_visible()]
        for patches in self._roi_patches.values():
            for patch in (patches if isinstance(patches, list) else [patches]):
                try:
                    patch.remove()
                except (ValueError, NotImplementedError):
                    pass
        self._roi_patches = {}
        for region in self._roi_manager.get_all_regions():
            region_id = region['id']
            tool_type = region['tool']
            coords = region['coords']
            color = self._roi_manager.get_color(region_id)
            linewidth = 2.5 if region_id == self._roi_manager.get_selected_id() else 1.5
            
            patches_for_region = []
            for ax in target_axes:
                try:
                    if tool_type == 'rect':
                        patch = get_rectangle_patch(coords, edgecolor=color, linewidth=linewidth)
                    elif tool_type == 'ellipse':
                        patch = get_ellipse_patch(coords, edgecolor=color, linewidth=linewidth)
                    elif tool_type in ('polygon', 'freehand'):
                        patch = get_polygon_patch(coords, edgecolor=color, linewidth=linewidth)
                    else:
                        continue
                    
                    ax.add_patch(patch)
                    patches_for_region.append(patch)
                except Exception as e:
                    print(f"[ROI] Could not draw region {region_id}: {e}")
            if patches_for_region:
                self._roi_patches[region_id] = patches_for_region
        
        self._canvas_mpl.draw_idle()

    @staticmethod
    def _strip_image_axes(ax):
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def _setup_zoom(self):
        self._zoom_cid = self._canvas_mpl.mpl_connect('scroll_event', self._on_scroll_zoom)
        self._pan_press_cid = self._canvas_mpl.mpl_connect('button_press_event', self._on_pan_press)
        self._pan_release_cid = self._canvas_mpl.mpl_connect('button_release_event', self._on_pan_release)
        self._pan_motion_cid = self._canvas_mpl.mpl_connect('motion_notify_event', self._on_pan_motion)
        self._pan_origin = None

    def _on_scroll_zoom(self, event):
        ax = event.inaxes
        if ax is None or ax not in (self._ax_img, self._ax_flim):
            return
        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.3
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_range = (xlim[1] - xlim[0]) * scale_factor
        y_range = (ylim[1] - ylim[0]) * scale_factor

        ax.set_xlim([event.xdata - x_range * (event.xdata - xlim[0]) / (xlim[1] - xlim[0]),
                     event.xdata + x_range * (xlim[1] - event.xdata) / (xlim[1] - xlim[0])])
        ax.set_ylim([event.ydata - y_range * (event.ydata - ylim[0]) / (ylim[1] - ylim[0]),
                     event.ydata + y_range * (ylim[1] - event.ydata) / (ylim[1] - ylim[0])])

        self._canvas_mpl.draw_idle()

    def _on_pan_press(self, event):
        if event.button == 1 and self._drawing_mode.get() != 'select':
            return
        ax = event.inaxes
        if ax is None or ax not in self._active_image_axes():
            return
        if event.xdata is None:
            return
        if event.button == 3:
            selected_id = self._roi_manager.get_selected_id()
            if selected_id is None:
                selected_id = self._hit_test_roi(event.xdata, event.ydata, ax)
            if selected_id is not None:
                self._start_roi_drag(selected_id, event.xdata, event.ydata)
                return
        self._pan_origin = (event.xdata, event.ydata, ax)

    def _on_pan_release(self, event):
        if self._roi_drag is not None:
            self._finish_roi_drag()
        self._pan_origin = None

    def _on_pan_motion(self, event):
        # ROI dragging takes priority
        if self._roi_drag is not None:
            if event.xdata is not None and event.ydata is not None:
                self._update_roi_drag(event.xdata, event.ydata)
            return
        if self._pan_origin is None:
            return
        ox, oy, ax = self._pan_origin
        if event.inaxes != ax or event.xdata is None:
            return
        dx = ox - event.xdata
        dy = oy - event.ydata
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self._canvas_mpl.draw_idle()

    def _hit_test_roi(self, x, y, ax):
        for region_id, patches in self._roi_patches.items():
            for patch in (patches if isinstance(patches, list) else [patches]):
                if patch.axes is ax and patch.contains_point(ax.transData.transform((x, y))):
                    return region_id
        return None

    def _start_roi_drag(self, region_id, x, y):
        self._roi_drag = {'id': region_id, 'ox': x, 'oy': y}
        self._roi_manager.select_region(region_id)
        self._redraw_region_overlays()
        if self._roi_analysis_panel:
            self._roi_analysis_panel._refresh_region_list()

    def _update_roi_drag(self, x, y):
        drag = self._roi_drag
        dx = x - drag['ox']
        dy = y - drag['oy']
        region = self._roi_manager.get_region(drag['id'])
        if region is None:
            self._roi_drag = None
            return
        new_coords = [[c[0] + dx, c[1] + dy] for c in region['coords']]
        self._roi_manager.update_region(drag['id'], coords=new_coords)
        drag['ox'] = x
        drag['oy'] = y
        self._redraw_region_overlays()

    def _finish_roi_drag(self):
        self._roi_drag = None
        self._save_regions_update()
        if self._roi_analysis_panel:
            self._roi_analysis_panel._refresh_region_list()

    def _on_display_mode_changed(self):
        new_mode = self._sv_display_mode.get()
        if new_mode == self._display_mode:
            return
        self._display_mode = new_mode
        self._rebuild_layout()

    def _toggle_decay(self):
        show = self._bv_show_decay.get()
        if show == self._decay_visible:
            return
        self._decay_visible = show
        self._rebuild_layout()

    def _rebuild_layout(self):
        """Rebuild the GridSpec layout based on decay visibility.

        Decay visible (default):
            Row 0: [Intensity] [FLIM] [cbar]     height_ratio 1
            Row 1: [       Decay          ]       height_ratio 0.6

        Decay hidden:
            Row 0: [    FLIM   ] [cbar]           height_ratio 1
            Row 1: [ Intensity ]                  height_ratio 0.5
        """
        from matplotlib.gridspec import GridSpec

        flim_title = self._ax_flim.get_title() if self._ax_flim.get_visible() else 'FLIM Lifetime (ns)'
        img_title = self._ax_img.get_title() if self._ax_img.get_visible() else 'Intensity'

        current_lines = []
        for line in self._ax_decay.get_lines():
            current_lines.append({
                'x': line.get_xdata().copy(),
                'y': line.get_ydata().copy(),
                'color': line.get_color(),
                'lw': line.get_linewidth(),
                'label': line.get_label(),
                'alpha': line.get_alpha(),
                'marker': line.get_marker(),
                'ms': line.get_markersize(),
            })
        if current_lines:
            self._cached_decay_lines = current_lines
            self._cached_decay_title = self._ax_decay.get_title()
            self._cached_decay_yscale = self._ax_decay.get_yscale()
        decay_lines = self._cached_decay_lines
        decay_title = self._cached_decay_title
        decay_yscale = self._cached_decay_yscale

        for ax in (self._ax_img, self._ax_flim, self._ax_cbar, self._ax_decay, self._ax_resid):
            ax.remove()

        if self._decay_visible:
            gs = GridSpec(3, 3, figure=self._fig,
                          height_ratios=[1, 0.6, 0.3],
                          width_ratios=[1, 1, 0.05],
                          hspace=0.38, wspace=0.15)
            self._ax_img   = self._fig.add_subplot(gs[0, 0])
            self._ax_flim  = self._fig.add_subplot(gs[0, 1])
            self._ax_cbar  = self._fig.add_subplot(gs[0, 2])
            self._ax_decay = self._fig.add_subplot(gs[1, :])
            self._ax_resid = self._fig.add_subplot(gs[2, :], sharex=self._ax_decay)
        else:
            if self._display_mode == 'intensity':
                gs = GridSpec(1, 1, figure=self._fig)
                self._ax_img   = self._fig.add_subplot(gs[0, 0])
                self._ax_flim  = self._fig.add_axes([0, 0, 0.01, 0.01])
                self._ax_flim.set_visible(False)
                self._ax_cbar  = self._fig.add_axes([0, 0, 0.01, 0.01])
                self._ax_cbar.set_visible(False)
            else:
                gs = GridSpec(1, 2, figure=self._fig,
                              width_ratios=[1, 0.05],
                              wspace=0.08)
                self._ax_flim  = self._fig.add_subplot(gs[0, 0])
                self._ax_cbar  = self._fig.add_subplot(gs[0, 1])
                self._ax_img   = self._fig.add_axes([0, 0, 0.01, 0.01])
                self._ax_img.set_visible(False)
            self._ax_decay = self._fig.add_axes([0, 0, 0.01, 0.01])
            self._ax_decay.set_visible(False)
            self._ax_resid = self._fig.add_axes([0, 0, 0.01, 0.01])
            self._ax_resid.set_visible(False)

        for _ax in (self._ax_img, self._ax_flim):
            _ax.set_facecolor('black')

        if self._ax_img.get_visible() and self._intensity_map is not None:
            import numpy as np
            intensity_clipped = np.clip(self._intensity_map, 0,
                                        np.percentile(self._intensity_map, 99))
            self._ax_img.imshow(intensity_clipped, cmap='inferno', origin='upper')
            self._ax_img.set_title(img_title, fontsize=9, fontweight='bold', color='white')
            self._strip_image_axes(self._ax_img)
        elif self._ax_img.get_visible():
            self._ax_img.set_title(img_title, fontsize=9, fontweight='bold', color='white')
            self._strip_image_axes(self._ax_img)

        if self._ax_flim.get_visible():
            if self._lifetime_map is not None:
                import numpy as np
                scaled = flim_display.apply_color_scale(
                    self._lifetime_map,
                    vmin=self._flim_color_scale['vmin'],
                    vmax=self._flim_color_scale['vmax'],
                    gamma=self._flim_color_scale['gamma'],
                )
                cmap = flim_display.get_colormap(self._flim_color_scale['cmap'])
                cmap.set_bad(color='black')
                im = self._ax_flim.imshow(scaled, cmap=cmap, origin='upper',
                                           vmin=0, vmax=1)
                if self._ax_cbar.get_visible():
                    self._ax_cbar.clear()
                    self._flim_cbar = None
                    valid = self._lifetime_map[~np.isnan(self._lifetime_map)]
                    if valid.size > 0:
                        cs = self._flim_color_scale
                        d_min = cs['vmin'] if cs['vmin'] is not None else float(np.min(valid))
                        d_max = cs['vmax'] if cs['vmax'] is not None else float(np.max(valid))
                        cbar = self._fig.colorbar(im, cax=self._ax_cbar)
                        cbar.set_label('τ (ns)', fontsize=8, color='white')
                        self._flim_cbar = cbar
                        n_ticks = 5
                        tp = np.linspace(0, 1, n_ticks)
                        tv = d_min + tp * (d_max - d_min)
                        cbar.set_ticks(tp)
                        cbar.set_ticklabels([f"{v:.2f}" for v in tv], fontsize=7, color='white')
                        cbar.ax.tick_params(colors='white')
            self._ax_flim.set_title(flim_title, fontsize=9, fontweight='bold', color='white')
            self._strip_image_axes(self._ax_flim)

        if self._decay_visible and decay_lines:
            for ld in decay_lines:
                self._ax_decay.plot(
                    ld['x'], ld['y'],
                    color=ld['color'], linewidth=ld['lw'],
                    label=ld['label'], alpha=ld['alpha'],
                    marker=ld['marker'], markersize=ld['ms'],
                )
            self._ax_decay.set_yscale(decay_yscale)
            self._ax_decay.set_title(decay_title, fontsize=10, fontweight='bold', color='white')
            self._ax_decay.set_xlabel('Time (ns)', color='white')
            self._ax_decay.set_ylabel('Photon Count', color='white')
            self._ax_decay.set_facecolor('white')
            self._ax_decay.tick_params(labelsize=8, colors='white')
            self._ax_decay.grid(True, alpha=0.3)

        if self._decay_visible and self._cached_resid_data is not None:
            t_r, res_r = self._cached_resid_data
            self._ax_resid.set_facecolor('white')
            self._ax_resid.plot(t_r, res_r, color='steelblue', linewidth=1.0)
            self._ax_resid.axhline(0, color='red', linewidth=1.0,
                                   linestyle='--', alpha=0.8)
            self._ax_resid.set_ylabel('Resid. (σ)', fontsize=7, color='white')
            self._ax_resid.set_xlabel('Time (ns)', color='white')
            self._ax_resid.tick_params(labelsize=7, colors='white')
            self._ax_resid.grid(True, alpha=0.3)

        self._redraw_region_overlays()
        self._setup_drawing_events()

        self._canvas_mpl.draw_idle()

    def _setup_drawing_events(self):
        # Disconnect old handlers to prevent accumulation across layout rebuilds
        for cid in getattr(self, '_draw_cids', []):
            self._canvas_mpl.mpl_disconnect(cid)
        self._draw_cids = [
            self._canvas_mpl.mpl_connect('button_press_event', self._on_draw_press),
            self._canvas_mpl.mpl_connect('motion_notify_event', self._on_draw_motion),
            self._canvas_mpl.mpl_connect('button_release_event', self._on_draw_release),
        ]
    
    def _active_image_axes(self):
        return {ax for ax in (self._ax_img, self._ax_flim) if ax.get_visible()}

    def _on_draw_press(self, event):
        if event.button != 1:
            return
        if not event.inaxes or event.inaxes not in self._active_image_axes():
            return
        
        mode = self._drawing_mode.get()
        if mode == 'select':
            return
        
        self._is_drawing = True
        self._draw_coords = [[event.xdata, event.ydata]]
        self._mouse_press_event = event
        print(f"[Drawing] Started {mode} at ({event.xdata:.1f}, {event.ydata:.1f})")
    
    def _on_draw_motion(self, event):
        if not self._is_drawing or not event.inaxes or event.inaxes not in self._active_image_axes():
            return
        
        mode = self._drawing_mode.get()
        
        if mode in ('rect', 'ellipse') and len(self._draw_coords) > 0:
            if self._temp_line is not None:
                try:
                    self._temp_line.remove()
                except:
                    pass
                self._temp_line = None
            
            x0, y0 = self._draw_coords[0]
            x1, y1 = event.xdata, event.ydata
            
            from matplotlib.patches import Rectangle
            preview = Rectangle((min(x0, x1), min(y0, y1)), 
                               abs(x1 - x0), abs(y1 - y0),
                               edgecolor='cyan', facecolor='none', 
                               linewidth=1, linestyle='', alpha=0.5)
            event.inaxes.add_patch(preview)
            self._temp_line = preview
            self._canvas_mpl.draw_idle()
        
        elif mode in ('polygon', 'freehand'):
            self._draw_coords.append([event.xdata, event.ydata])
    
    def _on_draw_release(self, event):
        if not self._is_drawing or not event.inaxes or event.inaxes not in self._active_image_axes():
            return
        
        mode = self._drawing_mode.get()
        
        if mode in ('rect', 'ellipse'):
            if len(self._draw_coords) > 0:
                self._draw_coords.append([event.xdata, event.ydata])
                self._finalize_drawing(mode)
        
        elif mode == 'polygon':
            if len(self._draw_coords) >= 3 and event.button == 3:
                self._finalize_drawing(mode)
        elif mode == 'freehand':
            if len(self._draw_coords) >= 3:
                self._finalize_drawing(mode)

        if self._temp_line is not None:
            try:
                self._temp_line.remove()
            except:
                pass
            self._temp_line = None
        
        self._is_drawing = False
    
    def _finalize_drawing(self, tool_type: str):
        if len(self._draw_coords) < 2:
            print(f"[Drawing] Cancelled {tool_type} (insufficient points)")
            self._draw_coords = []
            return
        
        try:
            region_id = self._roi_manager.add_region(
                f"{tool_type}-{len(self._roi_manager.regions) + 1}",
                tool_type,
                self._draw_coords
            )
            self._redraw_region_overlays()
            self._save_regions_update()
            print(f"[Drawing] Added {tool_type} region {region_id}")
            
            if self._roi_analysis_panel:
                self._roi_analysis_panel._refresh_region_list()
        except Exception as e:
            print(f"[Drawing] Error finalizing: {e}")
        finally:
            self._draw_coords = []

    def grid(self, **kw):
        self.frame.grid(**kw)
