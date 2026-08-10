import csv
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np


class AnisotropyTool(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title('Time-Resolved Anisotropy')
        self.geometry('1180x820')
        self.minsize(980, 700)
        self.result = None
        self.peak_bin = None
        self._colorbar = None
        self._closed = False
        self._poll_after_id = None
        self.protocol('WM_DELETE_WINDOW', self._close)
        self._build_controls()
        self._build_plot()

    def _build_controls(self):
        controls = ttk.Frame(self, padding=10)
        controls.pack(fill='x')
        controls.columnconfigure(1, weight=1)

        self.parallel_path = tk.StringVar()
        self.perpendicular_path = tk.StringVar()
        self.parallel_irf_path = tk.StringVar()
        self.perpendicular_irf_path = tk.StringVar()
        self.analysis_mode = tk.StringVar(value='direct')
        self.fixed_lifetime_ns = tk.DoubleVar(value=3.0)
        self.g_factor = tk.DoubleVar(value=1.0)
        self.parallel_exposure = tk.DoubleVar(value=1.0)
        self.perpendicular_exposure = tk.DoubleVar(value=1.0)
        self.parallel_channel = tk.IntVar(value=0)
        self.perpendicular_channel = tk.IntVar(value=0)
        self.background_start = tk.IntVar(value=2)
        self.background_stop = tk.IntVar(value=10)
        self.analysis_start_ns = tk.DoubleVar(value=0.0)
        self.analysis_stop_ns = tk.DoubleVar(value=8.0)
        self.spatial_window = tk.IntVar(value=5)
        self.stride = tk.IntVar(value=1)
        self.min_bin_photons = tk.DoubleVar(value=25.0)
        self.min_map_photons = tk.DoubleVar(value=500.0)
        self.auto_register = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Choose the parallel and perpendicular PTU files.')

        self._file_row(controls, 0, 'Parallel PTU', self.parallel_path)
        self._file_row(controls, 1, 'Perpendicular PTU', self.perpendicular_path)
        self._file_row(controls, 2, 'Parallel IRF export', self.parallel_irf_path)
        self._file_row(
            controls, 3, 'Perpendicular IRF export', self.perpendicular_irf_path)

        modes = ttk.LabelFrame(controls, text='Analysis method', padding=8)
        modes.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        ttk.Radiobutton(
            modes, text='Direct r(t) diagnostic (no IRF)',
            variable=self.analysis_mode, value='direct').pack(side='left')
        ttk.Radiobutton(
            modes, text='Preferred global fit (Lakowicz Section 11.2.2)',
            variable=self.analysis_mode, value='global').pack(side='left', padx=12)
        ttk.Button(modes, text='Method info...',
                   command=self._show_method_info).pack(side='left')

        settings = ttk.LabelFrame(controls, text='Analysis settings', padding=8)
        settings.grid(row=5, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        fields = [
            ('Known lifetime (ns, global fit)', self.fixed_lifetime_ns),
            ('G factor', self.g_factor),
            ('Parallel exposure (relative)', self.parallel_exposure),
            ('Perpendicular exposure (relative)', self.perpendicular_exposure),
            ('Parallel photon channel', self.parallel_channel),
            ('Perpendicular photon channel', self.perpendicular_channel),
            ('Background start bin', self.background_start),
            ('Background stop bin', self.background_stop),
            ('Post-peak start (ns)', self.analysis_start_ns),
            ('Post-peak stop (ns)', self.analysis_stop_ns),
            ('Spatial window', self.spatial_window),
            ('Stride', self.stride),
            ('Min photons / time bin', self.min_bin_photons),
            ('Min photons / map', self.min_map_photons),
        ]
        for index, (label, variable) in enumerate(fields):
            row, column = divmod(index, 3)
            base = column * 2
            ttk.Label(settings, text=label).grid(
                row=row, column=base, sticky='w', padx=(0, 4), pady=2)
            ttk.Entry(settings, textvariable=variable, width=12).grid(
                row=row, column=base + 1, sticky='w', padx=(0, 12), pady=2)
        ttk.Checkbutton(
            settings, text='Auto-register perpendicular image to parallel image',
            variable=self.auto_register).grid(
                row=5, column=0, columnspan=6, sticky='w', pady=(4, 0))

        note = ('File roles are explicit; FLIMKit does not infer them from names. '
                'G=1 is an assumption unless calibrated independently.')
        ttk.Label(controls, text=note, foreground='#555555').grid(
            row=6, column=0, columnspan=3, sticky='w', pady=(6, 0))

        actions = ttk.Frame(controls)
        actions.grid(row=7, column=0, columnspan=3, sticky='ew', pady=(8, 0))
        self.calculate_button = ttk.Button(
            actions, text='Calculate', command=self._start_analysis)
        self.calculate_button.pack(side='left')
        self.save_npz_button = ttk.Button(
            actions, text='Save NPZ...', command=self._save_npz, state='disabled')
        self.save_npz_button.pack(side='left', padx=(6, 0))
        self.save_csv_button = ttk.Button(
            actions, text='Save CSV...', command=self._save_csv, state='disabled')
        self.save_csv_button.pack(side='left', padx=(6, 0))
        ttk.Label(actions, textvariable=self.status).pack(side='left', padx=12)

    def _file_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky='ew', padx=6, pady=2)
        ttk.Button(parent, text='Browse...',
                   command=lambda: self._browse(variable)).grid(
                       row=row, column=2, sticky='e', pady=2)

    def _build_plot(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(10.8, 6.0), dpi=100,
                             constrained_layout=True)
        self.figure.get_layout_engine().set(w_pad=0.08, h_pad=0.05)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)
        self._draw_empty()

    def _draw_empty(self):
        titles = ['Parallel intensity', 'Perpendicular intensity',
                  'Summed anisotropy decay', 'Anisotropy map']
        for axis, title in zip(self.axes.flat, titles):
            axis.clear()
            axis.set_title(title)
            axis.text(0.5, 0.5, 'No result yet', ha='center', va='center',
                      transform=axis.transAxes, color='#777777')
            axis.set_axis_off()
        self.canvas.draw_idle()

    def _browse(self, variable):
        path = filedialog.askopenfilename(
            parent=self, title='Select PTU file',
            filetypes=[('PicoQuant PTU', '*.ptu'), ('All files', '*.*')])
        if path:
            variable.set(path)

    def _show_method_info(self):
        messagebox.showinfo(
            'Time-resolved anisotropy methods',
            'Direct r(t) diagnostic:\n'
            'r(t) = [I_parallel(t) - G I_perpendicular(t)] / '
            '[I_parallel(t) + 2 G I_perpendicular(t)]\n\n'
            'This ratio is useful for inspection and maps, but convolution and '
            'division do not commute. It should not be used to fit rotational '
            'correlation times near the IRF response.\n\n'
            'Preferred global fit:\n'
            'Fits the raw parallel and perpendicular photon counts together. '
            'It uses separate IRFs, a known fluorescence lifetime, fixed G and '
            'relative exposures, one rotational correlation time, one common '
            'IRF timing shift, and separate fitted backgrounds. Previous laser '
            'pulses are included.\n\n'
            'The reported r(0) is the resolved time-zero anisotropy. It is not '
            'automatically the fundamental anisotropy because very fast motion '
            'may be hidden by the IRF.\n\n'
            'Reference: Lakowicz, Principles of Fluorescence Spectroscopy, '
            'Chapter 11, Section 11.2.2, Preferred Analysis of TD Anisotropy Data.',
            parent=self)

    def _settings(self):
        parallel = Path(self.parallel_path.get()).expanduser()
        perpendicular = Path(self.perpendicular_path.get()).expanduser()
        if not parallel.is_file() or not perpendicular.is_file():
            raise ValueError('Choose two existing PTU files')
        if parallel.resolve() == perpendicular.resolve():
            raise ValueError('Parallel and perpendicular files must be different')
        analysis_mode = self.analysis_mode.get()
        if analysis_mode not in {'direct', 'global'}:
            raise ValueError('Choose a valid analysis method')
        parallel_irf = Path(self.parallel_irf_path.get()).expanduser()
        perpendicular_irf = Path(self.perpendicular_irf_path.get()).expanduser()
        if analysis_mode == 'global':
            if not parallel_irf.is_file() or not perpendicular_irf.is_file():
                raise ValueError(
                    'Preferred global fitting requires parallel and perpendicular IRF files')
            if parallel_irf.resolve() == perpendicular_irf.resolve():
                raise ValueError('Choose a separate IRF file for each polarization')
        fixed_lifetime_ns = self.fixed_lifetime_ns.get()
        if (analysis_mode == 'global'
                and (not np.isfinite(fixed_lifetime_ns)
                     or fixed_lifetime_ns <= 0)):
            raise ValueError(
                'Known fluorescence lifetime must be positive and finite')
        parallel_channel = self.parallel_channel.get()
        perpendicular_channel = self.perpendicular_channel.get()
        if parallel_channel < 0 or perpendicular_channel < 0:
            raise ValueError('Photon channels must be non-negative integers')
        background_start = self.background_start.get()
        background_stop = self.background_stop.get()
        if background_start < 0 or background_stop <= background_start:
            raise ValueError('Background bins must satisfy 0 <= start < stop')
        analysis_start = self.analysis_start_ns.get()
        analysis_stop = self.analysis_stop_ns.get()
        if (not np.isfinite(analysis_start) or analysis_start < 0
                or not np.isfinite(analysis_stop)
                or analysis_stop <= analysis_start):
            raise ValueError(
                'Post-peak times must be finite and satisfy 0 <= start < stop')
        window = self.spatial_window.get()
        stride = self.stride.get()
        if window < 1 or window % 2 == 0:
            raise ValueError('Spatial window must be a positive odd number')
        if stride < 1:
            raise ValueError('Stride must be positive')
        parallel_exposure = self.parallel_exposure.get()
        perpendicular_exposure = self.perpendicular_exposure.get()
        if (not np.isfinite(parallel_exposure) or parallel_exposure <= 0
                or not np.isfinite(perpendicular_exposure)
                or perpendicular_exposure <= 0):
            raise ValueError('Exposure values must be positive and finite')
        g_factor = self.g_factor.get()
        if not np.isfinite(g_factor) or g_factor <= 0:
            raise ValueError('G factor must be positive and finite')
        min_bin_photons = self.min_bin_photons.get()
        min_map_photons = self.min_map_photons.get()
        if (not np.isfinite(min_bin_photons) or min_bin_photons < 0
                or not np.isfinite(min_map_photons)
                or min_map_photons < 0):
            raise ValueError('Photon thresholds must be finite and non-negative')
        return {
            'parallel_path': parallel,
            'perpendicular_path': perpendicular,
            'analysis_mode': analysis_mode,
            'parallel_irf_path': parallel_irf if analysis_mode == 'global' else None,
            'perpendicular_irf_path': (
                perpendicular_irf if analysis_mode == 'global' else None),
            'fixed_lifetime_ns': fixed_lifetime_ns,
            'g_factor': g_factor,
            'parallel_exposure': parallel_exposure,
            'perpendicular_exposure': perpendicular_exposure,
            'parallel_channel': parallel_channel,
            'perpendicular_channel': perpendicular_channel,
            'background_bins': slice(background_start, background_stop),
            'analysis_start_ns': analysis_start,
            'analysis_stop_ns': analysis_stop,
            'spatial_window': window,
            'stride': stride,
            'min_bin_photons': min_bin_photons,
            'min_map_photons': min_map_photons,
            'auto_register': self.auto_register.get(),
        }

    def _start_analysis(self):
        try:
            settings = self._settings()
        except Exception as exc:
            messagebox.showerror('Invalid settings', str(exc), parent=self)
            return
        self.calculate_button.configure(state='disabled')
        self.status.set('Reading and analysing PTUs...')
        self._result_queue = queue.Queue()
        threading.Thread(target=self._analysis_worker, args=(settings,),
                         daemon=True).start()
        self._poll_after_id = self.after(100, self._poll_worker)

    def _analysis_worker(self, settings):
        try:
            result, peak_bin = run_analysis(settings)
        except Exception as exc:
            self._result_queue.put(('error', exc))
            return
        self._result_queue.put(('success', result, peak_bin))

    def _poll_worker(self):
        self._poll_after_id = None
        if self._closed:
            return
        try:
            message = self._result_queue.get_nowait()
        except queue.Empty:
            self._poll_after_id = self.after(100, self._poll_worker)
            return
        if message[0] == 'error':
            self._analysis_failed(message[1])
        else:
            self._analysis_finished(*message[1:])

    def _close(self):
        self._closed = True
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self.destroy()

    def _analysis_failed(self, exc):
        self.calculate_button.configure(state='normal')
        self.status.set('Analysis failed.')
        messagebox.showerror('Anisotropy error', str(exc), parent=self)

    def _analysis_finished(self, result, peak_bin):
        self.result = result
        self.peak_bin = peak_bin
        self.calculate_button.configure(state='normal')
        self.save_npz_button.configure(state='normal')
        self.save_csv_button.configure(state='normal')
        shift_y, shift_x = result.perpendicular_shift
        self.status.set(
            f'Done. Perpendicular shift: ({shift_y:.2f}, {shift_x:.2f}) px')
        self._draw_result()

    def _draw_result(self):
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        for axis in self.axes.flat:
            axis.clear()
        if getattr(self.result, 'polarized_fit', None) is not None:
            self._draw_global_fit()
            self.canvas.draw_idle()
            return
        self.axes[0, 0].imshow(self.result.parallel_intensity, cmap='gray')
        self.axes[0, 0].set_title('Parallel intensity')
        self.axes[0, 1].imshow(self.result.perpendicular_intensity, cmap='gray')
        self.axes[0, 1].set_title('Perpendicular intensity (registered)')
        for axis in self.axes[0]:
            axis.set_axis_off()

        time_relative = self.result.time_ns - self.result.time_ns[self.peak_bin]
        start_ns = self.result.metadata.get('analysis_start_ns', 0.0)
        stop_ns = self.result.metadata.get('analysis_stop_ns', 8.0)
        selected = (time_relative >= start_ns) & (time_relative < stop_ns)
        self.axes[1, 0].plot(
            time_relative[selected], self.result.anisotropy_decay[selected],
            color='#2468a2')
        self.axes[1, 0].set_xlim(start_ns, stop_ns)
        self.axes[1, 0].axhline(0.4, color='#999999', linestyle=':', linewidth=1)
        self.axes[1, 0].axhline(-0.2, color='#999999', linestyle=':', linewidth=1)
        self.axes[1, 0].set_xlabel('Time after peak (ns)')
        self.axes[1, 0].set_ylabel('Anisotropy r(t)')
        self.axes[1, 0].set_title('Summed anisotropy decay')

        finite_map = self.result.anisotropy_map[
            np.isfinite(self.result.anisotropy_map)]
        if finite_map.size:
            vmin = min(-0.2, float(np.percentile(finite_map, 2)))
            vmax = max(0.4, float(np.percentile(finite_map, 98)))
        else:
            vmin, vmax = -0.2, 0.4
        image = self.axes[1, 1].imshow(
            self.result.anisotropy_map, cmap='coolwarm', vmin=vmin, vmax=vmax)
        self.axes[1, 1].set_title(
            f'Anisotropy map\n{self.result.spatial_window}x'
            f'{self.result.spatial_window} window, stride {self.result.stride}')
        self.axes[1, 1].set_axis_off()
        self._colorbar = self.figure.colorbar(
            image, ax=self.axes[1, 1], fraction=0.04, pad=0.03,
            extend='both', location='left')
        self.canvas.draw_idle()

    def _draw_global_fit(self):
        fit = self.result.polarized_fit
        fit_bins = len(fit.parallel_model)
        time_relative = (
            self.result.time_ns[:fit_bins] - self.result.time_ns[self.peak_bin])
        parallel_observed = (
            self.result.parallel_decay[:fit_bins] + self.result.parallel_background)
        perpendicular_observed = (
            self.result.perpendicular_decay[:fit_bins]
            + self.result.perpendicular_background)
        channels = (
            (self.axes[0, 0], parallel_observed, fit.parallel_model,
             'Parallel global fit'),
            (self.axes[0, 1], perpendicular_observed, fit.perpendicular_model,
             'Perpendicular global fit'),
        )
        for axis, observed, model, title in channels:
            axis.semilogy(time_relative, np.maximum(observed, 1e-3),
                          color='#777777', linewidth=1, label='Measured')
            axis.semilogy(time_relative, np.maximum(model, 1e-3),
                          color='#2468a2', linewidth=1.5, label='Global model')
            axis.set_title(title)
            axis.set_xlabel('Time after peak (ns)')
            axis.set_ylabel('Photon counts')
            axis.legend(fontsize=8)

        self.axes[1, 0].plot(
            time_relative, fit.parallel_residual,
            color='#2468a2', linewidth=1, label='Parallel')
        self.axes[1, 0].plot(
            time_relative, fit.perpendicular_residual,
            color='#a34a28', linewidth=1, label='Perpendicular')
        self.axes[1, 0].axhline(0.0, color='#777777', linewidth=0.8)
        self.axes[1, 0].set_title('Residuals')
        self.axes[1, 0].set_xlabel('Time after peak (ns)')
        self.axes[1, 0].set_ylabel('Observed - model')
        self.axes[1, 0].legend(fontsize=8)

        summary = (
            'Preferred global polarized-decay fit\n\n'
            f'Fixed fluorescence lifetime: {fit.intensity_lifetime_ns:.4g} ns\n'
            f'Rotational correlation: {fit.rotational_correlation_ns:.4g} ns\n'
            f'Resolved r(0): {fit.initial_anisotropy:.4g}\n'
            f'Common IRF shift: {fit.common_irf_shift_bins:.4g} bins\n'
            f'Fitted backgrounds: {fit.parallel_background:.4g}, '
            f'{fit.perpendicular_background:.4g}\n'
            f'Poisson deviance: {fit.poisson_deviance:.4g}\n\n'
            'Lakowicz, Section 11.2.2\n'
            'Separate IRFs fitted simultaneously')
        if not getattr(fit, 'success', True):
            summary += ('\n\nWARNING: optimizer did not converge:\n'
                        + str(getattr(fit, 'message', 'unknown reason')))
        parameters_at_bounds = getattr(fit, 'parameters_at_bounds', ())
        if parameters_at_bounds:
            summary += ('\n\nWARNING: fit reached parameter bounds:\n'
                        + ', '.join(parameters_at_bounds))
        self.axes[1, 1].text(
            0.05, 0.95, summary, ha='left', va='top',
            transform=self.axes[1, 1].transAxes)
        self.axes[1, 1].set_axis_off()

    def _save_npz(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Save anisotropy result',
            defaultextension='.npz', filetypes=[('NPZ files', '*.npz')])
        if not path:
            return
        from flimkit.FLIM.anisotropy import save_anisotropy_npz
        self.result.metadata['peak_bin'] = int(self.peak_bin)
        save_anisotropy_npz(self.result, path)
        self.status.set(f'Saved {Path(path).name}')

    def _save_csv(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Save summed anisotropy decay',
            defaultextension='.csv', filetypes=[('CSV files', '*.csv')])
        if not path:
            return
        metadata = self.result.metadata
        shift_y, shift_x = self.result.perpendicular_shift
        provenance = {
            'parallel_file': metadata.get('parallel_file', ''),
            'perpendicular_file': metadata.get('perpendicular_file', ''),
            'parallel_role': metadata.get('parallel_role', 'parallel'),
            'perpendicular_role': metadata.get('perpendicular_role', 'perpendicular'),
            'g_factor': self.result.g_factor,
            'parallel_exposure': self.result.parallel_exposure,
            'perpendicular_exposure': self.result.perpendicular_exposure,
            'parallel_channel': metadata.get('parallel_channel', ''),
            'perpendicular_channel': metadata.get('perpendicular_channel', ''),
            'background_start_bin': metadata.get('background_start_bin', ''),
            'background_stop_bin': metadata.get('background_stop_bin', ''),
            'analysis_start_ns': metadata.get('analysis_start_ns', ''),
            'analysis_stop_ns': metadata.get('analysis_stop_ns', ''),
            'min_bin_photons': metadata.get('min_bin_photons', ''),
            'min_map_photons': metadata.get('min_map_photons', ''),
            'perpendicular_shift_y': shift_y,
            'perpendicular_shift_x': shift_x,
            'spatial_window': self.result.spatial_window,
            'stride': self.result.stride,
        }
        fieldnames = [
            'time_ns', 'time_after_peak_ns', 'parallel', 'perpendicular',
            'anisotropy', 'valid',
        ]
        fit = getattr(self.result, 'polarized_fit', None)
        fit_fields = []
        if fit is not None:
            fit_fields = [
                'parallel_observed', 'perpendicular_observed',
                'parallel_model', 'perpendicular_model',
                'parallel_residual', 'perpendicular_residual',
            ]
            provenance.update({
                'intensity_lifetime_ns': fit.intensity_lifetime_ns,
                'rotational_correlation_ns': fit.rotational_correlation_ns,
                'initial_anisotropy': fit.initial_anisotropy,
                'common_irf_shift_bins': fit.common_irf_shift_bins,
                'parallel_fit_background': fit.parallel_background,
                'perpendicular_fit_background': fit.perpendicular_background,
                'poisson_deviance': fit.poisson_deviance,
            })
        fieldnames.extend([*fit_fields, *provenance])
        peak_time = self.result.time_ns[self.peak_bin]
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, (time_ns, parallel, perpendicular, anisotropy) in enumerate(zip(
                    self.result.time_ns, self.result.parallel_decay,
                    self.result.perpendicular_decay,
                    self.result.anisotropy_decay)):
                fit_values = {}
                if fit is not None and index < len(fit.parallel_model):
                    fit_values = {
                        'parallel_observed': (
                            self.result.parallel_decay[index]
                            + self.result.parallel_background),
                        'perpendicular_observed': (
                            self.result.perpendicular_decay[index]
                            + self.result.perpendicular_background),
                        'parallel_model': fit.parallel_model[index],
                        'perpendicular_model': fit.perpendicular_model[index],
                        'parallel_residual': fit.parallel_residual[index],
                        'perpendicular_residual': fit.perpendicular_residual[index],
                    }
                writer.writerow({
                    'time_ns': time_ns,
                    'time_after_peak_ns': time_ns - peak_time,
                    'parallel': parallel,
                    'perpendicular': perpendicular,
                    'anisotropy': anisotropy,
                    'valid': np.isfinite(anisotropy),
                    **fit_values,
                    **provenance,
                })
        self.status.set(f'Saved {Path(path).name}')


def load_irf_curve(path, n_bins, tcspc_res):
    from flimkit.FLIM.irf_tools import irf_from_xlsx
    from flimkit.utils.xlsx_tools import load_irf_export

    exported = load_irf_export(path, debug=False)
    return irf_from_xlsx(exported, n_bins, tcspc_res)


def run_analysis(settings):
    from flimkit.FLIM.anisotropy import (
        analyze_anisotropy, estimate_translation, fit_polarized_decays)
    from flimkit.formats.PTU.reader import PTUFile

    with PTUFile(settings['parallel_path'], verbose=False) as parallel_file:
        parallel = parallel_file.pixel_stack(
            channel=settings['parallel_channel'])
        time_ns = np.asarray(parallel_file.time_ns, dtype=float)
        parallel_period_ns = getattr(parallel_file, 'period_ns', None)
        tcspc_res = getattr(parallel_file, 'tcspc_res', None)
    with PTUFile(settings['perpendicular_path'], verbose=False) as perpendicular_file:
        perpendicular = perpendicular_file.pixel_stack(
            channel=settings['perpendicular_channel'])
        perpendicular_time = np.asarray(perpendicular_file.time_ns, dtype=float)
        perpendicular_period_ns = getattr(perpendicular_file, 'period_ns', None)
    if parallel.shape != perpendicular.shape:
        raise ValueError('Polarization PTUs must have matching image and time shapes')
    if not np.allclose(time_ns, perpendicular_time):
        raise ValueError('Polarization PTUs must have matching time axes')
    analysis_mode = settings.get('analysis_mode', 'direct')
    global_fit_bins = None
    repetition_period_ns = None
    if analysis_mode == 'global':
        if (parallel_period_ns is None
                or not np.isfinite(parallel_period_ns)
                or parallel_period_ns <= 0
                or perpendicular_period_ns is None
                or not np.isfinite(perpendicular_period_ns)
                or perpendicular_period_ns <= 0):
            raise ValueError('Preferred global fitting requires a laser period')
        if not np.isclose(parallel_period_ns, perpendicular_period_ns):
            raise ValueError('Polarization PTUs must have matching laser periods')
        if tcspc_res is None or not np.isfinite(tcspc_res) or tcspc_res <= 0:
            raise ValueError('Preferred global fitting requires TCSPC resolution')
        repetition_period_ns = float(parallel_period_ns)
        time_step_ns = float(np.median(np.diff(time_ns)))
        global_fit_bins = int(round(repetition_period_ns / time_step_ns))
        if global_fit_bins < 2 or global_fit_bins > time_ns.size:
            raise ValueError(
                'Laser period is incompatible with the stored TCSPC time axis')

    combined_decay = (
        parallel.sum(axis=(0, 1)) / settings['parallel_exposure']
        + 2.0 * settings['g_factor'] * perpendicular.sum(axis=(0, 1))
        / settings['perpendicular_exposure'])
    peak_bin = int(np.argmax(combined_decay))
    time_relative = time_ns - time_ns[peak_bin]
    selected = ((time_relative >= settings['analysis_start_ns'])
                & (time_relative < settings['analysis_stop_ns']))
    selected_bins = np.flatnonzero(selected)
    if selected_bins.size == 0:
        raise ValueError('Post-peak range contains no TCSPC bins')
    analysis_bins = slice(int(selected_bins[0]), int(selected_bins[-1]) + 1)

    shift_yx = (0.0, 0.0)
    if settings['auto_register']:
        shift_yx = estimate_translation(
            parallel.sum(axis=-1), perpendicular.sum(axis=-1))
    result = analyze_anisotropy(
        parallel, perpendicular, time_ns,
        background_bins=settings['background_bins'],
        analysis_bins=analysis_bins, g_factor=settings['g_factor'],
        spatial_window=settings['spatial_window'], stride=settings['stride'],
        min_bin_photons=settings['min_bin_photons'],
        min_map_photons=settings['min_map_photons'],
        perpendicular_shift=shift_yx,
        parallel_exposure=settings['parallel_exposure'],
        perpendicular_exposure=settings['perpendicular_exposure'])
    if analysis_mode == 'global':
        assert global_fit_bins is not None
        assert repetition_period_ns is not None
        fit_time_ns = time_ns[:global_fit_bins]
        parallel_irf = load_irf_curve(
            settings['parallel_irf_path'], global_fit_bins, tcspc_res)
        perpendicular_irf = load_irf_curve(
            settings['perpendicular_irf_path'], global_fit_bins, tcspc_res)
        result.polarized_fit = fit_polarized_decays(
            parallel.sum(axis=(0, 1))[:global_fit_bins],
            perpendicular.sum(axis=(0, 1))[:global_fit_bins],
            fit_time_ns, parallel_irf=parallel_irf,
            perpendicular_irf=perpendicular_irf,
            intensity_lifetime_ns=settings['fixed_lifetime_ns'],
            g_factor=settings['g_factor'],
            parallel_exposure=settings['parallel_exposure'],
            perpendicular_exposure=settings['perpendicular_exposure'],
            initial_parallel_background=result.parallel_background,
            initial_perpendicular_background=result.perpendicular_background,
            repetition_period_ns=repetition_period_ns)
    result.metadata.update({
        'parallel_file': Path(settings['parallel_path']).name,
        'perpendicular_file': Path(settings['perpendicular_path']).name,
        'parallel_role': 'parallel',
        'perpendicular_role': 'perpendicular',
        'parallel_channel': settings['parallel_channel'],
        'perpendicular_channel': settings['perpendicular_channel'],
        'background_start_bin': settings['background_bins'].start,
        'background_stop_bin': settings['background_bins'].stop,
        'analysis_start_ns': settings['analysis_start_ns'],
        'analysis_stop_ns': settings['analysis_stop_ns'],
        'min_bin_photons': settings['min_bin_photons'],
        'min_map_photons': settings['min_map_photons'],
        'auto_registration': settings['auto_register'],
        'analysis_mode': analysis_mode,
    })
    if analysis_mode == 'global':
        result.metadata.update({
            'parallel_irf_file': Path(settings['parallel_irf_path']).name,
            'perpendicular_irf_file': Path(
                settings['perpendicular_irf_path']).name,
            'repetition_period_ns': repetition_period_ns,
            'global_fit_bins': global_fit_bins,
            'fixed_lifetime_ns': settings['fixed_lifetime_ns'],
            'global_fit_model': (
                'fixed single fluorescence lifetime; '
                'single rotational correlation'),
            'global_fit_reference': 'Lakowicz, Chapter 11, Section 11.2.2',
        })
    return result, peak_bin


def show_anisotropy_tool(parent):
    return AnisotropyTool(parent)
