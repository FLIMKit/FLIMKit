from queue import Queue
from unittest.mock import patch

import pytest

from flimkit.UI.gui import _UIBuilder


def _tk_root_or_skip():
    import tkinter as tk

    root = None
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display is not available')
    assert root is not None
    root.withdraw()
    return root


def test_menu_handler_opens_anisotropy_tool():
    builder = _UIBuilder.__new__(_UIBuilder)
    builder.root = object()

    with patch('flimkit.UI.anisotropy_tool.show_anisotropy_tool') as show:
        builder._menu_anisotropy()

    show.assert_called_once_with(builder.root)


def test_anisotropy_dialog_exposes_explicit_file_roles():
    import tkinter as tk
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        labels = []
        pending = list(dialog.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            try:
                labels.append(widget.cget('text'))
            except tk.TclError:
                pass
        assert 'Parallel PTU' in labels
        assert 'Perpendicular PTU' in labels
        assert 'Parallel exposure (relative)' in labels
        assert 'Perpendicular exposure (relative)' in labels
        assert 'Parallel photon channel' in labels
        assert 'Perpendicular photon channel' in labels
        assert 'Calculate' in labels
        assert any('G=1 is an assumption' in label for label in labels)
    finally:
        root.destroy()


def test_anisotropy_irf_browser_lists_supported_exports():
    from flimkit.UI.anisotropy_tool import AnisotropyTool

    tool = AnisotropyTool.__new__(AnisotropyTool)
    variable = type('Variable', (), {'set': lambda self, value: setattr(self, 'value', value)})()
    with patch(
            'flimkit.UI.anisotropy_tool.filedialog.askopenfilename',
            return_value='/tmp/parallel.csv') as browse:
        tool._browse(variable, file_kind='irf')

    options = browse.call_args.kwargs
    assert options['title'] == 'Select IRF export'
    patterns = options['filetypes'][0][1]
    for extension in ('*.xlsx', '*.csv', '*.tsv', '*.txt', '*.dat', '*.ascii', '*.asc'):
        assert extension in patterns
    assert options['filetypes'][-1] == ('All files', '*.*')
    assert variable.value == '/tmp/parallel.csv'


def test_anisotropy_dialog_offers_direct_and_preferred_modes():
    import tkinter as tk
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        labels = []
        pending = list(dialog.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            try:
                labels.append(widget.cget('text'))
            except tk.TclError:
                pass
        assert 'Direct r(t) diagnostic (no IRF)' in labels
        assert 'Preferred global fit (Lakowicz Section 11.2.2)' in labels
        assert 'Parallel IRF export' in labels
        assert 'Perpendicular IRF export' in labels
        assert 'Method info...' in labels
    finally:
        root.destroy()


def test_method_info_states_global_fit_requirements():
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        with patch('flimkit.UI.anisotropy_tool.messagebox.showinfo') as showinfo:
            dialog._show_method_info()
        message = showinfo.call_args.args[1]
        assert 'known fluorescence lifetime' in message
        assert 'separate IRFs' in message
        assert 'separate fitted backgrounds' in message
        assert 'resolved time-zero anisotropy' in message
    finally:
        root.destroy()


def test_preferred_mode_requires_two_irf_files(tmp_path):
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.analysis_mode.set('global')

        with pytest.raises(ValueError, match='IRF'):
            dialog._settings()
    finally:
        root.destroy()


def test_preferred_mode_requires_positive_known_lifetime(tmp_path):
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    paths = [tmp_path / name for name in (
        'parallel.ptu', 'perpendicular.ptu', 'parallel.csv', 'perpendicular.csv')]
    for path in paths:
        path.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(paths[0])
        dialog.perpendicular_path.set(paths[1])
        dialog.parallel_irf_path.set(paths[2])
        dialog.perpendicular_irf_path.set(paths[3])
        dialog.analysis_mode.set('global')
        dialog.fixed_lifetime_ns.set(0.0)

        with pytest.raises(ValueError, match='lifetime'):
            dialog._settings()
    finally:
        root.destroy()


def test_anisotropy_dialog_rejects_nonfinite_exposure(tmp_path):
    import tkinter as tk
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.parallel_exposure.set(float('nan'))

        with pytest.raises(ValueError, match='Exposure values'):
            dialog._settings()
    finally:
        root.destroy()


def test_anisotropy_dialog_rejects_nonfinite_photon_threshold(tmp_path):
    import tkinter as tk
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.min_map_photons.set(float('nan'))

        with pytest.raises(ValueError, match='Photon thresholds'):
            dialog._settings()
    finally:
        root.destroy()


def test_anisotropy_dialog_rejects_nonfinite_analysis_time(tmp_path):
    import tkinter as tk
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.analysis_stop_ns.set(float('nan'))

        with pytest.raises(ValueError, match='Post-peak times'):
            dialog._settings()
    finally:
        root.destroy()


def test_anisotropy_dialog_rejects_negative_photon_channel(tmp_path):
    import tkinter as tk
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    parallel = tmp_path / 'parallel.ptu'
    perpendicular = tmp_path / 'perpendicular.ptu'
    parallel.touch()
    perpendicular.touch()
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.parallel_path.set(str(parallel))
        dialog.perpendicular_path.set(str(perpendicular))
        dialog.parallel_channel.set(-1)

        with pytest.raises(ValueError, match='Photon channels'):
            dialog._settings()
    finally:
        root.destroy()


def test_worker_returns_result_through_queue_without_touching_tk():
    from flimkit.UI.anisotropy_tool import AnisotropyTool

    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool._result_queue = Queue()
    expected = (object(), 7)

    with patch('flimkit.UI.anisotropy_tool.run_analysis', return_value=expected):
        tool._analysis_worker({})

    assert tool._result_queue.get_nowait() == ('success', *expected)


def test_closed_dialog_does_not_reschedule_worker_poll():
    from flimkit.UI.anisotropy_tool import AnisotropyTool

    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool._closed = True
    tool._result_queue = Queue()
    scheduled = []
    tool.after = lambda *args: scheduled.append(args)

    tool._poll_worker()

    assert scheduled == []


def test_redrawing_result_replaces_existing_colorbar():

    from types import SimpleNamespace
    import numpy as np
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.result = SimpleNamespace(
            parallel_intensity=np.ones((3, 3)),
            perpendicular_intensity=np.ones((3, 3)),
            time_ns=np.arange(4, dtype=float),
            anisotropy_decay=np.zeros(4),
            anisotropy_map=np.array([
                [-0.35, -0.25, -0.20],
                [-0.10, 0.00, 0.10],
                [-0.30, -0.20, 0.05],
            ]),
            spatial_window=1,
            stride=1,
            metadata={'analysis_start_ns': 0.0, 'analysis_stop_ns': 2.0},
        )
        dialog.peak_bin = 1

        dialog._draw_result()
        axes_after_first_draw = len(dialog.figure.axes)
        color_limits = dialog.axes[1, 1].images[0].get_clim()
        plotted_time = dialog.axes[1, 0].lines[0].get_xdata()
        dialog._draw_result()

        np.testing.assert_array_equal(plotted_time, [0.0, 1.0])
        assert color_limits[0] <= -0.34
        assert color_limits[1] >= 0.4
        assert dialog._colorbar.ax.yaxis.get_ticks_position() == 'left'
        assert axes_after_first_draw == 5
        assert len(dialog.figure.axes) == axes_after_first_draw
    finally:
        root.destroy()


def test_global_fit_mode_draws_polarized_models_and_residuals():
    from types import SimpleNamespace
    import matplotlib as mpl
    import numpy as np
    import warnings
    from flimkit.UI.anisotropy_tool import show_anisotropy_tool

    rc_overrides = {
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'axes.titlecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'font.size': 18.0,
        'axes.titlesize': 18.0,
        'axes.labelsize': 16.0,
        'xtick.labelsize': 14.0,
        'ytick.labelsize': 14.0,
    }
    original_colors = {key: mpl.rcParams[key] for key in rc_overrides}
    mpl.rcParams.update(rc_overrides)
    root = _tk_root_or_skip()
    try:
        dialog = show_anisotropy_tool(root)
        dialog.geometry('1180x820')
        dialog.update_idletasks()
        fit = SimpleNamespace(
            parallel_model=np.array([11.0, 8.0, 5.0]),
            perpendicular_model=np.array([7.0, 6.0, 4.0]),
            parallel_residual=np.array([0.0, 1.0, -1.0]),
            perpendicular_residual=np.array([1.0, 0.0, -1.0]),
            intensity_lifetime_ns=3.2,
            rotational_correlation_ns=1.4,
            initial_anisotropy=0.32,
            poisson_deviance=5.0,
            success=False,
            message='maximum evaluations reached',
            parameters_at_bounds=('initial_anisotropy',),
            common_irf_shift_bins=0.7,
            parallel_background=2.5,
            perpendicular_background=7.0,
        )
        dialog.result = SimpleNamespace(
            parallel_decay=np.array([9.0, 7.0, 3.0, 1.0]),
            perpendicular_decay=np.array([5.0, 4.0, 2.0, 1.0]),
            parallel_background=2.0,
            perpendicular_background=2.0,
            polarized_fit=fit,
            time_ns=np.arange(4, dtype=float),
            metadata={},
        )
        dialog.peak_bin = 1

        dialog._draw_result()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'error', message='constrained_layout not applied.*')
            dialog.canvas.draw()

        plot_axes = (dialog.axes[0, 0], dialog.axes[0, 1], dialog.axes[1, 0])
        assert all(axis.get_position().height >= 0.18 for axis in plot_axes)
        renderer = dialog.canvas.get_renderer()
        assert not dialog.axes[0, 0].xaxis.label.get_window_extent(renderer).overlaps(
            dialog.axes[1, 0].title.get_window_extent(renderer))
        assert dialog.axes[1, 1].texts[0].get_window_extent(renderer).y0 >= 0
        assert dialog.axes[0, 0].get_title() == 'Parallel global fit'
        assert dialog.axes[0, 1].get_title() == 'Perpendicular global fit'
        assert dialog.axes[1, 0].get_title() == 'Residuals'
        summary = dialog.axes[1, 1].texts[0].get_text()
        assert 'Rotational correlation' in summary
        assert 'Fixed fluorescence lifetime' in summary
        assert 'Common IRF shift: 0.7 bins' in summary
        assert 'Fitted backgrounds: 2.5, 7' in summary
        assert 'WARNING' in summary
        assert 'did not converge' in summary
        assert 'maximum evaluations reached' in summary
        assert 'initial_anisotropy' in summary
        assert len(dialog.axes[0, 0].lines) == 2
        assert len(dialog.axes[0, 1].lines) == 2
        assert len(dialog.axes[0, 0].lines[0].get_xdata()) == 3
        assert len(dialog.axes[1, 0].lines[0].get_xdata()) == 3
        assert dialog.axes[0, 0].title.get_color() == '#222222'
        assert dialog.axes[0, 0].xaxis.label.get_color() == '#222222'
        assert dialog.axes[0, 0].get_legend().get_texts()[0].get_color() == '#222222'
        assert dialog.axes[1, 1].texts[0].get_color() == '#222222'
    finally:
        mpl.rcParams.update(original_colors)
        root.destroy()


def test_run_analysis_preferred_mode_fits_both_decays_with_separate_irfs():
    from types import SimpleNamespace
    import numpy as np
    from flimkit.UI.anisotropy_tool import run_analysis

    stack = np.ones((2, 2, 16), dtype=float)

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.time_ns = np.arange(16, dtype=float) * 0.1
            self.tcspc_res = 0.1e-9
            self.period_ns = 1.51

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stack

    preferred_fit = object()
    expected = SimpleNamespace(
        metadata={}, parallel_background=2.0,
        perpendicular_background=3.0, polarized_fit=None)
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'parallel_irf_path': 'parallel.csv',
        'perpendicular_irf_path': 'perpendicular.csv',
        'analysis_mode': 'global',
        'fixed_lifetime_ns': 3.0,
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 1.0,
        'auto_register': False,
        'background_bins': slice(0, 2),
        'g_factor': 1.2,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 0.0,
        'min_map_photons': 0.0,
        'parallel_exposure': 1.5,
        'perpendicular_exposure': 0.8,
    }
    parallel_irf = np.eye(1, 15, 2).ravel()
    perpendicular_irf = np.eye(1, 15, 4).ravel()

    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit.FLIM.anisotropy.analyze_anisotropy',
                return_value=expected),
          patch('flimkit.UI.anisotropy_tool.load_irf_curve',
                side_effect=[parallel_irf, perpendicular_irf]),
          patch('flimkit.FLIM.anisotropy.fit_polarized_decays',
                return_value=preferred_fit) as fit):
        result, _ = run_analysis(settings)

    assert result.polarized_fit is preferred_fit
    call = fit.call_args
    np.testing.assert_array_equal(call.kwargs['parallel_irf'], parallel_irf)
    np.testing.assert_array_equal(
        call.kwargs['perpendicular_irf'], perpendicular_irf)
    assert call.kwargs['repetition_period_ns'] == 1.51
    assert call.args[0].shape == (15,)
    assert call.args[1].shape == (15,)
    assert call.args[2].shape == (15,)
    assert call.kwargs['intensity_lifetime_ns'] == 3.0
    assert call.kwargs['initial_parallel_background'] == 2.0
    assert call.kwargs['initial_perpendicular_background'] == 3.0
    assert call.kwargs['g_factor'] == 1.2
    assert result.metadata['analysis_mode'] == 'global'
    assert result.metadata['parallel_irf_file'] == 'parallel.csv'
    assert result.metadata['perpendicular_irf_file'] == 'perpendicular.csv'
    assert result.metadata['repetition_period_ns'] == 1.51
    assert result.metadata['global_fit_bins'] == 15
    assert result.metadata['fixed_lifetime_ns'] == 3.0


def test_run_analysis_records_photon_thresholds():
    from types import SimpleNamespace
    import numpy as np
    from flimkit.UI.anisotropy_tool import run_analysis

    stack = np.ones((2, 2, 4), dtype=float)

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.time_ns = np.arange(4, dtype=float)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stack

    expected = SimpleNamespace(metadata={})
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 2.0,
        'auto_register': False,
        'background_bins': slice(0, 1),
        'g_factor': 1.0,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 25.0,
        'min_map_photons': 500.0,
        'parallel_exposure': 1.0,
        'perpendicular_exposure': 1.0,
    }

    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit.FLIM.anisotropy.analyze_anisotropy',
                return_value=expected)):
        result, _ = run_analysis(settings)

    assert result.metadata['min_bin_photons'] == 25.0
    assert result.metadata['min_map_photons'] == 500.0


def test_run_analysis_uses_g_and_exposure_normalized_peak():
    from types import SimpleNamespace
    import numpy as np
    from flimkit.UI.anisotropy_tool import run_analysis

    stacks = {
        'parallel.ptu': np.array([[[100.0, 0.0]]]),
        'perpendicular.ptu': np.array([[[0.0, 80.0]]]),
    }

    class FakePTUFile:
        def __init__(self, path, verbose=False):
            self.path = path
            self.time_ns = np.arange(2, dtype=float)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def pixel_stack(self, channel):
            return stacks[self.path]

    expected = SimpleNamespace(metadata={})
    settings = {
        'parallel_path': 'parallel.ptu',
        'perpendicular_path': 'perpendicular.ptu',
        'parallel_channel': 0,
        'perpendicular_channel': 0,
        'analysis_start_ns': 0.0,
        'analysis_stop_ns': 0.5,
        'auto_register': False,
        'background_bins': slice(0, 1),
        'g_factor': 1.0,
        'spatial_window': 1,
        'stride': 1,
        'min_bin_photons': 0.0,
        'min_map_photons': 0.0,
        'parallel_exposure': 1.0,
        'perpendicular_exposure': 100.0,
    }

    with (patch('flimkit.formats.PTU.reader.PTUFile', FakePTUFile),
          patch('flimkit.FLIM.anisotropy.analyze_anisotropy',
                return_value=expected)):
        _, peak_bin = run_analysis(settings)

    assert peak_bin == 0


def test_global_fit_csv_includes_models_residuals_and_parameters(tmp_path):
    import csv
    from types import SimpleNamespace
    import numpy as np
    from flimkit.UI.anisotropy_tool import AnisotropyTool

    fit = SimpleNamespace(
        parallel_model=np.array([9.0]),
        perpendicular_model=np.array([4.0]),
        parallel_residual=np.array([1.0]),
        perpendicular_residual=np.array([0.0]),
        intensity_lifetime_ns=3.2,
        rotational_correlation_ns=1.4,
        initial_anisotropy=0.32,
        poisson_deviance=5.0,
        common_irf_shift_bins=0.7,
        parallel_background=2.5,
        perpendicular_background=7.0,
    )
    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool.peak_bin = 0
    tool.status = SimpleNamespace(set=lambda value: None)
    tool.result = SimpleNamespace(
        time_ns=np.array([1.0, 2.0]),
        parallel_decay=np.array([8.0, 3.0]),
        perpendicular_decay=np.array([3.0, 1.0]),
        parallel_background=2.0,
        perpendicular_background=3.0,
        anisotropy_decay=np.array([0.2, 0.2]),
        polarized_fit=fit,
        g_factor=1.0,
        parallel_exposure=1.0,
        perpendicular_exposure=1.0,
        perpendicular_shift=(0.0, 0.0),
        spatial_window=1,
        stride=1,
        metadata={'analysis_mode': 'global'},
    )
    path = tmp_path / 'global.csv'

    with patch('flimkit.UI.anisotropy_tool.filedialog.asksaveasfilename',
               return_value=str(path)):
        tool._save_csv()

    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['parallel_model'] == '9.0'
    assert rows[0]['parallel_observed'] == '10.0'
    assert rows[0]['perpendicular_observed'] == '6.0'
    assert rows[0]['perpendicular_residual'] == '0.0'
    assert rows[0]['intensity_lifetime_ns'] == '3.2'
    assert rows[0]['rotational_correlation_ns'] == '1.4'
    assert rows[0]['initial_anisotropy'] == '0.32'
    assert rows[0]['common_irf_shift_bins'] == '0.7'
    assert rows[0]['parallel_fit_background'] == '2.5'
    assert rows[0]['perpendicular_fit_background'] == '7.0'
    assert rows[1]['parallel_model'] == ''
    assert rows[1]['parallel_observed'] == ''


def test_csv_export_includes_relative_time_and_provenance(tmp_path):
    import csv
    from types import SimpleNamespace
    import numpy as np
    from flimkit.UI.anisotropy_tool import AnisotropyTool

    tool = AnisotropyTool.__new__(AnisotropyTool)
    tool.peak_bin = 1
    tool.status = SimpleNamespace(set=lambda value: None)
    tool.result = SimpleNamespace(
        time_ns=np.array([1.0, 2.0]),
        parallel_decay=np.array([10.0, 5.0]),
        perpendicular_decay=np.array([4.0, 2.0]),
        anisotropy_decay=np.array([0.2, 0.2]),
        g_factor=0.9,
        parallel_exposure=1.0,
        perpendicular_exposure=2.0,
        perpendicular_shift=(0.25, -0.5),
        spatial_window=5,
        stride=1,
        metadata={
            'parallel_file': 'parallel.ptu',
            'perpendicular_file': 'perpendicular.ptu',
            'parallel_channel': 0,
            'perpendicular_channel': 1,
            'background_start_bin': 2,
            'background_stop_bin': 10,
            'analysis_start_ns': 0.0,
            'analysis_stop_ns': 8.0,
            'min_bin_photons': 25.0,
            'min_map_photons': 500.0,
        },
    )
    path = tmp_path / 'decay.csv'

    with patch('flimkit.UI.anisotropy_tool.filedialog.asksaveasfilename',
               return_value=str(path)):
        tool._save_csv()

    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['time_after_peak_ns'] == '-1.0'
    assert rows[0]['g_factor'] == '0.9'
    assert rows[0]['parallel_file'] == 'parallel.ptu'
    assert rows[0]['perpendicular_channel'] == '1'
    assert rows[0]['min_map_photons'] == '500.0'
