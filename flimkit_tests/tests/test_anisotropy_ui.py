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
