import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flimkit.FLIM.assemble import Cancelled, assemble_tile_maps

PROBE = '''
import flimkit.plugins as plugins
from flimkit.formats import supported_extensions

assert not plugins.load_report(), 'plugins must not load before anything asks'
supported_extensions()
loaded = [entry.source for entry in plugins.load_report()]
if not loaded:
    raise SystemExit('a format lookup did not load plugins')
'''


def test_a_format_lookup_loads_plugins_without_a_gui():
    from tests.test_headless_imports import probe_env

    proc = subprocess.run(
        [sys.executable, '-c', PROBE], capture_output=True, text=True,
        env=probe_env())
    assert proc.returncode == 0, proc.stdout + proc.stderr


def tile(index, y0, x0, size=8):
    maps = {
        'tau_mean_amp': np.full((size, size), 2.0, dtype=np.float32),
        'tau_mean_int': np.full((size, size), 2.0, dtype=np.float32),
        'intensity': np.full((size, size), 100.0, dtype=np.float32),
        'tau1': np.full((size, size), 2.0, dtype=np.float32),
        'a1': np.ones((size, size), dtype=np.float32),
    }
    return {'pixel_maps': maps, 'pixel_y': y0, 'pixel_x': x0,
            'tile_h': size, 'tile_w': size}


def test_assembly_runs_when_nothing_cancels_it():
    canvas = assemble_tile_maps([tile(0, 0, 0), tile(1, 0, 8)], 8, 16, n_exp=1)
    assert canvas['tau_mean_amp'].shape == (8, 16)


def test_assembly_stops_when_the_job_is_cancelled():
    import threading

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        assemble_tile_maps([tile(0, 0, 0), tile(1, 0, 8)], 8, 16, n_exp=1,
                           cancel_event=cancel)
