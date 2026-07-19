import numpy as np
from types import SimpleNamespace
from flimkit.interactive import apply_point_mode_limits

class _Point:
    is_image = False

class _Image:
    is_image = True

class _NoAttr:
    pass

def _args():
    return SimpleNamespace(mode='both', cell_mask=True, intensity_threshold=50)

def test_point_file_forces_summed():
    a = _args()
    assert apply_point_mode_limits(a, _Point()) is True
    assert a.mode == 'summed'
    assert a.cell_mask is False
    assert a.intensity_threshold is None

def test_image_file_is_untouched():
    a = _args()
    assert apply_point_mode_limits(a, _Image()) is False
    assert a.mode == 'both'
    assert a.cell_mask is True
    assert a.intensity_threshold == 50

def test_reader_without_is_image_treated_as_image():
    a = _args()
    assert apply_point_mode_limits(a, _NoAttr()) is False
    assert a.mode == 'both'

def test_summed_mode_point_file_stays_summed():
    a = SimpleNamespace(mode='summed', cell_mask=False, intensity_threshold=None)
    assert apply_point_mode_limits(a, _Point()) is True
    assert a.mode == 'summed'
