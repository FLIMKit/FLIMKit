import numpy as np
import pytest

from flimkit.utils import display


def ends(cmap):
    return np.array(cmap(0.0)[:3]), np.array(cmap(1.0)[:3])


@pytest.mark.parametrize('name', list(display.COLORMAPS))
def test_no_flim_colormap_comes_back_to_its_start(name):
    start, end = ends(display.get_colormap(name))
    assert np.linalg.norm(start - end) > 0.3


@pytest.mark.parametrize('name', display.INTENSITY_COLORMAPS)
def test_intensity_colormaps_load(name):
    start, end = ends(display.get_colormap(name))
    assert np.linalg.norm(start - end) > 0.3


def test_values_outside_the_range_stay_at_the_ends():
    cmap = display.get_colormap('hsv')
    assert cmap(-0.5) == cmap(0.0)
    assert cmap(1.5) == cmap(1.0)


def test_each_call_returns_its_own_copy():
    first = display.get_colormap('viridis')
    first.set_bad('red')
    second = display.get_colormap('viridis')
    assert tuple(second.get_bad()) != tuple(first.get_bad())


def test_sessions_saved_with_the_old_names_still_load():
    for name in ('hsv', 'viridis', 'cool', 'hot', 'twilight'):
        display.get_colormap(name)
