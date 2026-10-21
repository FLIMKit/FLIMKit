import argparse

import numpy as np
import pytest

from flimkit.phasor_launcher import load_session, save_session
from flimkit.FLIM.batch import _binning, _window_kwargs

def _arrays(shape=(8, 8)):
    rng = np.random.default_rng(0)
    return dict(
        real_cal=rng.uniform(0.2, 0.8, shape),
        imag_cal=rng.uniform(0.1, 0.5, shape),
        mean=rng.uniform(0.0, 5.0, shape),
        frequency=19.5,
    )

def test_roundtrip_keeps_polygon_cursors_and_filter(tmp_path):
    cursors = [
        dict(type='ellipse', center_g=0.6, center_s=0.3, color='#d62728'),
        dict(type='poly', vertices=[(0.5, 0.2), (0.7, 0.2), (0.6, 0.4)], color='#1f77b4'),
    ]
    path = tmp_path / 'a_phasor.npz'
    save_session(
        path, cursors=cursors,
        params=dict(radius=0.05, radius_minor=0.03, angle_mode='semicircle'),
        phasor_filter=dict(method='wavelet', sigma=1.0, size=3),
        **_arrays(),
    )
    sess = load_session(path)
    assert [c['type'] for c in sess['cursors']] == ['ellipse', 'poly']
    assert sess['cursors'][0]['center_g'] == pytest.approx(0.6)
    assert sess['cursors'][1]['vertices'] == [(0.5, 0.2), (0.7, 0.2), (0.6, 0.4)]
    assert sess['phasor_filter']['method'] == 'wavelet'

def test_session_without_filter_or_types_still_loads(tmp_path):
    path = tmp_path / 'old_phasor.npz'
    arrays = _arrays()
    np.savez_compressed(
        path,
        real_cal=arrays['real_cal'], imag_cal=arrays['imag_cal'], mean=arrays['mean'],
        frequency=np.float64(19.5),
        cursor_g=np.array([0.6]), cursor_s=np.array([0.3]),
        cursor_colors=np.array(['#d62728'], dtype='U10'),
        param_radius=np.float64(0.05), param_radius_minor=np.float64(0.03),
        param_angle_mode=np.array('semicircle'),
        ptu_file=np.array(''), irf_file=np.array(''),
    )
    sess = load_session(path)
    assert sess['cursors'][0]['type'] == 'ellipse'
    assert sess['phasor_filter']['method'] == 'none'

def test_batch_window_kwargs_parse_expert_settings():
    args = argparse.Namespace(fit_start_ns=0.5, fit_end_ns=18.0,
                              exclude_ns='19.6-21.2', irf_shift_bins=3, binning=8)
    kw = _window_kwargs(args)
    assert kw['exclude_ns'] == [(19.6, 21.2)]
    assert kw['fit_start_ns'] == 0.5
    assert kw['fit_end_ns'] == 18.0
    assert kw['irf_shift_bins'] == 3
    assert _binning(args) == 8

def test_batch_window_kwargs_defaults():
    kw = _window_kwargs(argparse.Namespace())
    assert kw == dict(irf_shift_bins=2, fit_start_ns=None, fit_end_ns=None, exclude_ns=None)
    assert _binning(argparse.Namespace()) == 1
