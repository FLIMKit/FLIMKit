import numpy as np
import pytest
from flimkit.phasor.export import savePhasorPlots
from flimkit.phasor_launcher import save_session, load_session

def _session(tmp_path, cursors, shape=(32, 32), phasor_filter=None):
    rng = np.random.default_rng(0)
    left = np.arange(shape[1])[None, :] < shape[1] // 2
    g = np.where(left, 0.6, 0.35) + 0.02 * rng.standard_normal(shape)
    s = np.where(left, 0.4, 0.33) + 0.02 * rng.standard_normal(shape)
    mean = rng.poisson(50, shape).astype(float)
    path = tmp_path / 'scan_phasor.npz'
    save_session(str(path), real_cal=g, imag_cal=s, mean=mean, frequency=80.0,
                 cursors=cursors, params=dict(radius=0.05, radius_minor=0.03),
                 display_image=mean, phasor_filter=phasor_filter)
    return load_session(str(path))

def _is_png(path):
    return path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'

def test_writes_plot_and_image(tmp_path):
    cursors = [dict(type='ellipse', center_g=0.6, center_s=0.4, color='#d62728'),
               dict(type='poly', vertices=[(0.28, 0.26), (0.42, 0.26), (0.35, 0.42)], color='#1f77b4')]
    written = savePhasorPlots(_session(tmp_path, cursors), tmp_path / 'out', 'scan')
    assert [f.name for f in written] == ['scan_phasor.png', 'scan_phasor_image.png']
    assert all(_is_png(f) for f in written)

def test_no_cursors_still_writes_both(tmp_path):
    written = savePhasorPlots(_session(tmp_path, []), tmp_path, 'scan')
    assert len(written) == 2

def test_saved_filter_is_reapplied(tmp_path):
    sess = _session(tmp_path, [], phasor_filter=dict(method='median', size=3, sigma=1.0))
    assert sess['phasor_filter']['method'] == 'median'
    assert len(savePhasorPlots(sess, tmp_path, 'scan')) == 2

def test_point_measurement_skips_image(tmp_path):
    sess = _session(tmp_path, [], shape=(1, 1))
    written = savePhasorPlots(sess, tmp_path, 'scan')
    assert [f.name for f in written] == ['scan_phasor.png']

def test_no_valid_pixels_raises(tmp_path):
    sess = _session(tmp_path, [])
    with pytest.raises(ValueError):
        savePhasorPlots(sess, tmp_path, 'scan', min_photons=1e9)
