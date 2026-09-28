import numpy as np
import pytest
from flimkit.utils.export_png import tag_pixel_size_um, lifetime_limits, colorbar_ticks, save_colorbar

def test_pixel_size_reads_pixresol_in_um():
    assert tag_pixel_size_um({'ImgHdr_PixResol': 0.6512607843137255}) == pytest.approx(0.65126, rel=1e-4)

def test_pixel_size_falls_back_to_pixres():
    assert tag_pixel_size_um({'ImgHdr_PixRes': 0.2}) == 0.2

def test_pixel_size_missing_is_none():
    assert tag_pixel_size_um({}) is None
    assert tag_pixel_size_um({'ImgHdr_PixResol': 0}) is None

def test_pixel_size_from_real_ptu_tags():
    from pathlib import Path
    p = Path('/Users/as-hunt/Downloads/20260623 Test.sptw/Series008_z4.ptu')
    if not p.exists():
        pytest.skip('sample PTU not on this machine')
    from flimkit.formats import FLIMFile
    size = tag_pixel_size_um(FLIMFile(str(p), verbose=False).tags)
    assert size == pytest.approx(0.6513, rel=1e-3)

def test_limits_use_user_range():
    img = np.linspace(0, 10, 100).reshape(10, 10)
    assert lifetime_limits(img, 2.0, 4.0) == (2.0, 4.0)

def test_limits_auto_match_display_percentiles():
    img = np.linspace(0, 10, 101).reshape(1, -1)
    lo, hi = lifetime_limits(img)
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(9.8)

def test_limits_ignore_nan():
    img = np.array([[np.nan, 1.0], [2.0, 3.0]])
    lo, hi = lifetime_limits(img, None, None, (0, 100))
    assert (lo, hi) == (1.0, 3.0)

def test_ticks_follow_gamma():
    pos, vals = colorbar_ticks(0.0, 4.0, gamma=2.0, n=3)
    assert list(vals) == [0.0, 2.0, 4.0]
    assert pos[1] == pytest.approx(0.5 ** 0.5)

def test_save_colorbar_writes_png(tmp_path):
    from matplotlib import colormaps
    out = tmp_path / 'cb.png'
    save_colorbar(out, colormaps['viridis'], 1.0, 3.0, 'τ (ns)')
    assert out.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'

def _pixel_um(path):
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        x = page.tags['XResolution'].value
        unit = page.tags['ResolutionUnit'].value
    assert int(unit) == 3
    return 1e4 / (x[0] / x[1])

def test_tiff_resolution_empty_without_area():
    from flimkit.utils.export_png import tiff_resolution
    assert tiff_resolution(np.zeros((4, 4)), None) == {}

def test_field_area_matches_pixel_size():
    from flimkit.utils.export_png import field_area_um2
    assert field_area_um2(0.5, (256, 128)) == pytest.approx(0.25 * 256 * 128)
    assert field_area_um2(None, (4, 4)) is None

def test_weighted_tau_tiffs_carry_pixel_size(tmp_path):
    from flimkit.utils.export_png import field_area_um2
    from flimkit.utils.enhanced_outputs import save_weighted_tau_images, save_individual_tau_maps
    maps = {
        'intensity': np.full((32, 32), 100.0),
        'tau_1': np.full((32, 32), 2.0), 'a1': np.ones((32, 32)),
        'tau_2': np.full((32, 32), 0.5), 'a2': np.ones((32, 32)),
    }
    area = field_area_um2(0.6513, (256, 256))
    save_weighted_tau_images(maps, tmp_path, roi_name='r', n_exp=2, field_area_um2=area)
    save_individual_tau_maps(maps, tmp_path, roi_name='r', n_exp=2, field_area_um2=area)
    for name in ('r_intensity.tif', 'r_tau_intensity_weighted.tif', 'r_tau_amplitude_weighted.tif', 'r_tau1.tif', 'r_a2.tif'):
        assert _pixel_um(tmp_path / name) == pytest.approx(0.6513 * 8, rel=1e-6)

def test_assembled_and_lifetime_tiffs_carry_pixel_size(tmp_path):
    from flimkit.FLIM.assemble import save_assembled_maps
    from flimkit.utils.lifetime_image import make_lifetime_image, make_component_rgb_tiff
    rng = np.random.default_rng(0)
    canvas = {
        'intensity': rng.uniform(10, 100, (40, 60)),
        'tau_mean_amp': rng.uniform(1, 3, (40, 60)),
        'a1': rng.uniform(0, 1, (40, 60)), 'a2': rng.uniform(0, 1, (40, 60)),
    }
    area = 0.2 ** 2 * 40 * 60
    save_assembled_maps(canvas, {}, tmp_path, 'c', 2, field_area_um2=area)
    make_lifetime_image(canvas, tmp_path, 'c', verbose=False, field_area_um2=area)
    make_component_rgb_tiff(canvas, tmp_path, 'c', 2, verbose=False, field_area_um2=area)
    for name in ('c_intensity.tif', 'c_tau_mean_amp.tif', 'c_tau_intensity_weighted.tif',
                 'c_tau_intensity_weighted_fullrange.tif', 'c_component_rgb.tif'):
        assert _pixel_um(tmp_path / name) == pytest.approx(0.2, rel=1e-6)

def test_ptu_field_area_from_tags():
    from flimkit.utils.export_png import flim_field_area_um2
    class F:
        tags = {'ImgHdr_PixResol': 0.5, 'ImgHdr_PixX': 10, 'ImgHdr_PixY': 20}
    assert flim_field_area_um2(F()) == pytest.approx(0.25 * 200)
