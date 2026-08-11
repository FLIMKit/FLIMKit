import numpy as np
import pytest

from flimkit.FLIM.assemble import assemble_tile_maps, derive_global_tau, save_assembled_maps
from flimkit.formats.PTU.stitch import _adapt_pixel_maps

TAUS_NS = [6.1, 1.9, 0.35]


def raw_maps(ny=6, nx=6, n_exp=3):
    maps = {
        'intensity': np.full((ny, nx), 500.0, np.float32),
        'tau_mean_amp': np.full((ny, nx), 1.8, np.float32),
        'tau_mean_int': np.full((ny, nx), 3.9, np.float32),
        'chi2_r': np.ones((ny, nx), np.float32),
        'calibrated_chi2_r': np.ones((ny, nx), np.float32),
    }
    for k in range(1, n_exp + 1):
        maps[f'alpha_{k}'] = np.full((ny, nx), 0.3, np.float32)
        maps[f'tau_{k}'] = np.full((ny, nx), TAUS_NS[k - 1], np.float32)
    return maps


def one_tile_canvas(adapted, ny=6, nx=6, n_exp=3):
    tile = {'pixel_maps': adapted, 'pixel_y': 0, 'pixel_x': 0,
            'tile_h': ny, 'tile_w': nx}
    return assemble_tile_maps([tile], ny, nx, n_exp)


def test_adapter_keeps_the_intensity_weighted_map():
    adapted = _adapt_pixel_maps(raw_maps(), 3, TAUS_NS)
    assert 'tau_mean_int' in adapted
    assert np.isfinite(np.asarray(adapted['tau_mean_int'], float)).all()


def test_intensity_weighted_canvas_is_not_all_nan():
    canvas = one_tile_canvas(_adapt_pixel_maps(raw_maps(), 3, TAUS_NS))
    tau_int = np.asarray(canvas['tau_mean_int'], float)
    assert np.isfinite(tau_int).sum() == tau_int.size
    assert np.allclose(tau_int, 3.9)


def test_both_weightings_survive_assembly():
    canvas = one_tile_canvas(_adapt_pixel_maps(raw_maps(), 3, TAUS_NS))
    for key, want in (('tau_mean_amp', 1.8), ('tau_mean_int', 3.9)):
        arr = np.asarray(canvas[key], float)
        assert np.isfinite(arr).all(), key
        assert np.allclose(arr, want), key


def test_intensity_weighted_tif_is_written_with_signal(tmp_path):
    import tifffile
    canvas = one_tile_canvas(_adapt_pixel_maps(raw_maps(), 3, TAUS_NS))
    save_assembled_maps(canvas=canvas, global_summary={}, output_dir=tmp_path,
                        roi_name='roi', n_exp=3, tau_display_min=0.0,
                        tau_display_max=5.0, tau_weighting='int')
    tif = tifffile.imread(str(tmp_path / 'roi_tau_mean_int.tif'))
    assert tif.max() > 0
    saved = np.load(str(tmp_path / 'roi_tau_mean_int.npy'))
    assert np.isfinite(saved).all()


def test_adapter_output_satisfies_the_global_tau_contract():
    canvas = one_tile_canvas(_adapt_pixel_maps(raw_maps(), 3, TAUS_NS))
    summary = derive_global_tau(canvas, n_exp=3)
    assert 'error' not in summary
    assert summary['n_pixels_fitted'] == 36


def test_adapter_tolerates_a_fitter_without_the_map():
    maps = raw_maps()
    del maps['tau_mean_int']
    adapted = _adapt_pixel_maps(maps, 3, TAUS_NS)
    assert np.isnan(np.asarray(adapted['tau_mean_int'], float)).all()
