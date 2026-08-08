import numpy as np
import pytest

from flimkit.FLIM.fit_tools import calibrated_chi2


def test_calibrated_chi2_uses_expected_sparse_bin_contributions():
    data = np.array([0.0, 1.0, 3.0, 8.0])
    model = np.array([0.1, 0.5, 3.5, 8.0])
    numerator = np.sum((data - model) ** 2 / np.maximum(model, 1.0))
    expected = np.sum(np.minimum(model, 1.0))

    assert calibrated_chi2(data, model) == pytest.approx(numerator / expected)


def test_calibrated_chi2_supports_per_pixel_arrays():
    data = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    model = np.array([[0.2, 0.8, 2.0], [0.5, 0.1, 3.5], [0.0, 0.0, 0.0]])
    numerator = np.sum((data - model) ** 2 / np.maximum(model, 1.0), axis=1)
    expected = np.sum(np.minimum(model, 1.0), axis=1)
    expected_result = np.array([numerator[0] / expected[0], numerator[1] / expected[1], np.nan])

    np.testing.assert_allclose(
        calibrated_chi2(data, model, axis=1), expected_result, equal_nan=True)


def test_calibrated_chi2_matches_bin_average_for_dense_model():
    data = np.array([2.0, 5.0, 8.0])
    model = np.array([2.5, 4.0, 7.0])
    numerator = np.sum((data - model) ** 2 / model)

    assert calibrated_chi2(data, model) == pytest.approx(numerator / model.size)


def test_calibrated_chi2_is_undefined_for_zero_model():
    assert np.isnan(calibrated_chi2(np.zeros(4), np.zeros(4)))


def test_calibrated_chi2_rejects_invalid_poisson_models():
    data = np.array([1.0, 2.0])

    assert np.isnan(calibrated_chi2(data, np.array([-0.2, 1.0])))
    assert np.isnan(calibrated_chi2(data, np.array([np.nan, 1.0])))
    assert np.isnan(calibrated_chi2(data, np.array([np.inf, 1.0])))
    assert np.isnan(calibrated_chi2(np.array([-1.0, 2.0]), np.ones(2)))


def test_calibrated_chi2_has_unit_mean_for_sparse_poisson_data():
    rng = np.random.default_rng(42)
    model = np.geomspace(0.01, 0.9, 128)
    data = rng.poisson(model, size=(5000, model.size))

    calibrated = calibrated_chi2(data, model, axis=1)
    legacy = np.sum((data - model) ** 2 / np.maximum(model, 1.0), axis=1) / model.size

    assert np.mean(calibrated) == pytest.approx(1.0, abs=0.02)
    assert np.mean(legacy) < 0.3


def test_summed_summary_adds_calibrated_full_and_tail_values():
    from flimkit.FLIM.fitters import _make_summary
    from flimkit.FLIM.models import reconvolution_model

    n_bins = 128
    tcspc_res = 0.1e-9
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    popt = np.array([2e-9, 20.0, 0.0])
    model = reconvolution_model(
        popt, tcspc_res, n_bins, irf, 1, 0.0, False, False, False)
    data = np.random.default_rng(7).poisson(model)
    fit_idx = np.arange(n_bins)

    summary = _make_summary(
        popt, data, tcspc_res, n_bins, irf, 1, 0.0,
        False, False, False, fit_idx)
    tail_idx = fit_idx[fit_idx >= summary['tail_start_bin']]
    legacy_numerator = np.sum((data - model) ** 2 / np.maximum(model, 1.0))

    assert summary['chi2_pearson'] == pytest.approx(legacy_numerator)
    assert summary['reduced_chi2_pearson'] == pytest.approx(
        legacy_numerator / (n_bins - len(popt)))
    assert summary['calibrated_chi2_pearson'] == pytest.approx(
        calibrated_chi2(data[fit_idx], model[fit_idx]))
    assert summary['calibrated_chi2_tail_pearson'] == pytest.approx(
        calibrated_chi2(data[tail_idx], model[tail_idx]))


def test_tail_summary_adds_calibrated_value():
    from flimkit.FLIM.fitters import _make_summary_tail
    from flimkit.FLIM.models import tail_model

    n_bins = 128
    tcspc_res = 0.1e-9
    popt = np.array([2e-9, 20.0])
    model = tail_model(popt, tcspc_res, n_bins, 1, 0.0, False)
    data = np.random.default_rng(8).poisson(model)
    fit_idx = np.arange(n_bins)

    summary = _make_summary_tail(
        popt, data, tcspc_res, n_bins, 1, 0.0, False, fit_idx)
    expected = calibrated_chi2(data, model)
    legacy_numerator = np.sum((data - model) ** 2 / np.maximum(model, 1.0))

    assert summary['chi2_pearson'] == pytest.approx(legacy_numerator)
    assert summary['reduced_chi2_pearson'] == pytest.approx(
        legacy_numerator / (n_bins - len(popt)))
    assert summary['calibrated_chi2_pearson'] == pytest.approx(expected)
    assert summary['calibrated_chi2_tail_pearson'] == pytest.approx(expected)


def test_distribution_summary_adds_calibrated_full_and_tail_values():
    from flimkit.FLIM.fitters import _make_summary_dist
    from flimkit.FLIM.models import dist_reconvolution_model

    n_bins = 128
    tcspc_res = 0.1e-9
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    popt = np.array([2e-9, 0.3e-9, 20.0, 0.0])
    model = dist_reconvolution_model(
        popt, tcspc_res, n_bins, irf, 1, 'gaussian',
        0.0, False, False)
    data = np.random.default_rng(9).poisson(model)
    fit_idx = np.arange(n_bins)

    summary = _make_summary_dist(
        popt, data, tcspc_res, n_bins, irf, 1, 'gaussian',
        0.0, False, False, fit_idx)
    tail_idx = fit_idx[fit_idx >= summary['tail_start_bin']]
    legacy_numerator = np.sum((data - model) ** 2 / np.maximum(model, 1.0))

    assert summary['chi2_pearson'] == pytest.approx(legacy_numerator)
    assert summary['reduced_chi2_pearson'] == pytest.approx(
        legacy_numerator / (n_bins - len(popt)))
    assert summary['calibrated_chi2_pearson'] == pytest.approx(
        calibrated_chi2(data[fit_idx], model[fit_idx]))
    assert summary['calibrated_chi2_tail_pearson'] == pytest.approx(
        calibrated_chi2(data[tail_idx], model[tail_idx]))


def test_per_pixel_cpu_adds_calibrated_map():
    from flimkit.FLIM.fit_tools import estimate_bg
    from flimkit.FLIM.fitters import _basis_rows, fit_per_pixel

    n_bins = 128
    tcspc_res = 0.1e-9
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    model = 20.0 * np.exp(-np.arange(n_bins) * tcspc_res / 2e-9)
    decay = np.random.default_rng(10).poisson(model).astype(float)
    stack = decay[None, None, :]
    global_popt = np.array([2e-9, 20.0, 0.0])

    maps = fit_per_pixel(
        stack, tcspc_res, n_bins, irf, False, False, False,
        global_popt, 1, min_photons=1, use_gpu=False)
    tau = maps['tau_1'][0, 0] * 1e-9
    amp = maps['alpha_1'][0, 0]
    bg = estimate_bg(decay, int(np.argmax(decay)))
    basis = _basis_rows(
        np.array([tau]), np.arange(n_bins) * tcspc_res,
        tcspc_res, n_bins, False, irf_fft=np.fft.fft(irf))[0]
    fitted_model = amp * basis + bg

    assert maps['calibrated_chi2_r'][0, 0] == pytest.approx(
        calibrated_chi2(decay, fitted_model))


def test_available_gpu_backend_matches_cpu_calibrated_map():
    from flimkit.FLIM.fitters import fit_per_pixel
    from flimkit.GPU import get_backend

    backend = get_backend()
    if backend is None:
        pytest.skip('no GPU backend available')
    n_bins = 128
    tcspc_res = 0.1e-9
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    model = 20.0 * np.exp(-np.arange(n_bins) * tcspc_res / 2e-9)
    decay = np.random.default_rng(14).poisson(model).astype(float)
    stack = decay[None, None, :]
    global_popt = np.array([2e-9, 20.0, 0.0])

    cpu = fit_per_pixel(
        stack, tcspc_res, n_bins, irf, False, False, False,
        global_popt, 1, min_photons=1, use_gpu=False)
    gpu = fit_per_pixel(
        stack, tcspc_res, n_bins, irf, False, False, False,
        global_popt, 1, min_photons=1, use_gpu='auto', gpu_backend=backend)

    assert gpu['calibrated_chi2_r'][0, 0] == pytest.approx(
        cpu['calibrated_chi2_r'][0, 0], rel=1e-5)


def test_backend_scatter_adds_calibrated_map():
    from flimkit.GPU._base import _BackendMixin

    data = np.array([[0.0, 1.0, 3.0, 2.0]])
    basis = np.array([[1.0, 0.5, 0.25, 0.125]])
    amp = np.array([2.0])
    bg = np.array([0.1])
    model = amp[:, None] * basis + bg[:, None]
    maps = _BackendMixin._init_maps(
        1, 1, 1, np.array([[data.sum()]]), np.array([2.0]), True)

    _BackendMixin._scatter_1exp(
        maps, np.array([0]), np.array([2e-9]), amp, bg,
        data, basis, 1, 1, data.shape[1])

    assert maps['calibrated_chi2_r'][0, 0] == pytest.approx(
        calibrated_chi2(data[0], model[0]))


def test_backend_free_tau_scatter_keeps_calibrated_values():
    from flimkit.GPU._base import _BackendMixin

    maps = _BackendMixin._init_maps(
        1, 1, 1, np.array([[10.0]]), np.array([2.0]), True)
    _BackendMixin._scatter_free_tau(
        maps, np.array([0]), np.array([[2e-9]]), np.array([[3.0]]),
        np.array([0.5]), np.array([1.25]), 1, 1, 1)

    assert maps['chi2_r'][0, 0] == pytest.approx(0.5)
    assert maps['calibrated_chi2_r'][0, 0] == pytest.approx(1.25)


def test_per_pixel_distribution_adds_calibrated_map():
    from flimkit.FLIM.fit_tools import estimate_bg
    from flimkit.FLIM.fitters import fit_per_pixel_dist
    from flimkit.FLIM.models import dist_reconvolution_model

    n_bins = 128
    tcspc_res = 0.1e-9
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    global_popt = np.array([2e-9, 0.3e-9, 20.0, 0.0])
    model = dist_reconvolution_model(
        global_popt, tcspc_res, n_bins, irf, 1, 'gaussian',
        0.0, False, False)
    decay = np.random.default_rng(11).poisson(model).astype(float)

    maps = fit_per_pixel_dist(
        decay[None, None, :], tcspc_res, n_bins, irf,
        global_popt, 1, 'gaussian', fit_bg=False, fit_sigma=False,
        min_photons=1, n_tau_grid=20, n_width_grid=15,
        use_gpu=False)
    tau = maps['tau_center_1'][0, 0] * 1e-9
    width = maps['width_1'][0, 0] * 1e-9
    amp = maps['alpha_1'][0, 0]
    bg = estimate_bg(decay, int(np.argmax(decay)))
    fitted_model = dist_reconvolution_model(
        np.array([tau, width, amp, 0.0]), tcspc_res, n_bins,
        irf, 1, 'gaussian', bg, False, False)

    assert maps['calibrated_chi2_r'][0, 0] == pytest.approx(
        calibrated_chi2(decay, fitted_model))


def test_per_pixel_multicomponent_distribution_adds_calibrated_map():
    from flimkit.FLIM.fit_tools import estimate_bg
    from flimkit.FLIM.fitters import fit_per_pixel_dist
    from flimkit.FLIM.models import dist_reconvolution_model

    n_bins = 64
    tcspc_res = 0.1e-9
    irf = np.zeros(n_bins)
    irf[0] = 1.0
    global_popt = np.array([
        0.8e-9, 2.5e-9, 0.15e-9, 0.4e-9, 30.0, 20.0, 0.0])
    model = dist_reconvolution_model(
        global_popt, tcspc_res, n_bins, irf, 2, 'gaussian',
        0.0, False, False)
    decay = np.random.default_rng(13).poisson(model).astype(float)

    maps = fit_per_pixel_dist(
        decay[None, None, :], tcspc_res, n_bins, irf,
        global_popt, 2, 'gaussian', fit_bg=False, fit_sigma=False,
        min_photons=1, use_gpu=False)
    fitted_popt = np.array([
        maps['tau_center_1'][0, 0] * 1e-9,
        maps['tau_center_2'][0, 0] * 1e-9,
        maps['width_1'][0, 0] * 1e-9,
        maps['width_2'][0, 0] * 1e-9,
        maps['alpha_1'][0, 0], maps['alpha_2'][0, 0], 0.0])
    bg = estimate_bg(decay, int(np.argmax(decay)))
    fitted_model = dist_reconvolution_model(
        fitted_popt, tcspc_res, n_bins, irf, 2, 'gaussian',
        bg, False, False)

    assert maps['calibrated_chi2_r'][0, 0] == pytest.approx(
        calibrated_chi2(decay, fitted_model))


def test_batch_exports_include_calibrated_map():
    from flimkit.utils.batch_fit import _STACK_MAPS

    assert 'calibrated_chi2_r' in _STACK_MAPS


def test_stitch_adapter_keeps_calibrated_map():
    from flimkit.formats.PTU.stitch import _adapt_pixel_maps

    pixel_maps = {
        'intensity': np.ones((1, 1)),
        'tau_mean_amp': np.ones((1, 1)),
        'chi2_r': np.ones((1, 1)),
        'calibrated_chi2_r': np.full((1, 1), 1.25),
    }

    adapted = _adapt_pixel_maps(pixel_maps, 1, np.array([2.0]))

    assert adapted['calibrated_chi2_r'][0, 0] == pytest.approx(1.25)


def test_assembled_canvas_keeps_calibrated_map():
    from flimkit.FLIM.assemble import assemble_tile_maps

    pixel_maps = {
        'intensity': np.ones((1, 1)),
        'tau_mean_amp': np.ones((1, 1)),
        'calibrated_chi2_r': np.full((1, 1), 1.25),
        'tau1': np.full((1, 1), 2.0),
        'a1': np.ones((1, 1)),
    }
    tile_results = [{
        'pixel_maps': pixel_maps,
        'pixel_y': 0,
        'pixel_x': 0,
        'tile_h': 1,
        'tile_w': 1,
    }]

    canvas = assemble_tile_maps(tile_results, 1, 1, 1)

    assert canvas['calibrated_chi2_r'][0, 0] == pytest.approx(1.25)


def test_zstack_display_restores_calibrated_summed_values(tmp_path):
    import json
    from flimkit.UI.flim_display import load_zstack_display_slices

    reference = {
        'taus_ns': [2.0],
        'nexp': 1,
        'calibrated_chi2_pearson': 1.1,
        'calibrated_chi2_tail_pearson': 0.9,
    }
    (tmp_path / 'sample_reference_fit.json').write_text(json.dumps(reference))
    slice_dir = tmp_path / 'z0000'
    slice_dir.mkdir()
    np.save(slice_dir / 'intensity.npy', np.ones((1, 1)))

    slices = load_zstack_display_slices(tmp_path)
    summary = slices[0]['fit_result']['global_summary']

    assert summary['calibrated_chi2_pearson'] == pytest.approx(1.1)
    assert summary['calibrated_chi2_tail_pearson'] == pytest.approx(0.9)
