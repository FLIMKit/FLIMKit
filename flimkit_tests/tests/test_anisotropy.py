import numpy as np
import pytest


def test_polarized_decay_models_follow_lakowicz_channel_equations():
    from flimkit.FLIM.anisotropy import polarized_decay_models

    time_ns = np.arange(6, dtype=float)
    delta_irf = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fluorescence = 300.0 * np.exp(-time_ns / 4.0)
    anisotropy = 0.3 * np.exp(-time_ns / 2.0)

    parallel, perpendicular = polarized_decay_models(
        time_ns, delta_irf, delta_irf,
        intensity_lifetime_ns=4.0, rotational_correlation_ns=2.0,
        initial_anisotropy=0.3, amplitude=300.0,
        g_factor=2.0, parallel_exposure=2.0,
        perpendicular_exposure=1.0,
        parallel_background=5.0, perpendicular_background=7.0)

    expected_parallel = 5.0 + 2.0 * fluorescence * (1.0 + 2.0 * anisotropy) / 3.0
    expected_perpendicular = 7.0 + fluorescence * (1.0 - anisotropy) / 6.0
    np.testing.assert_allclose(parallel, expected_parallel)
    np.testing.assert_allclose(perpendicular, expected_perpendicular)


def test_polarized_decay_models_include_previous_laser_pulses():
    from flimkit.FLIM.anisotropy import polarized_decay_models

    time_ns = np.arange(4, dtype=float)
    delta_irf = np.array([1.0, 0.0, 0.0, 0.0])
    tau_ns = 2.0
    theta_ns = 1.0
    period_ns = 4.0
    amplitude = 90.0
    r0 = 0.3

    parallel, perpendicular = polarized_decay_models(
        time_ns, delta_irf, delta_irf,
        intensity_lifetime_ns=tau_ns,
        rotational_correlation_ns=theta_ns,
        initial_anisotropy=r0, amplitude=amplitude,
        repetition_period_ns=period_ns)

    intensity = (amplitude * np.exp(-time_ns / tau_ns)
                 / (1.0 - np.exp(-period_ns / tau_ns)))
    effective_ns = 1.0 / (1.0 / tau_ns + 1.0 / theta_ns)
    polarized = (amplitude * r0 * np.exp(-time_ns / effective_ns)
                 / (1.0 - np.exp(-period_ns / effective_ns)))
    np.testing.assert_allclose(parallel, (intensity + 2.0 * polarized) / 3.0)
    np.testing.assert_allclose(perpendicular, (intensity - polarized) / 3.0)


def test_polarized_decay_models_reject_mismatched_laser_period():
    from flimkit.FLIM.anisotropy import polarized_decay_models

    time_ns = np.arange(8, dtype=float)
    irf = np.eye(1, 8, 0).ravel()

    with pytest.raises(ValueError, match='histogram duration'):
        polarized_decay_models(
            time_ns, irf, irf, intensity_lifetime_ns=3.0,
            rotational_correlation_ns=1.0, initial_anisotropy=0.2,
            amplitude=100.0, repetition_period_ns=6.0)


def test_polarized_decay_models_apply_common_irf_shift():
    from flimkit.FLIM.anisotropy import polarized_decay_models

    time_ns = np.arange(8, dtype=float)
    delta_irf = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    unshifted, _ = polarized_decay_models(
        time_ns, delta_irf, delta_irf,
        intensity_lifetime_ns=3.0, rotational_correlation_ns=1.0,
        initial_anisotropy=0.2, amplitude=100.0)
    shifted, _ = polarized_decay_models(
        time_ns, delta_irf, delta_irf,
        intensity_lifetime_ns=3.0, rotational_correlation_ns=1.0,
        initial_anisotropy=0.2, amplitude=100.0,
        common_irf_shift_bins=1.0)

    np.testing.assert_allclose(shifted[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(shifted[1:], unshifted[:-1])


def test_global_polarized_fit_recovers_rotation_shift_and_backgrounds():
    from flimkit.FLIM.anisotropy import (
        fit_polarized_decays, polarized_decay_models)

    time_ns = np.arange(128, dtype=float) * 0.1
    bins = np.arange(time_ns.size, dtype=float)
    parallel_irf = np.exp(-0.5 * ((bins - 8.0) / 1.2) ** 2)
    perpendicular_irf = np.exp(-0.5 * ((bins - 11.0) / 1.8) ** 2)
    parallel, perpendicular = polarized_decay_models(
        time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.2,
        rotational_correlation_ns=1.4,
        initial_anisotropy=0.32, amplitude=12000.0,
        g_factor=1.3, parallel_exposure=1.5,
        perpendicular_exposure=0.8,
        parallel_background=2.5, perpendicular_background=7.0,
        repetition_period_ns=12.8, common_irf_shift_bins=0.7)

    fitted = fit_polarized_decays(
        parallel, perpendicular, time_ns,
        parallel_irf=parallel_irf, perpendicular_irf=perpendicular_irf,
        intensity_lifetime_ns=3.2,
        g_factor=1.3, parallel_exposure=1.5,
        perpendicular_exposure=0.8,
        initial_parallel_background=0.0,
        initial_perpendicular_background=0.0,
        repetition_period_ns=12.8,
        initial_rotational_ns=2.0, initial_anisotropy=0.2)

    assert fitted.success
    assert fitted.intensity_lifetime_ns == 3.2
    assert fitted.rotational_correlation_ns == pytest.approx(1.4, rel=1e-4)
    assert fitted.initial_anisotropy == pytest.approx(0.32, rel=1e-4)
    assert fitted.common_irf_shift_bins == pytest.approx(0.7, abs=1e-4)
    assert fitted.parallel_background == pytest.approx(2.5, rel=1e-3)
    assert fitted.perpendicular_background == pytest.approx(7.0, rel=1e-3)
    np.testing.assert_allclose(fitted.parallel_model, parallel, rtol=1e-5)
    np.testing.assert_allclose(fitted.perpendicular_model, perpendicular, rtol=1e-5)


def test_global_fit_avoids_unequal_irf_bias_from_divided_anisotropy():
    from scipy.optimize import curve_fit
    from flimkit.FLIM.anisotropy import (
        fit_polarized_decays, polarized_decay_models)

    time_ns = np.arange(256, dtype=float) * 0.1
    bins = np.arange(time_ns.size, dtype=float)
    parallel_irf = np.exp(-0.5 * ((bins - 8.0) / 1.2) ** 2)
    perpendicular_irf = np.exp(-0.5 * ((bins - 11.0) / 2.6) ** 2)
    perpendicular_irf *= 0.9 + 0.1 * np.exp(
        -np.maximum(bins - 11.0, 0.0) / 5.0)
    parallel, perpendicular = polarized_decay_models(
        time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, rotational_correlation_ns=0.8,
        initial_anisotropy=0.36, amplitude=30000.0,
        g_factor=0.78, parallel_exposure=1.0,
        perpendicular_exposure=1.6,
        parallel_background=2.5, perpendicular_background=7.0,
        repetition_period_ns=25.6, common_irf_shift_bins=0.7)

    fitted = fit_polarized_decays(
        parallel, perpendicular, time_ns, parallel_irf, perpendicular_irf,
        intensity_lifetime_ns=3.0, g_factor=0.78,
        parallel_exposure=1.0, perpendicular_exposure=1.6,
        initial_parallel_background=2.5,
        initial_perpendicular_background=7.0,
        repetition_period_ns=25.6)

    parallel_rate = parallel - 2.5
    perpendicular_rate = 0.78 * (perpendicular - 7.0) / 1.6
    ratio = ((parallel_rate - perpendicular_rate)
             / (parallel_rate + 2.0 * perpendicular_rate))
    peak = int(np.argmax(parallel + 2.0 * 0.78 * perpendicular / 1.6))
    fit_slice = slice(peak, peak + 70)
    relative_time = time_ns[fit_slice] - time_ns[peak]
    ratio_parameters, _ = curve_fit(
        lambda time, r0, theta: r0 * np.exp(-time / theta),
        relative_time, ratio[fit_slice], p0=(0.3, 1.0),
        bounds=([-0.2, 0.05], [0.8, 10.0]))

    assert fitted.rotational_correlation_ns == pytest.approx(0.8, rel=0.005)
    assert abs(ratio_parameters[1] / 0.8 - 1.0) > 0.2


def test_global_polarized_fit_reports_parameters_at_bounds():
    from flimkit.FLIM.anisotropy import (
        fit_polarized_decays, polarized_decay_models)

    time_ns = np.arange(64, dtype=float) * 0.1
    irf = np.eye(1, 64, 5).ravel()
    parallel, perpendicular = polarized_decay_models(
        time_ns, irf, irf,
        intensity_lifetime_ns=3.0, rotational_correlation_ns=1.0,
        initial_anisotropy=-0.3, amplitude=5000.0,
        repetition_period_ns=6.4)

    fitted = fit_polarized_decays(
        parallel, perpendicular, time_ns, irf, irf,
        intensity_lifetime_ns=3.0, repetition_period_ns=6.4,
        initial_anisotropy=-0.1)

    assert 'initial_anisotropy' in fitted.parameters_at_bounds


def test_calculate_anisotropy_uses_parallel_perpendicular_formula():
    from flimkit.FLIM.anisotropy import calculate_anisotropy

    parallel = np.array([60.0, 30.0])
    perpendicular = np.array([20.0, 10.0])

    result = calculate_anisotropy(parallel, perpendicular)

    np.testing.assert_allclose(result, [0.4, 0.4])


def test_calculate_anisotropy_applies_g_factor():
    from flimkit.FLIM.anisotropy import calculate_anisotropy

    result = calculate_anisotropy(
        np.array([80.0]), np.array([20.0]), g_factor=2.0)

    np.testing.assert_allclose(result, [0.25])


def test_calculate_anisotropy_normalizes_exposure():
    from flimkit.FLIM.anisotropy import calculate_anisotropy

    result = calculate_anisotropy(
        np.array([120.0]), np.array([20.0]),
        parallel_exposure=2.0, perpendicular_exposure=1.0)

    np.testing.assert_allclose(result, [0.4])


def test_calculate_anisotropy_masks_low_denominators():
    from flimkit.FLIM.anisotropy import calculate_anisotropy

    result = calculate_anisotropy(
        np.array([40.0, 400.0]), np.array([20.0, 100.0]),
        min_denominator=100.0)

    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1], 0.5)


def test_spatial_window_sum_uses_overlapping_valid_windows():
    from flimkit.FLIM.anisotropy import spatial_window_sum

    cube = np.arange(16, dtype=float).reshape(4, 4, 1)

    pooled = spatial_window_sum(cube, window_size=3, stride=1)

    assert pooled.shape == (2, 2, 1)
    np.testing.assert_allclose(pooled[..., 0], [[45.0, 54.0],
                                                [81.0, 90.0]])


def test_subtract_background_uses_per_decay_prepeak_median():
    from flimkit.FLIM.anisotropy import subtract_background

    data = np.array([[2.0, 2.0, 6.0, 10.0],
                     [1.0, 3.0, 7.0, 11.0]])

    corrected, background = subtract_background(data, slice(0, 2))

    np.testing.assert_allclose(background, [2.0, 2.0])
    np.testing.assert_allclose(corrected, [[0.0, 0.0, 4.0, 8.0],
                                           [-1.0, 1.0, 5.0, 9.0]])


def test_analyze_anisotropy_has_no_global_fit_by_default():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    data = np.ones((1, 1, 3), dtype=float)
    result = analyze_anisotropy(
        data, data, np.arange(3, dtype=float),
        background_bins=slice(0, 1))

    assert result.polarized_fit is None


def test_analyze_anisotropy_returns_decay_and_overlapping_map():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    parallel = np.full((5, 5, 6), 2.0)
    perpendicular = np.full((5, 5, 6), 3.0)
    parallel[..., 2:] = 22.0
    perpendicular[..., 2:] = 13.0

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(6, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 6),
        spatial_window=3, stride=1)

    assert result.anisotropy_cube.shape == (3, 3, 6)
    assert result.window_origins.shape == (3, 3, 2)
    np.testing.assert_array_equal(result.window_origins[0, 0], [0, 0])
    np.testing.assert_array_equal(result.window_origins[-1, -1], [2, 2])
    np.testing.assert_allclose(result.anisotropy_decay[2:], 0.25)
    np.testing.assert_allclose(result.anisotropy_map, 0.25)
    np.testing.assert_allclose(result.parallel_background, 50.0)
    np.testing.assert_allclose(result.perpendicular_background, 75.0)


def test_estimate_translation_finds_shift_to_align_moving_image():
    from scipy.ndimage import shift
    from flimkit.FLIM.anisotropy import estimate_translation

    y, x = np.mgrid[:64, :64]
    reference = np.exp(-((y - 24.0) ** 2 + (x - 35.0) ** 2) / 40.0)
    reference += 0.6 * np.exp(-((y - 43.0) ** 2 + (x - 15.0) ** 2) / 18.0)
    moving = shift(reference, (1.25, -0.75), order=1, mode='constant')

    estimated = estimate_translation(reference, moving)

    np.testing.assert_allclose(estimated, (-1.25, 0.75), atol=0.2)


def test_estimate_translation_rejects_featureless_images():
    from flimkit.FLIM.anisotropy import estimate_translation

    image = np.ones((16, 16), dtype=float)

    with pytest.raises(ValueError, match='spatial variation'):
        estimate_translation(image, image)


def test_estimate_translation_rejects_shift_at_search_boundary():
    from scipy.ndimage import shift
    from flimkit.FLIM.anisotropy import estimate_translation

    y, x = np.mgrid[:64, :64]
    reference = np.exp(-((y - 30.0) ** 2 + (x - 30.0) ** 2) / 30.0)
    moving = shift(reference, (4.0, 0.0), order=1, mode='constant')

    with pytest.raises(RuntimeError, match='search boundary'):
        estimate_translation(reference, moving, max_shift=1.0)


def test_estimate_translation_rejects_low_confidence_alignment():
    from flimkit.FLIM.anisotropy import estimate_translation

    generator = np.random.default_rng(20260808)
    reference = generator.random((64, 64))
    moving = generator.random((64, 64))

    with pytest.raises(RuntimeError, match='confidence'):
        estimate_translation(reference, moving)


def test_apply_translation_moves_only_spatial_axes():
    from flimkit.FLIM.anisotropy import apply_translation

    cube = np.zeros((5, 5, 2), dtype=float)
    cube[2, 1] = [1.0, 2.0]

    registered = apply_translation(cube, (0.0, 1.0))

    np.testing.assert_allclose(registered[2, 2], [1.0, 2.0])
    np.testing.assert_allclose(registered.sum(axis=(0, 1)), [1.0, 2.0])


def test_analyze_anisotropy_applies_perpendicular_registration_shift():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    parallel = np.ones((5, 5, 4), dtype=float)
    perpendicular = np.ones((5, 5, 4), dtype=float)
    parallel[2, 2, 2:] = 21.0
    perpendicular[2, 1, 2:] = 11.0

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(4, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 4),
        perpendicular_shift=(0.0, 1.0))

    np.testing.assert_allclose(result.anisotropy_map[2, 2], 0.25)
    np.testing.assert_allclose(result.perpendicular_shift, (0.0, 1.0))


def test_registration_masks_windows_without_full_perpendicular_support():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    parallel = np.ones((4, 4, 3), dtype=float)
    perpendicular = np.ones((4, 4, 3), dtype=float)
    parallel[..., 2] = 11.0
    perpendicular[..., 2] = 6.0

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(3, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 3),
        perpendicular_shift=(0.0, 1.0))

    assert not result.valid_mask[:, 0].any()
    assert not result.map_valid_mask[:, 0].any()
    assert result.map_valid_mask[:, 1:].all()


def test_analyze_ptu_pair_uses_reader_contract_and_explicit_orientation():
    from flimkit.FLIM.anisotropy import analyze_ptu_pair

    parallel = np.full((3, 3, 4), 2.0)
    perpendicular = np.full((3, 3, 4), 3.0)
    parallel[..., 2:] = 22.0
    perpendicular[..., 2:] = 13.0
    stacks = {'parallel.ptu': parallel, 'perpendicular.ptu': perpendicular}
    channels = {}

    class FakeReader:
        def __init__(self, path, verbose=False):
            self.path = path
            self.time_ns = np.arange(4, dtype=float)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def pixel_stack(self, channel=None):
            channels[self.path] = channel
            return stacks[self.path]

    result = analyze_ptu_pair(
        'parallel.ptu', 'perpendicular.ptu',
        parallel_channel=1, perpendicular_channel=2,
        background_bins=slice(0, 2), analysis_bins=slice(2, 4),
        min_bin_photons=7.0, min_map_photons=11.0,
        reader_class=FakeReader)

    np.testing.assert_allclose(result.anisotropy_decay[2:], 0.25)
    assert channels == {'parallel.ptu': 1, 'perpendicular.ptu': 2}
    assert result.metadata['parallel_role'] == 'parallel'
    assert result.metadata['perpendicular_role'] == 'perpendicular'
    assert result.metadata['background_start_bin'] == 0
    assert result.metadata['background_stop_bin'] == 2
    assert result.metadata['analysis_start_bin'] == 2
    assert result.metadata['analysis_stop_bin'] == 4
    assert result.metadata['min_bin_photons'] == 7.0
    assert result.metadata['min_map_photons'] == 11.0


def test_analyze_ptu_pair_rejects_invalid_photon_channels():
    from flimkit.FLIM.anisotropy import analyze_ptu_pair

    with pytest.raises(ValueError, match='non-negative integers'):
        analyze_ptu_pair(
            'parallel.ptu', 'perpendicular.ptu',
            parallel_channel=-1, perpendicular_channel=0,
            background_bins=slice(0, 1), reader_class=object)


def test_analyze_anisotropy_masks_bins_using_observed_counts():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    parallel = np.zeros((1, 1, 4), dtype=float)
    perpendicular = np.zeros((1, 1, 4), dtype=float)
    parallel[..., 2:] = 40.0
    perpendicular[..., 2:] = 40.0

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(4, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 4),
        min_bin_photons=100.0)

    assert np.isnan(result.anisotropy_cube[..., 2:]).all()


def test_valid_masks_exclude_nonfinite_anisotropy_values():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    parallel = np.array([[[20.0, 20.0, 10.0]]])
    perpendicular = np.array([[[20.0, 20.0, 10.0]]])

    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(3, dtype=float),
        background_bins=slice(0, 2), analysis_bins=slice(2, 3))

    assert not result.valid_mask[0, 0, 2]
    assert not result.map_valid_mask[0, 0]
    assert np.isnan(result.anisotropy_cube[0, 0, 2])
    assert np.isnan(result.anisotropy_map[0, 0])


def test_analyze_anisotropy_rejects_nonfinite_photon_thresholds():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    data = np.ones((1, 1, 3), dtype=float)

    with pytest.raises(ValueError, match='Photon thresholds'):
        analyze_anisotropy(
            data, data, np.arange(3, dtype=float),
            background_bins=slice(0, 1), min_map_photons=np.nan)


def test_analyze_anisotropy_rejects_scalar_background_selector():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    data = np.ones((1, 1, 3), dtype=float)

    with pytest.raises(ValueError, match='background_bins'):
        analyze_anisotropy(
            data, data, np.arange(3, dtype=float), background_bins=0)


def test_analyze_anisotropy_rejects_scalar_analysis_selector():
    from flimkit.FLIM.anisotropy import analyze_anisotropy

    data = np.ones((1, 1, 3), dtype=float)

    with pytest.raises(ValueError, match='analysis_bins'):
        analyze_anisotropy(
            data, data, np.arange(3, dtype=float),
            background_bins=slice(0, 1), analysis_bins=2)


def test_save_anisotropy_npz_preserves_masks_and_safe_metadata(tmp_path):
    from flimkit.FLIM.anisotropy import (
        PolarizedFitResult, analyze_anisotropy, save_anisotropy_npz)

    parallel = np.ones((2, 2, 4), dtype=float)
    perpendicular = np.ones((2, 2, 4), dtype=float)
    result = analyze_anisotropy(
        parallel, perpendicular, np.arange(4, dtype=float),
        background_bins=slice(0, 1), analysis_bins=slice(1, 4))
    result.metadata.update({
        'parallel_file': 'parallel.ptu',
        'perpendicular_file': 'perpendicular.ptu',
        'g_factor': 999.0,
        'file': 'metadata-value',
    })
    result.polarized_fit = PolarizedFitResult(
        intensity_lifetime_ns=3.2,
        rotational_correlation_ns=1.4,
        initial_anisotropy=0.32,
        amplitude=12000.0,
        parallel_model=np.arange(3, dtype=float),
        perpendicular_model=np.arange(3, dtype=float) + 1.0,
        parallel_residual=np.zeros(3),
        perpendicular_residual=np.ones(3),
        poisson_deviance=5.0,
        success=True,
        message='complete',
        common_irf_shift_bins=0.7,
        parallel_background=2.5,
        perpendicular_background=7.0)
    path = tmp_path / 'result.npz'

    save_anisotropy_npz(result, path)

    saved = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(saved['valid_mask'], result.valid_mask)
    np.testing.assert_array_equal(saved['window_origins'], result.window_origins)
    assert saved['parallel_background'].item() == result.parallel_background
    assert saved['perpendicular_background'].item() == result.perpendicular_background
    assert saved['g_factor'].item() == result.g_factor
    assert 'metadata_g_factor' not in saved.files
    assert saved['metadata_file'].item() == 'metadata-value'
    assert saved['parallel_file'].item() == 'parallel.ptu'
    assert '/private/' not in saved['parallel_file'].item()
    assert saved['fit_intensity_lifetime_ns'].item() == 3.2
    assert saved['fit_rotational_correlation_ns'].item() == 1.4
    assert saved['fit_initial_anisotropy'].item() == 0.32
    assert saved['fit_common_irf_shift_bins'].item() == 0.7
    assert saved['fit_parallel_background'].item() == 2.5
    assert saved['fit_perpendicular_background'].item() == 7.0
    np.testing.assert_array_equal(
        saved['fit_parallel_observed'],
        (result.parallel_decay + result.parallel_background)[:3])
    np.testing.assert_array_equal(
        saved['fit_parallel_model'], result.polarized_fit.parallel_model)
