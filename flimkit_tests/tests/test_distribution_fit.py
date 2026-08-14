import pytest
import numpy as np
from pathlib import Path
import sys

_tests_dir   = str(Path(__file__).parent.parent)
_project_root = str(Path(__file__).parent.parent.parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mock_data import (
    generate_synthetic_gaussian_dist_decay,
    MOCK_TCSPC_RES,
    MOCK_IRF_CENTER,
    MOCK_IRF_FWHM_BINS,
)


def _irf(n_bins, tcspc_res, fwhm_ns, center_bin):
    from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
    return gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, center_bin)


class TestDistKernel:

    def test_gaussian_kernel_normalised(self):
        from flimkit.FLIM.models import _dist_kernel
        n_bins    = 128
        tcspc_res = MOCK_TCSPC_RES
        tau_c     = 2e-9
        sigma     = 0.3e-9
        k = _dist_kernel(tcspc_res, n_bins, tau_c, sigma, 1.0, 'gaussian')
        assert k.shape == (n_bins,)
        assert k.min() >= 0.0
        assert k.max() > 0.0

    def test_lorentzian_kernel_positive(self):
        from flimkit.FLIM.models import _dist_kernel
        n_bins    = 128
        tcspc_res = MOCK_TCSPC_RES
        tau_c     = 2e-9
        gamma     = 0.5e-9
        k = _dist_kernel(tcspc_res, n_bins, tau_c, gamma, 1.0, 'lorentzian')
        assert k.shape == (n_bins,)
        assert k.min() >= 0.0

    def test_amp_scales_kernel(self):
        from flimkit.FLIM.models import _dist_kernel
        k1 = _dist_kernel(MOCK_TCSPC_RES, 128, 2e-9, 0.3e-9, 1.0, 'gaussian')
        k2 = _dist_kernel(MOCK_TCSPC_RES, 128, 2e-9, 0.3e-9, 3.0, 'gaussian')
        np.testing.assert_allclose(k2, 3.0 * k1, rtol=1e-6)


class TestDistBasisGrid:

    def test_grid_shape(self):
        from flimkit.FLIM.models import build_dist_basis_grid
        n_bins    = 64
        tcspc_res = MOCK_TCSPC_RES
        n_tau, n_w = 5, 4
        tau_grid   = np.linspace(0.5e-9, 4.0e-9, n_tau)
        width_grid = np.linspace(0.1e-9, 0.8e-9, n_w)
        irf        = np.zeros(n_bins); irf[10] = 1.0
        basis, params = build_dist_basis_grid(tcspc_res, n_bins, irf, tau_grid, width_grid, 'gaussian')
        assert basis.shape  == (n_tau * n_w, n_bins)
        assert params.shape == (n_tau * n_w, 2)

    def test_param_pairs_match(self):
        from flimkit.FLIM.models import build_dist_basis_grid
        n_bins    = 64
        tau_grid  = np.array([1e-9, 2e-9])
        w_grid    = np.array([0.2e-9, 0.4e-9])
        irf       = np.zeros(n_bins); irf[5] = 1.0
        _, params = build_dist_basis_grid(MOCK_TCSPC_RES, n_bins, irf, tau_grid, w_grid, 'gaussian')
        expected_tau = np.array([1e-9, 1e-9, 2e-9, 2e-9], dtype=np.float32)
        expected_w   = np.array([0.2e-9, 0.4e-9, 0.2e-9, 0.4e-9], dtype=np.float32)
        np.testing.assert_allclose(params[:, 0], expected_tau, rtol=1e-5)
        np.testing.assert_allclose(params[:, 1], expected_w,   rtol=1e-5)


class TestGaussianDistRecovery:

    @pytest.fixture(params=[1.0, 2.5, 4.0], ids=['tau1ns', 'tau2p5ns', 'tau4ns'])
    def known_tau(self, request):
        return request.param

    def test_gaussian_tau_center_recovery(self, known_tau):
        try:
            from flimkit.FLIM.fitters import fit_summed_dist
        except ImportError:
            pytest.skip('fitters module not available')

        n_bins    = 256
        tcspc_res = MOCK_TCSPC_RES
        sigma_ns  = known_tau * 0.15
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9

        decay = generate_synthetic_gaussian_dist_decay(
            n_bins=n_bins, tcspc_res=tcspc_res,
            tau_center_ns=known_tau, sigma_ns=sigma_ns,
            bg=5.0, peak_counts=50_000.0,
            irf_fwhm_bins=MOCK_IRF_FWHM_BINS,
            irf_center_bin=MOCK_IRF_CENTER, noise=True,
        )
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

        _popt, summary = fit_summed_dist(
            decay, tcspc_res, n_bins, irf,
            n_components=1, dist_type='gaussian',
            fit_bg=True, fit_sigma=False,
            tau_min_ns=0.1, tau_max_ns=15.0,
            optimizer='lm_multistart', n_restarts=3, workers=1,
        )

        recovered = summary['tau_centers_ns'][0]
        rel_err   = abs(recovered - known_tau) / known_tau
        assert rel_err < 0.20, (
            f'Gaussian dist τ̄ recovery: true={known_tau:.2f} ns, '
            f'recovered={recovered:.3f} ns (rel err {rel_err:.1%})'
        )

    def test_summary_keys_present(self):
        try:
            from flimkit.FLIM.fitters import fit_summed_dist
        except ImportError:
            pytest.skip('fitters module not available')

        n_bins    = 256
        tcspc_res = MOCK_TCSPC_RES
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9

        decay = generate_synthetic_gaussian_dist_decay(
            n_bins=n_bins, tcspc_res=tcspc_res,
            tau_center_ns=2.0, sigma_ns=0.4,
            bg=5.0, peak_counts=50_000.0,
            irf_fwhm_bins=MOCK_IRF_FWHM_BINS,
            irf_center_bin=MOCK_IRF_CENTER, noise=False,
        )
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

        _popt, summary = fit_summed_dist(
            decay, tcspc_res, n_bins, irf,
            n_components=1, dist_type='gaussian',
            fit_bg=True, fit_sigma=False,
            tau_min_ns=0.1, tau_max_ns=15.0,
            optimizer='lm_multistart', n_restarts=2, workers=1,
        )

        for key in ('tau_centers_ns', 'widths_ns', 'fwhms_ns', 'amps', 'fractions',
                    'tau_mean_amp_ns', 'tau_mean_int_ns', 'reduced_chi2',
                    'dist_type', 'n_components', 'model', 'residuals'):
            assert key in summary, f'Missing key: {key}'

        assert summary['dist_type'] == 'gaussian'
        assert summary['n_components'] == 1
        assert len(summary['tau_centers_ns']) == 1
        assert summary['reduced_chi2'] > 0.0


class TestLorentzianDistRecovery:

    def test_lorentzian_tau_center_recovery(self):
        try:
            from flimkit.FLIM.fitters import fit_summed_dist
        except ImportError:
            pytest.skip('fitters module not available')

        n_bins     = 256
        tcspc_res  = MOCK_TCSPC_RES
        tau_ns     = 2.0
        sigma_ns   = tau_ns * 0.15
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9

        decay = generate_synthetic_gaussian_dist_decay(
            n_bins=n_bins, tcspc_res=tcspc_res,
            tau_center_ns=tau_ns, sigma_ns=sigma_ns,
            bg=5.0, peak_counts=50_000.0,
            irf_fwhm_bins=MOCK_IRF_FWHM_BINS,
            irf_center_bin=MOCK_IRF_CENTER, noise=False,
        )
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

        _popt, summary = fit_summed_dist(
            decay, tcspc_res, n_bins, irf,
            n_components=1, dist_type='lorentzian',
            fit_bg=True, fit_sigma=False,
            tau_min_ns=0.1, tau_max_ns=15.0,
            optimizer='lm_multistart', n_restarts=3, workers=1,
        )

        recovered = summary['tau_centers_ns'][0]
        rel_err   = abs(recovered - tau_ns) / tau_ns
        assert rel_err < 0.25, (
            f'Lorentzian dist τ̄ recovery: true={tau_ns:.2f} ns, '
            f'recovered={recovered:.3f} ns (rel err {rel_err:.1%})'
        )


class TestDistPerPixelCPU:

    def test_unimodal_grid_scan_shape(self):
        try:
            from flimkit.FLIM.fitters import fit_summed_dist, fit_per_pixel_dist
        except ImportError:
            pytest.skip('fitters module not available')

        n_bins    = 128
        tcspc_res = MOCK_TCSPC_RES
        ny, nx    = 4, 4
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)

        decay = generate_synthetic_gaussian_dist_decay(
            n_bins=n_bins, tcspc_res=tcspc_res,
            tau_center_ns=2.0, sigma_ns=0.3,
            bg=5.0, peak_counts=5_000.0,
            irf_fwhm_bins=MOCK_IRF_FWHM_BINS,
            irf_center_bin=MOCK_IRF_CENTER, noise=False,
        )
        _popt, _summary = fit_summed_dist(
            decay, tcspc_res, n_bins, irf,
            n_components=1, dist_type='gaussian',
            fit_bg=True, fit_sigma=False,
            tau_min_ns=0.1, tau_max_ns=15.0,
            optimizer='lm_multistart', n_restarts=2, workers=1,
        )

        rng   = np.random.default_rng(1)
        stack = rng.poisson(decay[None, None, :] * np.ones((ny, nx, 1))).astype(np.float32)

        maps = fit_per_pixel_dist(
            stack, tcspc_res, n_bins, irf,
            _popt, n_components=1, dist_type='gaussian',
            fit_bg=True, fit_sigma=False,
            min_photons=10,
            n_tau_grid=10, n_width_grid=8,
            use_gpu=False,
        )

        assert maps['tau_center_1'].shape == (ny, nx)
        assert maps['tau_mean_amp'].shape == (ny, nx)
        assert not np.all(np.isnan(maps['tau_center_1']))

    @pytest.mark.parametrize('dist_type', ['gaussian', 'lorentzian'])
    def test_unimodal_grid_scan_ignores_excluded_bins(self, dist_type):
        from flimkit.FLIM.fitters import fit_summed_dist, fit_per_pixel_dist

        n_bins = 128
        tcspc_res = MOCK_TCSPC_RES
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)
        decay = generate_synthetic_gaussian_dist_decay(
            n_bins=n_bins, tcspc_res=tcspc_res,
            tau_center_ns=2.0, sigma_ns=0.3,
            bg=5.0, peak_counts=5_000.0,
            irf_fwhm_bins=MOCK_IRF_FWHM_BINS,
            irf_center_bin=MOCK_IRF_CENTER, noise=False,
        )
        popt, _ = fit_summed_dist(
            decay, tcspc_res, n_bins, irf,
            n_components=1, dist_type=dist_type,
            fit_bg=True, fit_sigma=False,
            tau_min_ns=0.1, tau_max_ns=15.0,
            optimizer='lm_multistart', n_restarts=2, workers=1,
        )
        reflected = decay.copy()
        reflected[80:85] += 2_000.0
        fit_idx = np.setdiff1d(np.arange(n_bins), np.arange(80, 85))
        kwargs = dict(
            tcspc_res=tcspc_res, n_bins=n_bins, irf_prompt=irf,
            global_popt=popt, n_components=1, dist_type=dist_type,
            fit_bg=True, fit_sigma=False, min_photons=10,
            n_tau_grid=10, n_width_grid=8, fit_idx=fit_idx,
            use_gpu=False,
        )

        clean = fit_per_pixel_dist(decay[None, None, :], **kwargs)
        with_reflection = fit_per_pixel_dist(reflected[None, None, :], **kwargs)

        assert with_reflection['intensity'][0, 0] == pytest.approx(reflected.sum())
        assert with_reflection['intensity'][0, 0] > clean['intensity'][0, 0]
        for key in ('tau_center_1', 'width_1', 'alpha_1',
                    'chi2_r', 'calibrated_chi2_r'):
            np.testing.assert_allclose(clean[key], with_reflection[key], rtol=1e-12)

    def test_use_gpu_false_ignores_supplied_backend(self, monkeypatch):
        from flimkit.FLIM.fitters import fit_summed_dist, fit_per_pixel_dist
        from flimkit.FLIM.models import dist_reconvolution_model
        import flimkit.FLIM.fitters as fitters_module

        class FailingBackend:
            def batch_dist_scan_unimodal(self, *args, **kwargs):
                raise AssertionError('backend must not run when use_gpu=False')

        n_bins = 64
        tcspc_res = 0.05e-9
        irf = np.zeros(n_bins)
        irf[0] = 1.0
        decay = dist_reconvolution_model(
            np.array([2.0e-9, 0.3e-9, 3_000.0, 0.0, 5.0]),
            tcspc_res, n_bins, irf, 1, 'gaussian',
            bg_fixed=None, fit_bg=True, fit_sigma=False,
        )
        popt, _ = fit_summed_dist(
            decay, tcspc_res, n_bins, irf,
            n_components=1, dist_type='gaussian',
            fit_bg=True, fit_sigma=False,
            tau_min_ns=0.1, tau_max_ns=15.0,
            optimizer='lm_multistart', n_restarts=2, workers=1,
        )

        bg_input_sizes = []
        original_estimate_bg = fitters_module.estimate_bg

        def _record_bg_input(decay_input, peak_bin):
            bg_input_sizes.append(len(decay_input))
            return original_estimate_bg(decay_input, peak_bin)

        monkeypatch.setattr(fitters_module, 'estimate_bg', _record_bg_input)

        maps = fit_per_pixel_dist(
            decay[None, None, :], tcspc_res, n_bins, irf,
            popt, 1, 'gaussian', fit_bg=True, fit_sigma=False,
            min_photons=10, n_tau_grid=8, n_width_grid=6,
            use_gpu=False, gpu_backend=FailingBackend(),
        )
        cropped = fit_per_pixel_dist(
            decay[None, None, :], tcspc_res, n_bins, irf,
            popt, 1, 'gaussian', fit_bg=True, fit_sigma=False,
            min_photons=10, n_tau_grid=8, n_width_grid=6,
            use_gpu=False, fit_idx=np.arange(10, 50),
        )
        assert np.isfinite(maps['tau_center_1'][0, 0])
        assert np.isfinite(cropped['tau_center_1'][0, 0])
        assert bg_input_sizes == [n_bins, n_bins]

    def test_multicomponent_fit_ignores_excluded_bins(self):
        from flimkit.FLIM.fitters import fit_per_pixel_dist
        from flimkit.FLIM.models import dist_reconvolution_model

        n_bins = 64
        tcspc_res = 0.1e-9
        irf = np.zeros(n_bins)
        irf[0] = 1.0
        popt = np.array([
            0.8e-9, 2.5e-9, 0.15e-9, 0.4e-9,
            30.0, 20.0, 0.0,
        ])
        decay = dist_reconvolution_model(
            popt, tcspc_res, n_bins, irf, 2, 'gaussian',
            0.0, False, False,
        )
        reflected = decay.copy()
        reflected[20:25] += 10.0
        fit_idx = np.setdiff1d(np.arange(n_bins), np.arange(20, 25))

        def _fit(one_decay):
            return fit_per_pixel_dist(
                one_decay[None, None, :], tcspc_res, n_bins, irf,
                popt, n_components=2, dist_type='gaussian',
                fit_bg=False, fit_sigma=False, min_photons=1,
                fit_idx=fit_idx, use_gpu=False,
            )

        clean = _fit(decay)
        with_reflection = _fit(reflected)

        for key in ('tau_center_1', 'tau_center_2', 'width_1', 'width_2',
                    'alpha_1', 'alpha_2', 'chi2_r', 'calibrated_chi2_r'):
            np.testing.assert_allclose(clean[key], with_reflection[key], rtol=1e-10)

    def test_backend_grid_scan_ignores_excluded_bins(self):
        from flimkit.FLIM.fitters import fit_summed_dist, fit_per_pixel_dist
        from flimkit.GPU import get_backend

        backend = get_backend()
        if backend is None:
            pytest.skip('no accelerated backend available')
        n_bins = 128
        tcspc_res = MOCK_TCSPC_RES
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)
        decay = generate_synthetic_gaussian_dist_decay(
            n_bins=n_bins, tcspc_res=tcspc_res,
            tau_center_ns=2.0, sigma_ns=0.3,
            bg=5.0, peak_counts=5_000.0,
            irf_fwhm_bins=MOCK_IRF_FWHM_BINS,
            irf_center_bin=MOCK_IRF_CENTER, noise=False,
        )
        popt, _ = fit_summed_dist(
            decay, tcspc_res, n_bins, irf,
            n_components=1, dist_type='gaussian',
            fit_bg=True, fit_sigma=False,
            tau_min_ns=0.1, tau_max_ns=15.0,
            optimizer='lm_multistart', n_restarts=2, workers=1,
        )
        reflected = decay.copy()
        reflected[80:85] += 2_000.0
        fit_idx = np.setdiff1d(np.arange(n_bins), np.arange(80, 85))

        def _fit(one_decay):
            return fit_per_pixel_dist(
                one_decay[None, None, :], tcspc_res, n_bins, irf,
                popt, n_components=1, dist_type='gaussian',
                fit_bg=True, fit_sigma=False, min_photons=10,
                n_tau_grid=10, n_width_grid=8, fit_idx=fit_idx,
                gpu_backend=backend,
            )

        clean = _fit(decay)
        with_reflection = _fit(reflected)
        cpu_clean = fit_per_pixel_dist(
            decay[None, None, :], tcspc_res, n_bins, irf,
            popt, n_components=1, dist_type='gaussian',
            fit_bg=True, fit_sigma=False, min_photons=10,
            n_tau_grid=10, n_width_grid=8, fit_idx=fit_idx,
            use_gpu=False,
        )

        for key in ('tau_center_1', 'width_1', 'alpha_1',
                    'chi2_r', 'calibrated_chi2_r'):
            np.testing.assert_allclose(clean[key], with_reflection[key], rtol=1e-6)
        np.testing.assert_allclose(clean['chi2_r'], cpu_clean['chi2_r'], rtol=1e-5)

    @pytest.mark.parametrize('backend_mode', ['cpu', 'backend'])
    def test_tvb_grid_scan_ignores_excluded_bins(self, backend_mode):
        from flimkit.FLIM.fitters import fit_summed_dist, fit_per_pixel_dist
        from flimkit.GPU import get_backend

        backend = None
        if backend_mode == 'backend':
            backend = get_backend()
            if backend is None:
                pytest.skip('no accelerated backend available')
        n_bins = 128
        tcspc_res = MOCK_TCSPC_RES
        irf_fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9
        irf = _irf(n_bins, tcspc_res, irf_fwhm_ns, MOCK_IRF_CENTER)
        decay = generate_synthetic_gaussian_dist_decay(
            n_bins=n_bins, tcspc_res=tcspc_res,
            tau_center_ns=2.0, sigma_ns=0.3,
            bg=5.0, peak_counts=5_000.0,
            irf_fwhm_bins=MOCK_IRF_FWHM_BINS,
            irf_center_bin=MOCK_IRF_CENTER, noise=False,
        )
        popt, _ = fit_summed_dist(
            decay, tcspc_res, n_bins, irf,
            n_components=1, dist_type='gaussian',
            fit_bg=True, fit_sigma=False,
            tau_min_ns=0.1, tau_max_ns=15.0,
            optimizer='lm_multistart', n_restarts=2, workers=1,
        )
        reflected = decay.copy()
        reflected[80:85] += 2_000.0
        fit_idx = np.setdiff1d(np.arange(n_bins), np.arange(80, 85))
        tvb_profile = np.exp(-np.arange(n_bins, dtype=float) / 30.0)

        def _fit(one_decay):
            return fit_per_pixel_dist(
                one_decay[None, None, :], tcspc_res, n_bins, irf,
                popt, n_components=1, dist_type='gaussian',
                fit_bg=True, fit_sigma=False, min_photons=10,
                n_tau_grid=10, n_width_grid=8, fit_idx=fit_idx,
                use_gpu=False if backend_mode == 'cpu' else 'auto',
                gpu_backend=backend, tvb_profile=tvb_profile, fit_tvb=True,
            )

        clean = _fit(decay)
        with_reflection = _fit(reflected)

        for key in ('tau_center_1', 'width_1', 'alpha_1', 'tvb_scale',
                    'chi2_r', 'calibrated_chi2_r'):
            np.testing.assert_allclose(clean[key], with_reflection[key], rtol=1e-6)
