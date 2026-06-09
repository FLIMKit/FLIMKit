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
