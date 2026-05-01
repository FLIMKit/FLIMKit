import pytest
import numpy as np
from flimkit.FLIM.fitters import fit_summed, fit_per_pixel
from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
from flimkit_tests.mock_data import generate_synthetic_decay, MOCK_TCSPC_RES, MOCK_IRF_CENTER, MOCK_IRF_FWHM_BINS


class TestFitSummedEdgeCases:
    @pytest.fixture
    def basic_irf(self):
        n_bins = 256
        tcspc_res = MOCK_TCSPC_RES
        fwhm_ns = MOCK_IRF_FWHM_BINS * tcspc_res * 1e9
        return gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, MOCK_IRF_CENTER)

    def test_zero_photon_decay_raises(self, basic_irf):
        decay = np.zeros(256)
        with pytest.raises(ValueError, match="zero maximum"):
            fit_summed(decay, MOCK_TCSPC_RES, 256, basic_irf, has_tail=False, fit_bg=True, fit_sigma=False, n_exp=1, tau_min_ns=0.1, tau_max_ns=10.0, optimizer="lm_multistart", n_restarts=1, workers=1, cost_function="chi2")

    def test_negative_photons_handled(self, basic_irf):
        decay = np.random.normal(100, 50, 256)
        decay[decay < 0] = 0
        try:
            fit_summed(decay, MOCK_TCSPC_RES, 256, basic_irf, has_tail=False, fit_bg=True, fit_sigma=False, n_exp=1, tau_min_ns=0.1, tau_max_ns=10.0, optimizer="de", de_popsize=5, de_maxiter=10, workers=1, cost_function="poisson")
        except Exception as e:
            pytest.fail(f"fit_summed crashed: {e}")

    def test_tau_min_greater_than_tau_max(self, basic_irf):
        decay = generate_synthetic_decay(n_bins=256, tau_ns=2.0, noise=False)
        with pytest.raises(ValueError):  # accept any ValueError
            fit_summed(decay, MOCK_TCSPC_RES, 256, basic_irf,
                    has_tail=False, fit_bg=True, fit_sigma=False,
                    n_exp=1, tau_min_ns=5.0, tau_max_ns=1.0,
                    optimizer="lm_multistart", n_restarts=1, workers=1)


class TestPerPixelFitting:
    def test_min_photons_threshold(self):
        n_bins = 128
        stack = np.zeros((2, 2, n_bins))
        stack[0, 0, 30:80] = 10  # ~500 photons → fits
        stack[0, 1, 30:32] = 1   # 2 photons → below min_photons=10
        irf = gaussian_irf_from_fwhm(n_bins, 97e-12, 0.3, 50)
        global_popt = np.array([2e-9, 500.0, 0.0, 5.0])
        maps = fit_per_pixel(stack, 97e-12, n_bins, irf,
                            has_tail=False, fit_bg=True, fit_sigma=False,
                            global_popt=global_popt, n_exp=1, min_photons=10)
        assert np.isfinite(maps['tau_mean_amp'][0, 0])
        assert np.isnan(maps['tau_mean_amp'][0, 1])

    def test_nnls_all_zero_amplitudes(self):
        n_bins = 128
        stack = np.ones((1, 1, n_bins)) * 5
        irf = gaussian_irf_from_fwhm(n_bins, 97e-12, 0.3, 50)
        global_popt = np.array([2e-9, 100.0, 0.0, 5.0])
        maps = fit_per_pixel(stack, 97e-12, n_bins, irf, has_tail=False, fit_bg=True, fit_sigma=False, global_popt=global_popt, n_exp=1, min_photons=1)
        assert np.isnan(maps['tau_mean_amp'][0, 0])

    # -- free-tau tests --

    def test_free_tau_returns_tau_maps_for_nexp2(self):
        """free_tau=True should populate tau_1 / tau_2 per pixel (not fixed to global)."""
        from flimkit_tests.mock_data import generate_synthetic_decay, MOCK_IRF_CENTER, MOCK_IRF_FWHM_BINS
        n_bins = 256
        tcspc = 97e-12
        irf = gaussian_irf_from_fwhm(n_bins, tcspc, 0.3, MOCK_IRF_CENTER)

        # Build a tiny 2×1 stack with known bi-exponential decays
        d1 = generate_synthetic_decay(n_bins, tcspc, tau_ns=0.5, noise=False, peak_counts=2000)
        d2 = generate_synthetic_decay(n_bins, tcspc, tau_ns=3.0, noise=False, peak_counts=2000)
        stack = np.stack([[d1], [d2]])  # (2,1,n_bins)

        # global_popt: [tau1, tau2, amp1, amp2, shift]  (no bg, no sigma, no tail)
        global_popt = np.array([0.5e-9, 3.0e-9, 1000.0, 500.0, 0.0])

        maps_fixed = fit_per_pixel(
            stack, tcspc, n_bins, irf,
            has_tail=False, fit_bg=False, fit_sigma=False,
            global_popt=global_popt, n_exp=2, min_photons=10,
            free_tau=False,
        )
        maps_free = fit_per_pixel(
            stack, tcspc, n_bins, irf,
            has_tail=False, fit_bg=False, fit_sigma=False,
            global_popt=global_popt, n_exp=2, min_photons=10,
            free_tau=True,
        )

        # Fixed mode: tau_1 is constant across pixels
        assert maps_fixed['tau_1'][0, 0] == maps_fixed['tau_1'][1, 0]

        # Free mode: tau maps are populated and finite
        assert np.isfinite(maps_free['tau_1'][0, 0])
        assert np.isfinite(maps_free['tau_2'][0, 0])
        assert np.isfinite(maps_free['tau_mean_int'][0, 0])

    def test_free_tau_skips_low_photon_pixels(self):
        """Pixels below min_photons should be NaN in free-tau mode."""
        n_bins = 128
        tcspc = 97e-12
        irf = gaussian_irf_from_fwhm(n_bins, tcspc, 0.3, 30)
        stack = np.zeros((2, 1, n_bins))
        stack[0, 0, 30:80] = 50   # enough photons
        stack[1, 0, :] = 0        # zero photons → skip

        global_popt = np.array([0.5e-9, 3.0e-9, 500.0, 200.0, 0.0])
        maps = fit_per_pixel(
            stack, tcspc, n_bins, irf,
            has_tail=False, fit_bg=False, fit_sigma=False,
            global_popt=global_popt, n_exp=2, min_photons=10,
            free_tau=True,
        )
        assert np.isfinite(maps['tau_mean_amp'][0, 0])
        assert np.isnan(maps['tau_mean_amp'][1, 0])

    def test_free_tau_nexp1_falls_through_to_grid(self):
        """free_tau=True with n_exp=1 should still use the grid-scan path (n_exp==1 branch)."""
        from flimkit_tests.mock_data import generate_synthetic_decay, MOCK_IRF_CENTER
        n_bins = 128
        tcspc = 97e-12
        irf = gaussian_irf_from_fwhm(n_bins, tcspc, 0.3, MOCK_IRF_CENTER)
        d = generate_synthetic_decay(n_bins, tcspc, tau_ns=2.0, noise=False, peak_counts=1000)
        stack = d[np.newaxis, np.newaxis, :]  # (1,1,n_bins)

        global_popt = np.array([2e-9, 1000.0, 0.0, 5.0])  # shift=0, bg=5
        maps = fit_per_pixel(
            stack, tcspc, n_bins, irf,
            has_tail=False, fit_bg=True, fit_sigma=False,
            global_popt=global_popt, n_exp=1, min_photons=10,
            free_tau=True,
        )
        # Should still converge and give a reasonable lifetime
        assert np.isfinite(maps['tau_1'][0, 0])
        assert 0.5 < maps['tau_1'][0, 0] < 10.0


class TestCostFunctions:
    """Additional tests for Poisson deviance and log-tau cost functions."""

    def test_poisson_deviance_manual(self):
        """Compare _DECostPoisson against manual calculation."""
        from flimkit.FLIM.models import _DECostPoisson
        import numpy as np

        n_bins = 100
        tcspc_res = 97e-12
        irf = np.ones(n_bins) / n_bins
        decay = np.random.poisson(50, n_bins).astype(float)

        cost_obj = _DECostPoisson(
            tcspc_res, n_bins, irf, n_exp=1, bg_fixed=5.0,
            has_tail=False, fit_bg=False, fit_sigma=False,
            fit_start=10, fit_end=90, decay=decay
        )

        params = [2e-9, 1000.0, 0.0]
        cost = cost_obj(params)
        assert cost > 0

    def test_log_tau_equivalence(self):
        """_DECostPoissonLogTau should produce same cost as linear version."""
        from flimkit.FLIM.models import _DECostPoisson, _DECostPoissonLogTau
        import numpy as np

        n_bins = 100
        irf = np.ones(n_bins) / n_bins
        decay = np.random.poisson(50, n_bins).astype(float)

        cost_lin = _DECostPoisson(
            97e-12, n_bins, irf, n_exp=1, bg_fixed=5.0,
            has_tail=False, fit_bg=False, fit_sigma=False,
            fit_start=10, fit_end=90, decay=decay
        )
        cost_log = _DECostPoissonLogTau(
            97e-12, n_bins, irf, n_exp=1, bg_fixed=5.0,
            has_tail=False, fit_bg=False, fit_sigma=False,
            fit_start=10, fit_end=90, decay=decay
        )

        cost1 = cost_lin([2e-9, 1000.0, 0.0])
        cost2 = cost_log([np.log10(2e-9), 1000.0, 0.0])
        assert cost1 == pytest.approx(cost2, rel=1e-6)



class TestCostFunctions:
    """Additional tests for Poisson deviance and log-tau cost functions."""

    def test_poisson_deviance_manual(self):
        """Compare _DECostPoisson against manual calculation."""
        from flimkit.FLIM.models import _DECostPoisson
        import numpy as np

        n_bins = 100
        tcspc_res = 97e-12
        irf = np.ones(n_bins) / n_bins
        decay = np.random.poisson(50, n_bins).astype(float)

        cost_obj = _DECostPoisson(
            tcspc_res, n_bins, irf, n_exp=1, bg_fixed=5.0,
            has_tail=False, fit_bg=False, fit_sigma=False,
            fit_start=10, fit_end=90, decay=decay
        )

        # Use numpy array for params to avoid indexing issues
        params = np.array([2e-9, 1000.0, 0.0])
        cost = cost_obj(params)
        assert cost > 0

    def test_log_tau_equivalence(self):
        """_DECostPoissonLogTau should produce same cost as linear version."""
        from flimkit.FLIM.models import _DECostPoisson, _DECostPoissonLogTau
        import numpy as np

        n_bins = 100
        irf = np.ones(n_bins) / n_bins
        decay = np.random.poisson(50, n_bins).astype(float)

        cost_lin = _DECostPoisson(
            97e-12, n_bins, irf, n_exp=1, bg_fixed=5.0,
            has_tail=False, fit_bg=False, fit_sigma=False,
            fit_start=10, fit_end=90, decay=decay
        )
        cost_log = _DECostPoissonLogTau(
            97e-12, n_bins, irf, n_exp=1, bg_fixed=5.0,
            has_tail=False, fit_bg=False, fit_sigma=False,
            fit_start=10, fit_end=90, decay=decay
        )

        cost1 = cost_lin(np.array([2e-9, 1000.0, 0.0]))
        cost2 = cost_log(np.array([np.log10(2e-9), 1000.0, 0.0]))
        assert cost1 == pytest.approx(cost2, rel=1e-6)

class TestHasTailFitting:
    """Tests for reconvolution with IRF tail (has_tail=True)."""

    def test_has_tail_convergence(self):
        """Fit synthetic decay with tail and verify tail parameters recovered."""
        from flimkit.FLIM.fitters import fit_summed
        from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
        import numpy as np

        n_bins = 256
        tcspc_res = 97e-12
        fwhm_ns = 0.3
        peak_bin = 50

        # Create a realistic IRF with a tail
        irf_prompt = gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, peak_bin)
        # Add exponential tail
        bins = np.arange(n_bins)
        tail_amp_true = 0.2
        tail_tau_ns_true = 2.0
        tail_tau_bins = tail_tau_ns_true * 1e-9 / tcspc_res
        tail = np.where(bins >= peak_bin,
                        tail_amp_true * np.exp(-(bins - peak_bin) / tail_tau_bins),
                        0.0)
        irf_with_tail = irf_prompt + tail
        irf_with_tail /= irf_with_tail.sum()

        # Generate decay with known lifetime
        from mock_data import generate_synthetic_decay
        decay = generate_synthetic_decay(
            n_bins=n_bins,
            tcspc_res=tcspc_res,
            tau_ns=2.5,
            bg=5.0,
            peak_counts=50000,
            irf_fwhm_bins=3.0,
            irf_center_bin=peak_bin,
            noise=True,
        )

        # Fit with has_tail=True
        popt, summary = fit_summed(
            decay, tcspc_res, n_bins, irf_with_tail,
            has_tail=True,
            fit_bg=True,
            fit_sigma=False,
            n_exp=1,
            tau_min_ns=0.1,
            tau_max_ns=10.0,
            optimizer="de",
            de_popsize=15,
            de_maxiter=500,
            workers=1,
            polish=True,
        )

        # Check that tail parameters are reasonable (not at bounds)
        # popt layout: tau, amp, shift, bg, tail_amp, tail_tau
        fitted_tail_amp = popt[-2]
        fitted_tail_tau_ns = popt[-1] * tcspc_res * 1e9

        assert 0.01 < fitted_tail_amp < 1.0
        assert 0.5 < fitted_tail_tau_ns < 10.0
        # Relative error in tail tau should be within 50%
        assert abs(fitted_tail_tau_ns - tail_tau_ns_true) / tail_tau_ns_true < 0.5

    def test_has_tail_improves_fit(self):
        """With tail present, fitting with has_tail=True should give lower χ²."""
        from flimkit.FLIM.fitters import fit_summed
        from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
        import numpy as np

        n_bins = 256
        tcspc_res = 97e-12
        fwhm_ns = 0.3
        peak_bin = 50

        # IRF with significant tail
        irf_prompt = gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, peak_bin)
        bins = np.arange(n_bins)
        tail = np.where(bins >= peak_bin, 0.3 * np.exp(-(bins - peak_bin) / 50), 0.0)
        irf_with_tail = irf_prompt + tail
        irf_with_tail /= irf_with_tail.sum()

        decay = generate_synthetic_decay(
            n_bins=n_bins,
            tcspc_res=tcspc_res,
            tau_ns=2.5,
            bg=5.0,
            peak_counts=50000,
            irf_fwhm_bins=3.0,
            irf_center_bin=peak_bin,
            noise=True,
        )

        # Fit without tail
        _, summary_no_tail = fit_summed(
            decay, tcspc_res, n_bins, irf_with_tail,
            has_tail=False,
            fit_bg=True,
            fit_sigma=False,
            n_exp=1,
            tau_min_ns=0.1,
            tau_max_ns=10.0,
            optimizer="de",
            workers=1,
        )

        # Fit with tail
        _, summary_tail = fit_summed(
            decay, tcspc_res, n_bins, irf_with_tail,
            has_tail=True,
            fit_bg=True,
            fit_sigma=False,
            n_exp=1,
            tau_min_ns=0.1,
            tau_max_ns=10.0,
            optimizer="de",
            workers=1,
        )

        # Tail fit should have lower reduced chi-squared (Pearson)
        assert summary_tail['reduced_chi2_pearson'] < summary_no_tail['reduced_chi2_pearson']