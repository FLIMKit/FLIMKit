import pytest
import numpy as np
from pathlib import Path
import sys

_tests_dir = str(Path(__file__).parent.parent)
_project_root = str(Path(__file__).parent.parent.parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mock_data import (
    generate_synthetic_decay,
    generate_synthetic_biexp_decay,
    MOCK_TCSPC_RES,
    MOCK_IRF_CENTER,
    MOCK_IRF_FWHM_BINS,
)


def _irf(n_bins, tcspc_res, fwhm_ns, center_bin):
    from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
    return gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, center_bin)


class TestTailBasis:

    def test_shape_and_gating(self):
        from flimkit.FLIM.models import tail_basis
        n_bins = 128
        t0 = 20 * MOCK_TCSPC_RES
        b = tail_basis(MOCK_TCSPC_RES, n_bins, [2e-9], t0)
        assert b.shape == (1, n_bins)
        assert np.all(b[0, :20] == 0.0)
        assert b[0, 20] == pytest.approx(1.0)

    def test_decays_as_exponential(self):
        from flimkit.FLIM.models import tail_basis
        tau = 2e-9
        n_bins = 128
        b = tail_basis(MOCK_TCSPC_RES, n_bins, [tau], 0.0)[0]
        t = np.arange(n_bins) * MOCK_TCSPC_RES
        np.testing.assert_allclose(b, np.exp(-t / tau), rtol=1e-10)

    def test_one_lifetime_per_row(self):
        from flimkit.FLIM.models import tail_basis
        b = tail_basis(MOCK_TCSPC_RES, 64, [1e-9, 4e-9], 0.0)
        assert b.shape == (2, 64)
        assert b[1, -1] > b[0, -1]


class TestTailModel:

    def test_amplitude_is_linear(self):
        from flimkit.FLIM.models import tail_model
        p1 = [2e-9, 1.0]
        p2 = [2e-9, 3.0]
        m1 = tail_model(p1, MOCK_TCSPC_RES, 128, 1, 0.0, False)
        m2 = tail_model(p2, MOCK_TCSPC_RES, 128, 1, 0.0, False)
        np.testing.assert_allclose(m2, 3.0 * m1, rtol=1e-10)

    def test_background_is_added(self):
        from flimkit.FLIM.models import tail_model
        m_nobg = tail_model([2e-9, 100.0], MOCK_TCSPC_RES, 128, 1, 0.0, False)
        m_bg = tail_model([2e-9, 100.0, 7.0], MOCK_TCSPC_RES, 128, 1, 0.0, True)
        np.testing.assert_allclose(m_bg, m_nobg + 7.0, rtol=1e-10)

    def test_t0_shifts_the_onset(self):
        from flimkit.FLIM.models import tail_model
        t0 = 15 * MOCK_TCSPC_RES
        m = tail_model([2e-9, 100.0, t0], MOCK_TCSPC_RES, 128, 1, 0.0, False,
                       fit_t0=True)
        assert np.all(m[:15] == 0.0)
        assert m[15] == pytest.approx(100.0)

    def test_amplitude_is_defined_at_t0(self):
        from flimkit.FLIM.models import tail_model
        t0 = 30 * MOCK_TCSPC_RES
        m = tail_model([2e-9, 250.0, t0], MOCK_TCSPC_RES, 128, 1, 0.0, False,
                       fit_t0=True)
        assert m[30] == pytest.approx(250.0)

    def test_no_irf_dependence(self):
        from flimkit.FLIM.models import tail_model
        import inspect
        assert 'irf_prompt' not in inspect.signature(tail_model).parameters


class TestUnpackTailParams:

    def test_all_fixed(self):
        from flimkit.FLIM.models import unpack_tail_params
        taus, amps, t0, bg, tvb = unpack_tail_params(
            [1e-9, 3e-9, 10.0, 20.0], 2, False, False, False,
            t0_fixed=5e-10, bg_fixed=2.0)
        np.testing.assert_allclose(taus, [1e-9, 3e-9])
        np.testing.assert_allclose(amps, [10.0, 20.0])
        assert t0 == pytest.approx(5e-10)
        assert bg == pytest.approx(2.0)
        assert tvb == 0.0

    def test_all_free_in_order(self):
        from flimkit.FLIM.models import unpack_tail_params
        taus, amps, t0, bg, tvb = unpack_tail_params(
            [2e-9, 50.0, 1e-9, 3.0, 9.0], 1, True, True, True)
        np.testing.assert_allclose(taus, [2e-9])
        np.testing.assert_allclose(amps, [50.0])
        assert t0 == pytest.approx(1e-9)
        assert bg == pytest.approx(3.0)
        assert tvb == pytest.approx(9.0)


class TestFindTailFitStart:

    def test_starts_past_the_peak(self):
        from flimkit.FLIM.fit_tools import find_tail_fit_start
        decay = generate_synthetic_decay(n_bins=128, tau_ns=2.0, noise=False)
        peak = int(np.argmax(decay))
        start = find_tail_fit_start(decay, peak, 128)
        assert start > peak
        assert decay[start] < decay[peak]

    def test_flat_decay_does_not_hang(self):
        from flimkit.FLIM.fit_tools import find_tail_fit_start
        decay = np.zeros(64)
        assert find_tail_fit_start(decay, 0, 64) == 1


class TestTailFitRecovery:

    def test_monoexp_recovers_tau(self):
        from flimkit.FLIM.fitters import fit_summed_tail
        decay = generate_synthetic_decay(
            n_bins=256, tau_ns=2.0, bg=5.0, peak_counts=50_000.0, noise=True)
        _popt, summary = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=2,
            cost_function='poisson')
        assert summary['taus_ns'][0] == pytest.approx(2.0, rel=0.10)
        assert summary['fit_model'] == 'tail'

    def test_biexp_recovers_both_taus(self):
        from flimkit.FLIM.fitters import fit_summed_tail
        decay = generate_synthetic_biexp_decay(
            n_bins=256, tau1_ns=0.5, tau2_ns=3.0,
            a1=0.6, a2=0.4, bg=5.0, peak_counts=200_000.0, noise=True)
        _popt, summary = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 2, 0.1, 10.0,
            optimizer='de', de_popsize=20, de_maxiter=400,
            workers=1, cost_function='poisson')
        taus = np.sort(summary['taus_ns'])
        assert taus[1] == pytest.approx(3.0, rel=0.15)
        assert taus[0] == pytest.approx(0.5, rel=0.40)

    def test_window_starts_after_the_peak(self):
        from flimkit.FLIM.fitters import fit_summed_tail
        decay = generate_synthetic_decay(
            n_bins=256, tau_ns=2.0, peak_counts=50_000.0, noise=False)
        _popt, summary = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=0)
        fit_start = summary['fit_window_bins'][0]
        assert fit_start > int(np.argmax(decay))

    def test_free_t0_still_recovers_tau(self):
        from flimkit.FLIM.fitters import fit_summed_tail
        decay = generate_synthetic_decay(
            n_bins=256, tau_ns=2.0, bg=5.0, peak_counts=50_000.0, noise=True)
        _popt, summary = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 1, 0.2, 10.0,
            fit_t0=True,
            optimizer='lm_multistart', n_restarts=2)
        assert summary['taus_ns'][0] == pytest.approx(2.0, rel=0.12)

    def test_agrees_with_reconvolution(self):
        from flimkit.FLIM.fitters import fit_summed, fit_summed_tail
        decay = generate_synthetic_decay(
            n_bins=256, tau_ns=2.0, bg=5.0, peak_counts=100_000.0, noise=True)
        irf = _irf(256, MOCK_TCSPC_RES, MOCK_IRF_FWHM_BINS * MOCK_TCSPC_RES * 1e9,
                   MOCK_IRF_CENTER)
        _p_r, s_recon = fit_summed(
            decay, MOCK_TCSPC_RES, 256, irf,
            False, True, False, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=2)
        _p_t, s_tail = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=2)
        assert s_tail['taus_ns'][0] == pytest.approx(s_recon['taus_ns'][0], rel=0.10)


class TestTailSummaryFields:

    def test_intensities_follow_leica_definition(self):
        from flimkit.FLIM.fitters import fit_summed_tail
        decay = generate_synthetic_biexp_decay(
            n_bins=256, peak_counts=100_000.0, noise=True)
        _popt, s = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 2, 0.1, 10.0,
            optimizer='lm_multistart', n_restarts=1)
        expected = s['amps'] * (s['taus_ns'] * 1e-9) / MOCK_TCSPC_RES
        np.testing.assert_allclose(s['intensities'], expected, rtol=1e-9)
        assert s['i_sum'] == pytest.approx(float(s['intensities'].sum()))
        assert s['a_sum'] == pytest.approx(float(s['amps'].sum()))
        assert s['intensity_fractions'].sum() == pytest.approx(1.0)

    def test_mean_lifetimes_match_leica_weighting(self):
        from flimkit.FLIM.fitters import fit_summed_tail
        decay = generate_synthetic_biexp_decay(
            n_bins=256, peak_counts=100_000.0, noise=True)
        _popt, s = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 2, 0.1, 10.0,
            optimizer='lm_multistart', n_restarts=1)
        taus = s['taus_ns']
        amps = s['amps']
        tau_amp = float(np.dot(amps, taus) / amps.sum())
        tau_int = float(np.dot(s['intensities'], taus) / s['intensities'].sum())
        assert s['tau_mean_amp_ns'] == pytest.approx(tau_amp, rel=1e-6)
        assert s['tau_mean_int_ns'] == pytest.approx(tau_int, rel=1e-6)

    def test_no_irf_keys_reported(self):
        from flimkit.FLIM.fitters import fit_summed_tail
        decay = generate_synthetic_decay(n_bins=256, peak_counts=20_000.0)
        _popt, s = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=0)
        assert 'irf_fwhm_eff_ns' not in s
        assert 'irf_shift_bins' not in s
        assert 't0_ns' in s

    def test_print_summary_runs(self, capsys):
        from flimkit.FLIM.fitters import fit_summed_tail
        from flimkit.utils.misc import print_summary
        decay = generate_synthetic_decay(n_bins=256, peak_counts=20_000.0)
        _popt, s = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=0)
        print_summary(s, 'unused', 1)
        out = capsys.readouterr().out
        assert 'tail' in out
        assert 't0 (lifetime offset)' in out


class TestTailPerPixel:

    def _stack(self, n_bins=256, ny=4, nx=4, tau_ns=2.0):
        decay = generate_synthetic_decay(
            n_bins=n_bins, tau_ns=tau_ns, bg=2.0,
            peak_counts=5_000.0, noise=False)
        return np.repeat(np.repeat(decay[None, None, :], ny, 0), nx, 1).copy()

    def test_fixed_tau_map_recovers_tau(self):
        from flimkit.FLIM.fitters import fit_summed_tail, fit_per_pixel
        stack = self._stack()
        decay = stack.sum(axis=(0, 1))
        popt, s = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=1)
        maps = fit_per_pixel(
            stack, MOCK_TCSPC_RES, 256, None,
            False, True, False, popt, 1,
            min_photons=10,
            fit_idx=s['fit_idx'],
            use_gpu=False,
            fit_model='tail', t0_fixed=s['t0_ns'] * 1e-9)
        tau_map = maps['tau_mean_int']
        assert np.isfinite(tau_map).all()
        assert np.nanmean(tau_map) == pytest.approx(2.0, rel=0.15)

    def test_biexp_fixed_tau_projection(self):
        from flimkit.FLIM.fitters import fit_summed_tail, fit_per_pixel
        decay = generate_synthetic_biexp_decay(
            n_bins=256, peak_counts=20_000.0, noise=False)
        stack = np.repeat(np.repeat(decay[None, None, :], 3, 0), 3, 1).copy()
        popt, s = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 2, 0.1, 10.0,
            optimizer='lm_multistart', n_restarts=1)
        maps = fit_per_pixel(
            stack, MOCK_TCSPC_RES, 256, None,
            False, True, False, popt, 2,
            min_photons=10,
            fit_idx=s['fit_idx'],
            use_gpu=False,
            fit_model='tail', t0_fixed=s['t0_ns'] * 1e-9)
        assert np.isfinite(maps['tau_mean_amp']).all()
        assert np.isfinite(maps['chi2_r']).all()

    def test_defaults_window_to_bins_past_t0(self):
        from flimkit.FLIM.fitters import fit_summed_tail, fit_per_pixel
        stack = self._stack()
        decay = stack.sum(axis=(0, 1))
        popt, s = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=1)
        maps = fit_per_pixel(
            stack, MOCK_TCSPC_RES, 256, None,
            False, True, False, popt, 1,
            min_photons=10,
            use_gpu=False,
            fit_model='tail', t0_fixed=s['t0_ns'] * 1e-9)
        assert np.nanmean(maps['tau_mean_int']) == pytest.approx(2.0, rel=0.20)

    def test_free_tau_path_runs(self):
        from flimkit.FLIM.fitters import fit_summed_tail, fit_per_pixel
        decay = generate_synthetic_biexp_decay(
            n_bins=256, peak_counts=20_000.0, noise=False)
        stack = np.repeat(np.repeat(decay[None, None, :], 2, 0), 2, 1).copy()
        popt, s = fit_summed_tail(
            decay, MOCK_TCSPC_RES, 256,
            True, 2, 0.1, 10.0,
            optimizer='lm_multistart', n_restarts=1)
        maps = fit_per_pixel(
            stack, MOCK_TCSPC_RES, 256, None,
            False, True, False, popt, 2,
            min_photons=10,
            fit_idx=s['fit_idx'],
            free_tau=True, use_gpu=False,
            fit_model='tail', t0_fixed=s['t0_ns'] * 1e-9)
        assert np.isfinite(maps['tau_mean_amp']).all()

    def test_reconv_path_is_unchanged(self):
        from flimkit.FLIM.fitters import fit_summed, fit_per_pixel
        stack = self._stack()
        decay = stack.sum(axis=(0, 1))
        irf = _irf(256, MOCK_TCSPC_RES, MOCK_IRF_FWHM_BINS * MOCK_TCSPC_RES * 1e9,
                   MOCK_IRF_CENTER)
        popt, _s = fit_summed(
            decay, MOCK_TCSPC_RES, 256, irf,
            False, True, False, 1, 0.2, 10.0,
            optimizer='lm_multistart', n_restarts=1)
        maps = fit_per_pixel(
            stack, MOCK_TCSPC_RES, 256, irf,
            False, True, False, popt, 1,
            min_photons=10, use_gpu=False)
        assert np.nanmean(maps['tau_mean_int']) == pytest.approx(2.0, rel=0.15)
