import ast
from pathlib import Path

import numpy as np
import pytest
from flimkit.FLIM.models import apply_pileup
from flimkit.FLIM.fit_tools import (coates_pileup_correction, build_fit_idx,
                                    bins_from_ns)
from flimkit.GPU._base import fit_window
from flimkit.FLIM.fitters import fit_per_pixel

RES = 0.097e-9
N = 133


def test_fit_window_validates_indices():
    identity = np.arange(8)
    assert fit_window(identity, 8) is None
    np.testing.assert_array_equal(fit_window(identity[::-1], 8), identity[::-1])
    for invalid in (np.array([], dtype=int), np.array([0, 1, 1]),
                    np.array([-1, 0]), np.array([0, 8])):
        with pytest.raises(ValueError):
            fit_window(invalid, 8)


def test_distribution_dof_counts_all_fitted_terms():
    from flimkit.FLIM.fit_tools import distribution_dof

    assert distribution_dof(100, 1, False) == 96
    assert distribution_dof(100, 1, True) == 95
    assert distribution_dof(100, 2, False) == 93

def _decay(tau_ns=4.1, total=3e5):
    t = np.arange(N) * RES * 1e9
    d = np.exp(-t / tau_ns)
    return d / d.sum() * total

class TestApplyPileup:
    def test_exact_inverse_of_coates(self):
        true = _decay()
        n_sync = 1_000_000
        back = coates_pileup_correction(apply_pileup(true, n_sync), n_sync)
        assert np.allclose(back, true, rtol=1e-9, atol=1e-6)

    def test_loses_photons_and_shortens_decay(self):
        true = _decay(total=3e5)
        n_sync = 1_000_000
        piled = apply_pileup(true, n_sync)
        # pile-up eats late photons, so counts drop and the tail is suppressed hardest
        assert piled.sum() < true.sum()
        assert (piled[-20:].sum() / true[-20:].sum()) < (piled[:20].sum() / true[:20].sum())

    def test_negligible_in_single_photon_regime(self):
        true = _decay(total=1e4)
        piled = apply_pileup(true, 1_000_000)
        assert abs(piled.sum() / true.sum() - 1.0) < 0.01

    def test_needs_full_length_model(self):
        # bin i depends on the cumulative rate before it, so slicing first is wrong
        true = _decay()
        n_sync = 1_000_000
        w = np.arange(20, N)
        assert not np.allclose(apply_pileup(true, n_sync)[w],
                               apply_pileup(true[w], n_sync))

class TestCoatesGuards:
    def test_rejects_photon_count_as_n_sync(self):
        d = _decay(total=3e5)
        with pytest.raises(ValueError, match='excitation-pulse count'):
            coates_pileup_correction(d, int(d.sum()))

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            coates_pileup_correction(_decay(), 0)

class TestBuildFitIdx:
    def test_contiguous_window(self):
        assert np.array_equal(build_fit_idx(5, 100, N), np.arange(5, 100))

    def test_excludes_band(self):
        idx = build_fit_idx(5, 100, N, [(80, 90)])
        assert not (set(range(80, 90)) & set(idx.tolist()))
        assert len(idx) == 95 - 10

    def test_multiple_bands(self):
        idx = build_fit_idx(0, N, N, [(10, 20), (80, 90)])
        assert len(idx) == N - 20

    def test_clamps_to_bins(self):
        assert build_fit_idx(-5, 999, N)[0] == 0
        assert build_fit_idx(-5, 999, N)[-1] == N - 1

    def test_empty_window_raises(self):
        with pytest.raises(ValueError, match='empty'):
            build_fit_idx(10, 20, N, [(0, N)])

    def test_bins_from_ns(self):
        assert bins_from_ns(0.0, RES) == 0
        assert bins_from_ns(RES * 1e9 * 10, RES) == 10

class TestExclusionBandInFit:
    # an unmodelled reflection is a fixed fraction of signal, so its chi2
    # contribution grows linearly with photon count; excluding it flattens that
    def _synth(self, total, seed):
        t = np.arange(N) * RES * 1e9
        irf = np.exp(-0.5 * ((t - 1.0) / 0.15) ** 2); irf /= irf.sum()
        pure = np.real(np.fft.ifft(np.fft.fft(np.exp(-t / 4.1)) * np.fft.fft(irf)))
        pure /= pure.sum()
        refl = np.exp(-0.5 * ((t - 8.0) / 0.2) ** 2); refl /= refl.sum()
        clean = pure * total * 0.98 + refl * total * 0.02
        return np.random.default_rng(seed).poisson(clean).astype(float), irf

    def _fit(self, decay, irf, exclude=None):
        import io, contextlib
        from flimkit.FLIM.fitters import fit_summed
        with contextlib.redirect_stdout(io.StringIO()):
            _, summ = fit_summed(decay, RES, N, irf, False, True, False, 1, 0.5, 12.0,
                                 optimizer='lm_multistart', n_restarts=6,
                                 fit_start_ns=0.5, fit_end_ns=12.0, exclude_ns=exclude)
        return float(summ['taus_ns'][0]), summ['reduced_chi2']

    def test_reflection_inflates_chi2_with_photon_count(self):
        lo, irf = self._synth(2e4, 1)
        hi, _ = self._synth(1e6, 2)
        assert self._fit(hi, irf)[1] > 5.0 * self._fit(lo, irf)[1]

    def test_excluding_reflection_flattens_chi2(self):
        band = [(6.5, 9.5)]
        lo, irf = self._synth(2e4, 1)
        hi, _ = self._synth(1e6, 2)
        c_lo = self._fit(lo, irf, band)[1]
        c_hi = self._fit(hi, irf, band)[1]
        assert c_lo < 2.5 and c_hi < 2.5
        assert c_hi < 3.0 * c_lo

    def test_excluding_reflection_recovers_true_tau(self):
        decay, irf = self._synth(1e6, 2)
        tau_in = self._fit(decay, irf)[0]
        tau_ex = self._fit(decay, irf, [(6.5, 9.5)])[0]
        assert abs(tau_ex - 4.1) < 0.1
        assert abs(tau_ex - 4.1) < abs(tau_in - 4.1)

    def test_exclusion_reduces_dof(self):
        import io, contextlib
        from flimkit.FLIM.fitters import fit_summed
        decay, irf = self._synth(2e5, 3)
        out = []
        for ex in (None, [(7.2, 8.8)]):
            with contextlib.redirect_stdout(io.StringIO()):
                _, s = fit_summed(decay, RES, N, irf, False, True, False, 1, 0.5, 12.0,
                                  optimizer='lm_multistart', n_restarts=1,
                                  fit_start_ns=0.5, fit_end_ns=12.0, exclude_ns=ex)
            out.append(s['dof'])
        assert out[1] < out[0]


def test_interactive_distribution_calls_propagate_fit_window():
    import flimkit.interactive as interactive

    tree = ast.parse(Path(interactive.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    summed_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == 'fit_summed_dist'
    ]
    pixel_calls = [
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == 'fit_per_pixel_dist'
    ]

    assert len(summed_calls) == 2
    assert len(pixel_calls) == 2
    for node in summed_calls:
        keywords = {keyword.arg for keyword in node.keywords}
        assert {'fit_start_ns', 'fit_end_ns', 'exclude_ns'} <= keywords
    for node in pixel_calls:
        keywords = {keyword.arg for keyword in node.keywords}
        assert 'fit_idx' in keywords


def _piled_stack(tau_ns=3.0, n_sync_px=6000, total=4000, side=2):
    axis = np.arange(N) * RES
    irf = np.exp(-0.5 * ((np.arange(N) - 12) / 2.0) ** 2)
    irf = irf / irf.sum()
    shape = np.convolve(np.exp(-axis / (tau_ns * 1e-9)), irf)[:N]
    shape = shape / shape.sum() * total
    piled = apply_pileup(shape, n_sync_px)
    stack = np.repeat(np.repeat(piled[None, None, :], side, axis=0), side, axis=1)
    return np.ascontiguousarray(stack, dtype=float), irf


class TestPerPixelPileupReachesTheFreeTauFit:

    def _fit(self, correct, n_sync_px=6000, side=2):
        stack, irf = _piled_stack(n_sync_px=n_sync_px, side=side)
        popt = np.array([3e-9, 1e-9, 0.7, 0.3, 0.0])
        return fit_per_pixel(
            stack, RES, N, irf, has_tail=False, fit_bg=True, fit_sigma=False,
            global_popt=popt, n_exp=2, min_photons=50, free_tau=True,
            use_gpu=False, correct_pileup=correct,
            n_sync=n_sync_px * side * side)

    def test_the_correction_changes_the_answer(self):
        off = self._fit(False)
        on = self._fit(True)
        assert not np.allclose(off['tau_mean_amp'], on['tau_mean_amp'],
                               rtol=1e-9, atol=1e-12)

    def test_the_correction_moves_the_lifetime_towards_the_truth(self):
        off = self._fit(False)
        on = self._fit(True)
        truth = 3.0
        assert (abs(np.nanmedian(on['tau_mean_amp']) - truth)
                < abs(np.nanmedian(off['tau_mean_amp']) - truth))

    def test_a_negligible_pileup_rate_leaves_the_fit_alone(self):
        off = self._fit(False, n_sync_px=4_000_000)
        on = self._fit(True, n_sync_px=4_000_000)
        np.testing.assert_allclose(off['tau_mean_amp'], on['tau_mean_amp'],
                                   rtol=1e-3)


class TestPileupInTheModel:

    def _fit(self, n_sync_px=6000, side=2, **kwargs):
        stack, irf = _piled_stack(n_sync_px=n_sync_px, side=side)
        popt = np.array([3e-9, 1e-9, 0.7, 0.3, 0.0])
        return fit_per_pixel(
            stack, RES, N, irf, has_tail=False, fit_bg=True, fit_sigma=False,
            global_popt=popt, n_exp=2, min_photons=50, free_tau=True,
            use_gpu=False, n_sync=n_sync_px * side * side, **kwargs)

    def test_the_model_route_changes_the_answer(self):
        off = self._fit()
        on = self._fit(pileup_in_model=True)
        assert not np.allclose(off['tau_mean_amp'], on['tau_mean_amp'],
                               rtol=1e-9, atol=1e-12)

    def test_the_model_route_moves_the_lifetime_towards_the_truth(self):
        off = self._fit()
        on = self._fit(pileup_in_model=True)
        truth = 3.0
        assert (abs(np.nanmedian(on['tau_mean_amp']) - truth)
                < abs(np.nanmedian(off['tau_mean_amp']) - truth))

    def test_both_routes_at_once_is_refused(self):
        with pytest.raises(ValueError, match='pick one pile-up route'):
            self._fit(correct_pileup=True, pileup_in_model=True)

    def test_the_model_route_is_refused_on_a_fixed_tau_fit(self):
        stack, irf = _piled_stack(side=2)
        popt = np.array([3e-9, 1e-9, 0.7, 0.3, 0.0])
        with pytest.raises(ValueError, match='needs a free-tau reconvolution fit'):
            fit_per_pixel(stack, RES, N, irf, has_tail=False, fit_bg=True,
                          fit_sigma=False, global_popt=popt, n_exp=2,
                          min_photons=50, free_tau=False, use_gpu=False,
                          n_sync=6000 * 4, pileup_in_model=True)
