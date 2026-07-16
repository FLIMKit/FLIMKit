import numpy as np
import pytest
from flimkit.FLIM.models import apply_pileup
from flimkit.FLIM.fit_tools import (coates_pileup_correction, build_fit_idx,
                                    bins_from_ns)

RES = 0.097e-9
N = 133

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
