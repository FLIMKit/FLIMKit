import numpy as np
import pytest
from flimkit.FLIM.fitters import fit_summed
from flimkit.FLIM.models import reconvolution_model, period_grid_bins
from flimkit.FLIM.fit_tools import fast_component_warnings
from flimkit.utils.display import rebuild_summed_model

RES = 0.05e-9
N_PERIOD = 250
FAST = dict(workers=1, de_popsize=10, de_maxiter=300, n_starts=1, cost_function='poisson')

def _irf(n=N_PERIOD):
    irf = np.exp(-0.5 * ((np.arange(n) - 20) / 1.5) ** 2)
    return irf / irf.sum()

def _folded(taus_ns, amps, n_win, photons=1e6, bg=2.0, seed=1):
    t = np.arange(10 * N_PERIOD) * RES
    irf = np.pad(_irf(), (0, 9 * N_PERIOD))
    single = sum(a * np.convolve(irf, np.exp(-t / (tau * 1e-9)))[:10 * N_PERIOD] for tau, a in zip(taus_ns, amps))
    folded = single.reshape(10, N_PERIOD).sum(axis=0)[:n_win]
    return np.random.default_rng(seed).poisson(folded / folded.sum() * photons + bg).astype(float)

def _fit(decay, n_win, n_exp, period):
    return fit_summed(decay, RES, n_win, _irf()[:n_win], False, True, False, n_exp, 0.1, 10.0,
                      rep_period_ns=period, **FAST)[1]

def test_periodic_model_matches_folded_pulse_train():
    params = np.array([4e-9, 1.0, 0.0, 0.0])
    m = reconvolution_model(params, RES, N_PERIOD, _irf(), 1, 0.0, False, True, False,
                            rep_period_s=N_PERIOD * RES)
    t = np.arange(10 * N_PERIOD) * RES
    single = np.convolve(np.pad(_irf(), (0, 9 * N_PERIOD)), np.exp(-t / 4e-9))[:10 * N_PERIOD]
    assert m == pytest.approx(single.reshape(10, N_PERIOD).sum(axis=0), rel=1e-6, abs=1e-9)

def test_window_shorter_than_period_needs_the_periodic_term():
    n_win = 200
    decay = _folded([4.0], [1.0], n_win)
    off = _fit(decay, n_win, 1, None)
    on = _fit(decay, n_win, 1, N_PERIOD * RES * 1e9)
    assert on['reduced_chi2_pearson'] < 1.5
    assert off['reduced_chi2_pearson'] > 3 * on['reduced_chi2_pearson']
    assert abs(on['taus_ns'][0] - 4.0) < abs(off['taus_ns'][0] - 4.0)
    assert on['rep_period_ns'] == pytest.approx(N_PERIOD * RES * 1e9)
    assert off['rep_period_ns'] is None

def test_window_equal_to_period_corrects_the_fractions():
    t = np.arange(10 * N_PERIOD) * RES
    irf = np.pad(_irf(), (0, 9 * N_PERIOD))
    single = sum(0.5 * np.convolve(irf, np.exp(-t / (tau * 1e-9)))[:10 * N_PERIOD] for tau in (4.0, 0.8))
    decay = single.reshape(10, N_PERIOD).sum(axis=0)
    decay = decay / decay.sum() * 1e6 + 2.0
    off = _fit(decay, N_PERIOD, 2, None)
    on = _fit(decay, N_PERIOD, 2, N_PERIOD * RES * 1e9)
    assert on['fractions'] == pytest.approx([0.5, 0.5], abs=2e-3)
    assert off['fractions'][0] > 0.5 + 3e-3

def test_period_shorter_than_window_is_refused():
    with pytest.raises(ValueError, match='shorter than the histogram'):
        period_grid_bins(5e-9, RES, N_PERIOD)

def test_rebuilt_model_keeps_the_period():
    decay = _folded([4.0], [1.0], 200)
    s = _fit(decay, 200, 1, N_PERIOD * RES * 1e9)
    gs = {k: v for k, v in s.items() if k not in ('model', 'residuals')}
    assert rebuild_summed_model(gs, _irf()[:200], RES, 200) == pytest.approx(s['model'], rel=1e-6, abs=1e-6)

def test_rise_chi2_is_reported_and_near_one_for_a_good_fit():
    s = _fit(_folded([4.0], [1.0], N_PERIOD), N_PERIOD, 1, N_PERIOD * RES * 1e9)
    assert 0.3 < s['rise_chi2_per_bin_pearson'] < 2.0

def test_rise_chi2_flags_a_wrong_irf():
    decay = _folded([4.0], [1.0], N_PERIOD)
    wide = np.exp(-0.5 * ((np.arange(N_PERIOD) - 20) / 4.0) ** 2)
    _, s = fit_summed(decay, RES, N_PERIOD, wide / wide.sum(), False, True, False, 1, 0.1, 10.0, **FAST)
    assert s['rise_chi2_per_bin_pearson'] > 5 * s['reduced_chi2_tail_pearson']

def test_small_fast_component_is_flagged():
    notes = fast_component_warnings([3.9, 0.2], [0.99, 0.01], 0.15)
    assert len(notes) == 1 and 'τ2' in notes[0] and 'IRF' in notes[0]

def test_the_fraction_tested_is_the_photons_not_the_amplitude():
    notes = fast_component_warnings([3.848, 0.246], [0.842, 0.158], 0.097)
    assert len(notes) == 1 and '1.2% of the photons' in notes[0]

@pytest.mark.parametrize('taus, fracs, fwhm', [([3.9, 0.2], [0.2, 0.8], 0.15), ([3.9, 1.5], [0.99, 0.01], 0.15),
                                               ([0.2], [1.0], 0.15)])
def test_large_slow_or_single_components_are_not_flagged(taus, fracs, fwhm):
    assert fast_component_warnings(taus, fracs, fwhm) == []

def test_fast_component_warning_survives_bootstrap(monkeypatch):
    from flimkit.FLIM import fitters
    monkeypatch.setattr(fitters, 'fast_component_warnings', lambda *a: ['fast'])
    s = _fit(_folded([4.0], [1.0], N_PERIOD), N_PERIOD, 1, None)
    assert 'fast' in s['uncertainty_warnings'] and s['fit_warnings'] == ['fast']
    _, s = fit_summed(_folded([4.0], [1.0], N_PERIOD), RES, N_PERIOD, _irf(), False, True, False, 1, 0.1, 10.0,
                      bootstrap=3, **FAST)
    assert 'fast' in s['uncertainty_warnings']

def _stack(decay, ny=2, nx=2):
    return np.tile(decay, (ny, nx, 1))

def _popt_1exp(tau_ns, amp=1.0):
    return np.array([tau_ns * 1e-9, amp, 0.0, 0.0])

def test_per_pixel_grid_scan_uses_the_period():
    n_win = 200
    stack = _stack(_folded([4.0], [1.0], n_win, photons=2e5))
    T = N_PERIOD * RES * 1e9
    from flimkit.FLIM.fitters import fit_per_pixel
    off = fit_per_pixel(stack, RES, n_win, _irf()[:n_win], False, True, False, _popt_1exp(4.0), 1,
                        tau_min_ns=0.5, tau_max_ns=10.0, use_gpu=False)
    on = fit_per_pixel(stack, RES, n_win, _irf()[:n_win], False, True, False, _popt_1exp(4.0), 1,
                       tau_min_ns=0.5, tau_max_ns=10.0, use_gpu=False, rep_period_ns=T)
    assert np.nanmean(on['chi2_r']) < 0.5 * np.nanmean(off['chi2_r'])

def test_per_pixel_fixed_tau_fractions_follow_the_period():
    stack = _stack(_folded([4.0, 0.8], [0.5, 0.5], N_PERIOD, photons=5e5))
    popt = np.array([4.0e-9, 0.8e-9, 1.0, 1.0, 0.0, 0.0])
    from flimkit.FLIM.fitters import fit_per_pixel
    kw = dict(use_gpu=False)
    off = fit_per_pixel(stack, RES, N_PERIOD, _irf(), False, True, False, popt, 2, **kw)
    on = fit_per_pixel(stack, RES, N_PERIOD, _irf(), False, True, False, popt, 2,
                       rep_period_ns=N_PERIOD * RES * 1e9, **kw)
    assert abs(np.nanmean(on['frac_1']) - 0.5) < abs(np.nanmean(off['frac_1']) - 0.5)

def test_free_tau_with_a_period_runs_in_scipy(capsys):
    stack = _stack(_folded([4.0, 0.8], [0.5, 0.5], 200, photons=5e5), 1, 1)
    popt = np.array([4.0e-9, 0.8e-9, 1.0, 1.0, 0.0, 0.0])
    from flimkit.FLIM.fitters import fit_per_pixel
    maps = fit_per_pixel(stack, RES, 200, _irf()[:200], False, True, False, popt, 2,
                         free_tau=True, use_gpu=False, rep_period_ns=N_PERIOD * RES * 1e9)
    assert 'runs in SciPy' in capsys.readouterr().out
    assert np.isfinite(maps['tau_1']).all()

def test_tail_fit_carries_the_period():
    from flimkit.FLIM.fitters import fit_summed_tail
    decay = _folded([4.0, 0.8], [0.5, 0.5], N_PERIOD)
    kw = dict(workers=1, de_popsize=10, de_maxiter=300, n_starts=1)
    _, off = fit_summed_tail(decay, RES, N_PERIOD, True, 2, 0.1, 10.0, **kw)
    _, on = fit_summed_tail(decay, RES, N_PERIOD, True, 2, 0.1, 10.0, rep_period_ns=N_PERIOD * RES * 1e9, **kw)
    assert on['rep_period_ns'] == pytest.approx(N_PERIOD * RES * 1e9)
    assert on['taus_ns'] == pytest.approx(off['taus_ns'], rel=0.02)
    assert on['fractions'][0] < off['fractions'][0]

def test_distribution_model_matches_folded_pulse_train():
    from flimkit.FLIM.models import dist_reconvolution_model, _dist_kernel
    params = np.array([4e-9, 0.2e-9, 1.0, 0.0, 0.0])
    m = dist_reconvolution_model(params, RES, 200, _irf()[:200], 1, 'gaussian', 0.0, True, False,
                                 rep_period_s=N_PERIOD * RES)
    t_n = 10 * N_PERIOD
    single = np.convolve(np.pad(_irf(), (0, 9 * N_PERIOD)),
                         _dist_kernel(RES, t_n, 4e-9, 0.2e-9, 1.0, 'gaussian'))[:t_n]
    assert m == pytest.approx(single.reshape(10, N_PERIOD).sum(axis=0)[:200], rel=1e-4, abs=1e-6)

def test_window_one_bin_past_the_period_is_allowed():
    assert period_grid_bins(12.817e-9, 97e-12, 133) == 133

@pytest.mark.parametrize('value, expected', [(None, None), ('', None), ('12.5', 12.5), (12.5, 12.5)])
def test_resolve_rep_period(value, expected):
    from flimkit.FLIM.fit_tools import resolve_rep_period_ns
    assert resolve_rep_period_ns(value) == expected

def test_resolve_rep_period_auto_reads_the_file():
    from types import SimpleNamespace
    from flimkit.FLIM.fit_tools import resolve_rep_period_ns
    assert resolve_rep_period_ns('auto', SimpleNamespace(period_ns=12.817)) == pytest.approx(12.817)
    assert resolve_rep_period_ns('AUTO', lambda: SimpleNamespace(period_ns=50.0)) == 50.0
    with pytest.raises(ValueError, match='does not record'):
        resolve_rep_period_ns('auto', SimpleNamespace(period_ns=0.0))
    with pytest.raises(ValueError):
        resolve_rep_period_ns('-1')

class _Backend:
    plugin_id = 'stub'

    def __init__(self, periodic):
        if periodic == True:
            self.free_tau_rep_period = True
        self.calls = []

    def batch_free_tau_fit(self, *args, **kwargs):
        self.calls.append(kwargs)
        return {'stub': True}

def _free_tau_inputs():
    stack = _stack(_folded([4.0, 0.8], [0.5, 0.5], 200, photons=2e5, bg=1.0), 2, 2)
    popt = np.array([4.0e-9, 0.8e-9, 1.0, 1.0, 0.0, 4.0])
    return stack, popt

def test_a_backend_that_models_the_period_gets_it_with_the_summed_background():
    from flimkit.FLIM.fitters import fit_per_pixel
    stack, popt = _free_tau_inputs()
    be = _Backend(True)
    out = fit_per_pixel(stack, RES, 200, _irf()[:200], False, True, False, popt, 2, free_tau=True,
                        use_gpu=True, gpu_backend=be, rep_period_ns=N_PERIOD * RES * 1e9)
    assert out == {'stub': True}
    assert be.calls[0]['rep_period_s'] == pytest.approx(N_PERIOD * RES)
    assert be.calls[0]['bg_per_pixel'] == pytest.approx(1.0)

def test_a_backend_that_does_not_model_the_period_is_skipped():
    from flimkit.FLIM.fitters import fit_per_pixel
    stack, popt = _free_tau_inputs()
    be = _Backend(False)
    out = fit_per_pixel(stack, RES, 200, _irf()[:200], False, True, False, popt, 2, free_tau=True,
                        use_gpu=True, gpu_backend=be, rep_period_ns=N_PERIOD * RES * 1e9)
    assert be.calls == [] and 'tau_1' in out

def test_free_tau_with_the_summed_background_recovers_the_lifetimes():
    from flimkit.FLIM.fitters import fit_per_pixel
    stack, popt = _free_tau_inputs()
    m = fit_per_pixel(stack, RES, 200, _irf()[:200], False, True, False, popt, 2, free_tau=True,
                      use_gpu=False, rep_period_ns=N_PERIOD * RES * 1e9)
    assert np.nanmedian(m['tau_2']) == pytest.approx(4.0, rel=0.03)
    assert np.nanmedian(m['tau_1']) == pytest.approx(0.8, rel=0.05)
