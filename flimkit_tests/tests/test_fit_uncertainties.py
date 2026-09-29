import contextlib
import io
import sys
from pathlib import Path
import numpy as np
import pytest

_root = str(Path(__file__).parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from flimkit.FLIM.fit_tools import fit_uncertainties, propagate_uncertainty, uncertainty_warnings
from flimkit.FLIM.fitters import fit_summed, fit_summed_tail, fit_summed_dist
from flimkit.synth import build_decay

RES_NS = 0.025
N_BINS = 800

def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)

def _expected(taus, amps=None, photons=2e5):
    expected, irf, _ = build_decay(taus, amps, n_bins=N_BINS, tcspc_res_ns=RES_NS,
                                   n_photons=photons, irf_center_ns=1.0)
    return expected, irf

def _fit(decay, irf, n_exp):
    return _quiet(fit_summed, decay, RES_NS * 1e-9, N_BINS, irf, False, True, False,
                  n_exp, 0.1, 20, de_popsize=10, de_maxiter=300, workers=1)

def test_constant_model_matches_poisson_error():
    level = 400.0
    idx = np.arange(100)
    unc = fit_uncertainties(lambda q: np.full(100, q[0]), [level], idx)
    assert unc['stderr'][0] == pytest.approx(np.sqrt(level / 100), rel=1e-3)
    assert unc['free'][0]

def test_parameter_at_bound_is_left_out():
    idx = np.arange(50)
    unc = fit_uncertainties(lambda q: np.full(50, q[0] + q[1]), [100.0, 0.0], idx,
                            lo=[0.0, 0.0], hi=[1e4, 10.0])
    assert np.isfinite(unc['stderr'][0])
    assert np.isnan(unc['stderr'][1])
    assert not unc['free'][1]

def test_propagation_of_a_sum():
    idx = np.arange(200)
    t = np.linspace(0, 1, 200)
    unc = fit_uncertainties(lambda q: q[0] + q[1] * t, [50.0, 20.0], idx)
    direct = np.sqrt(unc['cov'][0, 0] + unc['cov'][1, 1] + 2 * unc['cov'][0, 1])
    assert propagate_uncertainty(lambda q: q[0] + q[1], [50.0, 20.0], unc) == pytest.approx(direct, rel=1e-3)

def test_warnings_for_loose_and_correlated_lifetimes():
    notes = uncertainty_warnings([4.0, 1.0], [0.01, 0.3], [[1.0, 0.99], [0.99, 1.0]])
    assert any('τ2 is poorly determined' in n for n in notes)
    assert any('strongly correlated' in n for n in notes)
    assert uncertainty_warnings([4.0], [0.01], [[1.0]]) == []

def test_single_exponential_error_matches_scatter():
    expected, irf = _expected([2.5])
    rng = np.random.default_rng(3)
    fits, errs = [], []
    for _ in range(12):
        _, s = _fit(rng.poisson(expected).astype(float), irf, 1)
        fits.append(s['taus_ns'][0])
        errs.append(s['taus_ns_err'][0])
    ratio = np.mean(errs) / np.std(fits, ddof=1)
    assert 0.6 < ratio < 1.6

def test_summed_fit_reports_uncertainties():
    expected, irf = _expected([3.0, 0.8], [0.5, 0.5])
    decay = np.random.default_rng(5).poisson(expected).astype(float)
    _, s = _fit(decay, irf, 2)
    assert s['taus_ns_err'].shape == (2,)
    assert np.all(np.isfinite(s['taus_ns_err']))
    assert np.all(s['taus_ns_err'] < 0.1 * s['taus_ns'])
    assert s['fractions_err'].shape == (2,)
    assert np.isfinite(s['tau_mean_amp_ns_err'])
    assert np.isfinite(s['tau_mean_int_ns_err'])
    assert s['tau_corr'].shape == (2, 2)
    assert s['tau_corr'][0, 0] == pytest.approx(1.0)
    assert np.isfinite(s['irf_shift_bins_err'])
    assert s['uncertainty_warnings'] == []

def test_errors_follow_the_sorted_lifetimes():
    expected, irf = _expected([0.8, 3.0], [0.5, 0.5], photons=5e4)
    decay = np.random.default_rng(7).poisson(expected).astype(float)
    _, s = _fit(decay, irf, 2)
    assert s['taus_ns'][0] > s['taus_ns'][1]
    assert s['taus_ns_err'][0] / s['taus_ns'][0] < 0.1

def test_tail_fit_reports_uncertainties():
    expected, _ = _expected([2.5])
    decay = np.random.default_rng(9).poisson(expected).astype(float)
    _, s = _quiet(fit_summed_tail, decay, RES_NS * 1e-9, N_BINS, True, 1, 0.1, 20,
                  de_popsize=10, de_maxiter=300, workers=1)
    assert 0 < s['taus_ns_err'][0] < 0.05

def test_distribution_fit_reports_uncertainties():
    expected, irf = _expected([2.5])
    decay = np.random.default_rng(11).poisson(expected).astype(float)
    _, s = _quiet(fit_summed_dist, decay, RES_NS * 1e-9, N_BINS, irf, 1, 'gaussian', True, False,
                  0.1, 20, de_popsize=10, de_maxiter=200, workers=1)
    assert s['tau_centers_ns_err'].shape == (1,)
    assert np.isfinite(s['tau_centers_ns_err'][0])
    assert s['widths_ns_err'].shape == (1,)

def test_poorly_determined_fit_is_flagged():
    expected, irf = _expected([4.0, 1.5, 0.6], [0.4, 0.3, 0.3], photons=2e4)
    decay = np.random.default_rng(13).poisson(expected).astype(float)
    _, s = _fit(decay, irf, 3)
    assert s['uncertainty_warnings']
    assert np.nanmax(s['taus_ns_err'] / s['taus_ns']) > 0.1

def test_results_table_shows_errors_and_warnings():
    from flimkit.UI.gui import _UIBuilder
    summary = dict(taus_ns=np.array([4.0, 1.0]), taus_ns_err=np.array([0.01, 0.3]),
                   amps=np.array([1.0, 1.0]), fractions=np.array([0.5, 0.5]),
                   tau_mean_amp_ns=2.5, tau_mean_amp_ns_err=0.1,
                   uncertainty_warnings=['τ2 is poorly determined'])
    rows = _UIBuilder._extract_summary_rows(object(), summary)
    labels = [r[0] for r in rows]
    assert ('τ2 ± (1σ)', '0.3000', 'ns') in rows
    assert 'τ_mean (amp-weighted) ± (1σ)' in labels
    assert '⚠ τ2 is poorly determined' in labels

def test_terminal_summary_prints_errors(capsys):
    from flimkit.utils.misc import print_summary
    expected, irf = _expected([2.5])
    decay = np.random.default_rng(15).poisson(expected).astype(float)
    _, s = _fit(decay, irf, 1)
    print_summary(s, 'gaussian', 1)
    out = capsys.readouterr().out
    assert f"± {s['taus_ns_err'][0]:.4f}" in out
    assert '1σ standard errors' in out
