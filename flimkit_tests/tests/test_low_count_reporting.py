import io
import contextlib
import numpy as np
import pytest
from flimkit.FLIM.fitters import fit_summed, COMPONENT_REL_LIMIT
from flimkit.FLIM.models import reconvolution_model
from flimkit.utils.misc import print_summary

RES = 50e-12
N = 256
FAST = dict(workers=1, de_popsize=10, de_maxiter=300, cost_function='poisson')

def _irf():
    t = np.arange(N)
    irf = np.exp(-0.5 * ((t - 20) / 1.5) ** 2)
    return irf / irf.sum()

def _decay(taus_ns, amps, photons, seed=0):
    p = np.r_[np.asarray(taus_ns) * 1e-9, amps, 0.0, 0.0]
    m = reconvolution_model(p, RES, N, _irf(), len(taus_ns), 0.0, False, True, False)
    return np.random.default_rng(seed).poisson(m / m.sum() * photons + 0.5).astype(float)

def _fit(decay, n_exp, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fit_summed(decay, RES, N, _irf(), False, True, False, n_exp, 0.1, 10.0, **{**FAST, **kw})

def test_fixed_lifetimes_stay_fixed_and_only_amplitudes_move():
    taus = [4.0, 1.5, 0.4]
    _, s = _fit(_decay(taus, [0.4, 0.35, 0.25], 3000), 3, fixed_taus_ns=taus)
    assert s['fixed_taus'] == True
    assert np.sort(s['taus_ns']) == pytest.approx(np.sort(taus), rel=1e-5)
    assert 'components_resolved' not in s
    assert np.sum(s['fractions']) == pytest.approx(1.0)

def test_fixed_lifetimes_need_one_per_component():
    with pytest.raises(ValueError, match='fixed_taus_ns needs 3'):
        _fit(_decay([4.0], [1.0], 1e4), 3, fixed_taus_ns=[4.0, 1.0])

def test_fixed_lifetimes_survive_the_bootstrap():
    taus = [3.0, 0.8]
    _, s = _fit(_decay(taus, [0.5, 0.5], 5000), 2, fixed_taus_ns=taus, bootstrap=4)
    assert s['n_bootstrap'] == 4
    assert np.all(np.asarray(s['taus_ns_boot_err']) < 1e-4)

def test_starved_three_exponential_is_flagged_and_leads_with_the_mean():
    _, s = _fit(_decay([4.0, 1.5, 0.4], [0.4, 0.35, 0.25], 1000, seed=2), 3)
    assert s['components_resolved'] == False
    assert s['uncertainty_warnings'][0].startswith('The 3 components are not resolved')
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        print_summary(s, 'gaussian', 3)
    text = out.getvalue()
    assert text.index('Not resolved at this photon count') < text.index('τ1 =')

def test_well_determined_two_exponential_is_not_flagged():
    _, s = _fit(_decay([4.0, 0.8], [0.5, 0.5], 2e6, seed=3), 2)
    assert s['components_resolved'] == True
    rel = np.asarray(s['taus_ns_err']) / np.asarray(s['taus_ns'])
    assert np.all(rel <= COMPONENT_REL_LIMIT)

def test_one_exponential_has_no_component_verdict():
    _, s = _fit(_decay([2.5], [1.0], 500), 1)
    assert 'components_resolved' not in s
