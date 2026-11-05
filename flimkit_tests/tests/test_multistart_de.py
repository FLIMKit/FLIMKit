import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from flimkit.FLIM import fitters
from flimkit.FLIM.fitters import (fit_summed, fit_summed_tail, resolve_n_starts, _de_multistart,
                                  DE_AUTO_STARTS_TRI_EXP, DE_NEAR_BEST_RTOL)
from flimkit.FLIM.models import reconvolution_model

RES = 97e-12
N = 128
FAST = dict(de_popsize=5, de_maxiter=20, cost_function='poisson')

def _irf():
    t = np.arange(N)
    irf = np.exp(-0.5 * ((t - 10) / 1.5) ** 2)
    return irf / irf.sum()

def _tri_decay(seed=0):
    params = np.array([0.3e-9, 1.2e-9, 4.0e-9, 4e4, 3e4, 2e4, 0.0, 5.0])
    model = reconvolution_model(params, RES, N, _irf(), 3, 0.0, False, True, False)
    return np.random.default_rng(seed).poisson(model).astype(float)

def _fit(decay, n_exp=3, **kw):
    return fit_summed(decay, RES, N, _irf(), False, True, False, n_exp, 0.1, 10.0, **{**FAST, **kw})

def _midpoint_de(calls):
    def fake(cost_fn, bounds, seed=None, **kw):
        calls.append(seed)
        x = np.array([(a + b) / 2.0 for a, b in bounds])
        return OptimizeResult(x=x, fun=float(cost_fn(x)), success=True)
    return fake

@pytest.mark.parametrize('n_exp, expected', [(1, 1), (2, 1), (3, DE_AUTO_STARTS_TRI_EXP)])
def test_automatic_default(n_exp, expected):
    assert DE_AUTO_STARTS_TRI_EXP == 12
    assert resolve_n_starts(None, n_exp) == expected

def test_explicit_n_starts_and_bad_value():
    assert resolve_n_starts(1, 3) == 1
    assert resolve_n_starts(5, 1) == 5
    with pytest.raises(ValueError):
        resolve_n_starts(0, 3)

class _Pol:
    def __init__(self, x, cost):
        self.x = x
        self.cost = cost

def _stub_run(monkeypatch, polish_costs, de_costs=None):
    seen = []
    def fake_de(cost_fn, bounds, seed=None, **kw):
        seen.append(seed)
        k = seed - 42
        fun = de_costs[k] if de_costs else 1.0
        return OptimizeResult(x=np.array([-9.0, float(k)]), fun=fun, success=True)
    def fake_ls(fn, x0, **kw):
        k = int(round(x0[1]))
        if polish_costs[k] is None:
            raise ValueError('polish failed')
        return _Pol(x0, polish_costs[k])
    monkeypatch.setattr(fitters, 'differential_evolution', fake_de)
    monkeypatch.setattr(fitters, 'least_squares', fake_ls)
    bounds = [(1e-10, 1e-8), (-1.0, 100.0)]
    popt, message, info = _de_multistart(None, bounds, 1, [1e-10, -1.0], [1e-8, 100.0], None,
                                         5, 10, 1, True, len(polish_costs), 42)
    return seen, popt, message, info

def test_lowest_polished_cost_wins(monkeypatch):
    costs = [5.0, 3.0, 3.0 * (1 + DE_NEAR_BEST_RTOL / 2), 4.0]
    seen, popt, message, info = _stub_run(monkeypatch, costs)
    assert seen == [42, 43, 44, 45]
    assert info['seed'] == 43
    assert info['start_costs'] == costs
    assert info['n_near_best'] == 2
    assert info['n_starts'] == 4
    assert int(round(popt[1])) == 1
    assert message == f'DE success=True, fun={1.0:.4e}; polished cost={3.0:.4e}'

def test_failed_polish_is_never_chosen(monkeypatch):
    _, _, _, info = _stub_run(monkeypatch, [None, 2.0, None])
    assert info['seed'] == 43
    assert info['start_costs'] == [None, 2.0, None]
    assert info['start_cost_kind'] == 'polished'

def test_de_cost_when_every_polish_fails(monkeypatch):
    _, _, message, info = _stub_run(monkeypatch, [None, None, None], de_costs=[3.0, 1.0, 2.0])
    assert info['seed'] == 43
    assert info['start_cost_kind'] == 'de'
    assert info['start_costs'] == [3.0, 1.0, 2.0]
    assert 'polished' not in message

def test_bootstrap_refits_use_one_start(monkeypatch):
    calls = []
    monkeypatch.setattr(fitters, 'differential_evolution', _midpoint_de(calls))
    _, s = _fit(_tri_decay(), workers=1, bootstrap=3)
    assert s['n_starts'] == 12
    assert len(calls) == 12 + 3
    assert calls[:12] == list(range(42, 54))
    assert calls[12:] == [42, 42, 42]

def test_tail_bootstrap_refits_use_one_start(monkeypatch):
    calls = []
    monkeypatch.setattr(fitters, 'differential_evolution', _midpoint_de(calls))
    _, s = fit_summed_tail(_tri_decay(), RES, N, True, 3, 0.1, 10.0, workers=1, bootstrap=3, **FAST)
    assert s['n_starts'] == 12
    assert len(calls) == 12 + 3
    assert calls[12:] == [42, 42, 42]

def test_one_or_two_exponentials_run_once(monkeypatch):
    calls = []
    monkeypatch.setattr(fitters, 'differential_evolution', _midpoint_de(calls))
    _fit(_tri_decay(), n_exp=2, workers=1)
    assert calls == [42]

def test_n_starts_one_is_the_first_start_of_many():
    p1, s1 = _fit(_tri_decay(), n_starts=1, workers=1)
    p3, s3 = _fit(_tri_decay(), n_starts=3, workers=1)
    assert s1['n_starts'] == 1 and s1['seed'] == 42 and s1['n_near_best'] == 1
    assert s1['start_costs'][0] == s3['start_costs'][0]
    assert s3['seed'] == 42 + int(np.argmin(s3['start_costs']))
    assert min(s3['start_costs']) <= s1['start_costs'][0]

def test_summary_fields_exist():
    _, s = _fit(_tri_decay(), n_starts=2, workers=1)
    for key in ('n_starts', 'start_costs', 'n_near_best', 'seed'):
        assert key in s
    assert len(s['start_costs']) == 2
    assert 1 <= s['n_near_best'] <= 2
    assert 'polished cost=' in s['optimizer_msg']

def test_same_seed_same_result_for_any_worker_count():
    p1, s1 = _fit(_tri_decay(), n_starts=2, workers=1)
    p2, s2 = _fit(_tri_decay(), n_starts=2, workers=-1)
    assert s1['start_costs'] == s2['start_costs']
    np.testing.assert_array_equal(p1, p2)
