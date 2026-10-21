import numpy as np
import pytest
from flimkit.utils.display import compute_residuals, rebuild_summed_model
from flimkit.FLIM.fitters import fit_summed

RES = 97e-12
N = 128

def _irf():
    t = np.arange(N)
    irf = np.exp(-0.5 * ((t - 10) / 1.5) ** 2)
    return irf / irf.sum()

def _decay(seed=0):
    from flimkit.FLIM.models import reconvolution_model
    params = np.array([3e-9, 1.0e5, 0.0, 20.0])
    model = reconvolution_model(params, RES, N, _irf(), 1, 0.0, False, True, False)
    return np.random.default_rng(seed).poisson(model).astype(float)

def test_saved_residuals_are_used():
    decay = np.array([10.0, 20.0])
    assert list(compute_residuals(decay, None, np.array([1.0, 2.0]))) == [1.0, 2.0]

def test_residuals_from_model_match_fitter_formula():
    decay = np.array([10.0, 0.0, 4.0])
    model = np.array([9.0, 0.5, 4.0])
    r = compute_residuals(decay, model)
    assert r == pytest.approx([1 / 3, -0.5, 0.0])

def test_no_model_no_residuals():
    assert compute_residuals(np.ones(4), None) is None

def test_model_longer_than_decay_is_trimmed():
    assert len(compute_residuals(np.ones(4), np.ones(6))) == 4

@pytest.mark.parametrize('fit_sigma', [False, True])
def test_rebuilt_model_matches_fit(fit_sigma):
    decay = _decay()
    _, summ = fit_summed(decay, RES, N, _irf(), False, True, fit_sigma, 1, 0.5, 12.0,
                         optimizer='lm_multistart', n_restarts=1, workers=1, cost_function='chi2')
    gs = {k: v for k, v in summ.items() if k not in ('model', 'residuals')}
    rebuilt = rebuild_summed_model(gs, _irf(), RES, N)
    assert rebuilt == pytest.approx(summ['model'], rel=1e-9, abs=1e-9)
    assert compute_residuals(decay, rebuilt) == pytest.approx(summ['residuals'], rel=1e-9, abs=1e-9)

def test_rebuild_refuses_tvb_and_tail_models():
    assert rebuild_summed_model({'taus_ns': [1.0], 'amps': [1.0], 'tvb_scale': 0.3}, _irf(), RES, N) is None
    assert rebuild_summed_model({'taus_ns': [1.0], 'amps': [1.0], 'fit_model': 'tail'}, _irf(), RES, N) is None
