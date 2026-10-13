import numpy as np
import pytest
from flimkit.FLIM.fitters import fit_per_pixel, _basis_rows
from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
from flimkit.FLIM.poisson_fit import poisson_amplitudes, fitFixedTauPoisson
from flimkit_tests.mock_data import MOCK_IRF_CENTER

pytest.importorskip('numba')

N_BINS = 512
TCSPC = 97e-12
TAUS = np.array([0.4e-9, 1.8e-9, 5.0e-9])

@pytest.fixture
def irf():
    return gaussian_irf_from_fwhm(N_BINS, TCSPC, 0.3, MOCK_IRF_CENTER)

def _basis(irf):
    t_axis = np.arange(N_BINS, dtype=float) * TCSPC
    return _basis_rows(TAUS, t_axis, TCSPC, N_BINS, False, irf_fft=np.fft.fft(irf)).T

def _popt():
    return np.array([*TAUS, 1000.0, 1000.0, 1000.0, 0.0])

def _stack(irf, photons, seed=0, shape=(8, 8)):
    A = _basis(irf)
    rng = np.random.default_rng(seed)
    amps = np.array([0.5, 0.3, 0.2]) / A.sum(axis=0)
    mean = A @ (amps * photons)
    return rng.poisson(np.broadcast_to(mean, shape + (N_BINS,))).astype(np.float64)

def _fit(irf, stack, **kw):
    return fit_per_pixel(stack, TCSPC, N_BINS, irf, has_tail=False, fit_bg=False,
                         fit_sigma=False, global_popt=_popt(), n_exp=3,
                         min_photons=5, use_gpu=False, **kw)

def test_exact_data_returns_exact_amplitudes(irf):
    A = _basis(irf)
    true = np.array([300.0, 120.0, 60.0])
    counts = (A @ true)[None, :]
    amps, bg = poisson_amplitudes(counts, A, offset=True, iters=5000, tol=1e-12)
    assert np.allclose(amps[0], true, rtol=0.02)
    assert bg[0] < 1e-3

def test_total_counts_are_preserved(irf):
    A = _basis(irf)
    stack = _stack(irf, 40.0, shape=(1, 16))[0]
    amps, bg = poisson_amplitudes(stack, A, offset=True)
    fitted = amps @ A.T + bg[:, None]
    assert np.allclose(fitted.sum(axis=1), stack.sum(axis=1), rtol=1e-3)

def test_default_estimator_is_poisson(irf):
    stack = _stack(irf, 60.0)
    default = _fit(irf, stack)
    explicit = _fit(irf, stack, estimator='poisson')
    for key in ('alpha_1', 'alpha_2', 'alpha_3', 'chi2_r'):
        assert np.allclose(default[key], explicit[key], equal_nan=True)

def test_nnls_remains_available(irf):
    stack = _stack(irf, 60.0)
    maps = _fit(irf, stack, estimator='nnls')
    assert np.isfinite(maps['alpha_1']).any()

def test_unknown_estimator_raises(irf):
    with pytest.raises(ValueError, match='estimator'):
        _fit(irf, _stack(irf, 60.0), estimator='lasx')

def test_modelled_background_keeps_the_projection(irf, capsys):
    maps = _fit(irf, _stack(irf, 60.0), bg_in_model=True)
    assert np.isfinite(maps['alpha_1']).any()
    assert 'estimator nnls' in capsys.readouterr().out

def test_fit_window_is_respected(irf):
    stack = _stack(irf, 80.0)
    window = np.arange(40, N_BINS - 40)
    maps = _fit(irf, stack, fit_idx=window)
    assert np.isfinite(maps['alpha_1']).all()

def test_low_counts_lose_no_photons_and_beat_the_projection(irf):
    stack = _stack(irf, 15.0, seed=3, shape=(32, 32))
    A = _basis(irf)
    truth = np.array([0.5, 0.3, 0.2])
    errs = {}
    totals = {}
    for estimator in ('poisson', 'nnls'):
        maps = _fit(irf, stack, estimator=estimator)
        alpha = np.stack([maps[f'alpha_{i + 1}'].ravel() for i in range(3)], axis=1)
        ok = np.isfinite(alpha).all(axis=1) & (alpha.sum(axis=1) > 0)
        counts = alpha[ok] * A.sum(axis=0)
        frac = counts / counts.sum(axis=1, keepdims=True)
        errs[estimator] = np.abs(frac - truth).mean()
        totals[estimator] = np.median(counts.sum(axis=1) / stack.reshape(-1, N_BINS)[ok].sum(axis=1))
    assert errs['poisson'] <= errs['nnls']
    assert totals['poisson'] <= 1.01

class _Recorder:
    estimators = ('nnls', 'poisson')

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def batch_fixed_tau(self, stack, A, taus_fixed, min_photons, correct_pileup,
                        n_sync_px, progress_callback=None, **kwargs):
        self.calls.append(kwargs.get('estimator'))
        return self.answer

def test_a_backend_that_offers_poisson_is_asked_first(irf):
    sentinel = {'from_backend': True}
    backend = _Recorder(sentinel)
    maps = fit_per_pixel(_stack(irf, 60.0), TCSPC, N_BINS, irf, has_tail=False, fit_bg=False,
                         fit_sigma=False, global_popt=_popt(), n_exp=3, min_photons=5,
                         gpu_backend=backend)
    assert maps is sentinel
    assert backend.calls == ['poisson']

def test_a_backend_that_declines_poisson_falls_back_to_the_core_fit(irf):
    backend = _Recorder(None)
    maps = fit_per_pixel(_stack(irf, 60.0), TCSPC, N_BINS, irf, has_tail=False, fit_bg=False,
                         fit_sigma=False, global_popt=_popt(), n_exp=3, min_photons=5,
                         gpu_backend=backend)
    assert backend.calls == ['poisson']
    assert np.isfinite(maps['alpha_1']).any()

def test_a_backend_without_the_poisson_estimator_is_not_asked(irf):
    backend = _Recorder({'from_backend': True})
    backend.estimators = ('nnls',)
    maps = fit_per_pixel(_stack(irf, 60.0), TCSPC, N_BINS, irf, has_tail=False, fit_bg=False,
                         fit_sigma=False, global_popt=_popt(), n_exp=3, min_photons=5,
                         gpu_backend=backend)
    assert backend.calls == []
    assert np.isfinite(maps['alpha_1']).any()
