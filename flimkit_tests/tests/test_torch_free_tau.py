import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('numba')

from flimkit.FLIM import nb_freetau
from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
from flimkit.FLIM.models import reconvolution_model
from flimkit.GPU.torch_backend import TorchBackend
from flimkit_tests.mock_data import MOCK_IRF_CENTER

N_BINS = 256
TCSPC = 97e-12
TAUS_NS = (0.6, 2.5)

@pytest.fixture
def pieces():
    irf = np.ascontiguousarray(
        gaussian_irf_from_fwhm(N_BINS, TCSPC, 0.3, MOCK_IRF_CENTER), dtype=float)
    popt = np.array([TAUS_NS[1] * 1e-9, TAUS_NS[0] * 1e-9, 800.0, 400.0, 0.0])
    clean = np.maximum(
        reconvolution_model(popt, TCSPC, N_BINS, irf, 2, 0.0, False, False, False), 0)
    rng = np.random.default_rng(11)
    stack = rng.poisson(clean[None, None, :] * np.ones((8, 8, 1))).astype(np.float32)
    return irf, stack, np.array([TAUS_NS[1] * 1e-9, TAUS_NS[0] * 1e-9])

def _numba_taus(irf, stack, taus_init, tau_min_s, tau_max_s):
    flat = stack.reshape(-1, N_BINS).astype(np.float64)
    raw = np.ascontiguousarray(flat)
    bg = nb_freetau.estimateBgRows(raw)
    idx = np.ascontiguousarray(np.arange(N_BINS), dtype=np.int64)
    amp_hi = raw.max() * 10.0
    p0 = np.ascontiguousarray(np.concatenate(
        [taus_init / 1e-9, np.full(2, raw.max() / 2)]))
    lo = np.ascontiguousarray([tau_min_s / 1e-9] * 2 + [0.0, 0.0])
    hi = np.ascontiguousarray([tau_max_s / 1e-9] * 2 + [amp_hi, amp_hi])
    p, _, _ = nb_freetau.fitFreeTau(raw, raw, bg, irf, idx, TCSPC / 1e-9,
                                    p0, lo, hi, 200, 1e-8)
    return np.sort(p[:, :2], axis=1)

def test_cpu_torch_supports_float64():
    assert TorchBackend('cpu')._supports_float64() == True

def test_mps_stays_on_scipy():
    if not torch.backends.mps.is_available():
        pytest.skip('no MPS on this machine')
    assert TorchBackend('mps')._supports_float64() == False

def test_batched_free_tau_matches_numba(pieces):
    irf, stack, taus_init = pieces
    tau_min_s, tau_max_s = 0.06e-9, 25e-9
    maps = TorchBackend('cpu').batch_free_tau_fit(
        stack, irf, TCSPC, taus_init, tau_min_s, tau_max_s, 2, 50, False, 0)
    got = np.sort(np.stack([maps['tau_1'].ravel(), maps['tau_2'].ravel()], axis=1), axis=1)
    want = _numba_taus(irf, stack, taus_init, tau_min_s, tau_max_s)
    assert np.isfinite(got).all()
    assert got == pytest.approx(want, abs=1e-6)

def test_batched_free_tau_lands_near_the_truth(pieces):
    irf, stack, taus_init = pieces
    maps = TorchBackend('cpu').batch_free_tau_fit(
        stack, irf, TCSPC, taus_init, 0.06e-9, 25e-9, 2, 50, False, 0)
    assert float(np.nanmedian(maps['tau_1'])) == pytest.approx(TAUS_NS[0], rel=0.25)
    assert float(np.nanmedian(maps['tau_2'])) == pytest.approx(TAUS_NS[1], rel=0.25)

def test_tvb_and_pileup_in_model_fall_back_to_scipy(pieces):
    irf, stack, taus_init = pieces
    be = TorchBackend('cpu')
    calls = []
    real = be._scipy_parallel_free_tau_fit

    def spy(*args, **kwargs):
        calls.append('scipy')
        return real(*args, **kwargs)

    be._scipy_parallel_free_tau_fit = spy
    be.batch_free_tau_fit(stack, irf, TCSPC, taus_init, 0.06e-9, 25e-9, 2, 50, False, 0,
                          n_sync_model=1000)
    assert calls == ['scipy']
