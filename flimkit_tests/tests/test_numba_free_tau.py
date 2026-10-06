import os
import numpy as np
import pytest
from flimkit.FLIM import fitters
from flimkit.FLIM.fitters import fit_per_pixel, _numba_free_tau_module
from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
from flimkit.FLIM.models import reconvolution_model
from flimkit_tests.mock_data import MOCK_IRF_CENTER

pytest.importorskip('numba')

N_BINS = 512
TCSPC = 97e-12
TAUS_NS = (0.5, 3.0)

@pytest.fixture
def irf():
    return gaussian_irf_from_fwhm(N_BINS, TCSPC, 0.3, MOCK_IRF_CENTER)

def _global_popt(amps=(4000.0, 2000.0), tail=None, bg=None):
    p = [TAUS_NS[0] * 1e-9, TAUS_NS[1] * 1e-9, amps[0], amps[1], 0.0]
    if bg is not None:
        p.append(bg)
    if tail is not None:
        p.extend(tail)
    return np.array(p)

def _two_pixel_stack(irf, has_tail=False, fit_bg=False):
    bg = 5.0 if fit_bg == True else None
    tail = (0.05, 20.0) if has_tail == True else None
    rows = []
    for scale in (1.0, 0.5):
        p = _global_popt(amps=(4000.0 * scale, 2000.0 * scale), tail=tail, bg=bg)
        rows.append(reconvolution_model(p, TCSPC, N_BINS, irf, 2, 0.0,
                                        has_tail, fit_bg, False))
    return np.stack(rows)[:, None, :], _global_popt(tail=tail, bg=bg)

def _fit(irf, stack, global_popt, has_tail, fit_bg, numba):
    before = os.environ.get('FLIMKIT_NUMBA_FREETAU')
    os.environ['FLIMKIT_NUMBA_FREETAU'] = '1' if numba == True else '0'
    try:
        return fit_per_pixel(
            stack, TCSPC, N_BINS, irf,
            has_tail=has_tail, fit_bg=fit_bg, fit_sigma=False,
            global_popt=global_popt, n_exp=2, min_photons=10,
            free_tau=True, use_gpu=False)
    finally:
        if before is None:
            del os.environ['FLIMKIT_NUMBA_FREETAU']
        else:
            os.environ['FLIMKIT_NUMBA_FREETAU'] = before

def test_numba_module_loads():
    assert _numba_free_tau_module(False, False, None, 2) is not None

@pytest.mark.parametrize('tail,tvb,n_sync,n_exp', [
    (True, False, None, 2),
    (False, True, None, 2),
    (False, False, 1000, 2),
    (False, False, None, 4),
])
def test_numba_declines_what_it_cannot_model(tail, tvb, n_sync, n_exp):
    assert _numba_free_tau_module(tail, tvb, n_sync, n_exp) is None

def test_env_switch_keeps_scipy(monkeypatch):
    monkeypatch.setenv('FLIMKIT_NUMBA_FREETAU', '0')
    assert _numba_free_tau_module(False, False, None, 2) is None

@pytest.mark.parametrize('numba', [True, False])
@pytest.mark.parametrize('has_tail,fit_bg', [(False, False), (True, True)])
def test_free_tau_recovers_known_taus(irf, numba, has_tail, fit_bg):
    stack, global_popt = _two_pixel_stack(irf, has_tail, fit_bg)
    maps = _fit(irf, stack, global_popt, has_tail, fit_bg, numba)
    for yi in (0, 1):
        taus = sorted([maps['tau_1'][yi, 0], maps['tau_2'][yi, 0]])
        assert taus[0] == pytest.approx(TAUS_NS[0], rel=0.01)
        assert taus[1] == pytest.approx(TAUS_NS[1], rel=0.01)

@pytest.mark.parametrize('has_tail,fit_bg', [(False, False), (True, True)])
def test_numba_agrees_with_scipy(irf, has_tail, fit_bg):
    stack, global_popt = _two_pixel_stack(irf, has_tail, fit_bg)
    nb_maps = _fit(irf, stack, global_popt, has_tail, fit_bg, True)
    sp_maps = _fit(irf, stack, global_popt, has_tail, fit_bg, False)
    for key in ('tau_1', 'tau_2', 'tau_mean_amp', 'tau_mean_int'):
        assert nb_maps[key] == pytest.approx(sp_maps[key], rel=0.02), key
    for key in ('frac_1', 'frac_2'):
        assert nb_maps[key] == pytest.approx(sp_maps[key], abs=0.02), key

def test_numba_block_is_capped_by_bytes(irf, monkeypatch):
    calls = []
    nbm = _numba_free_tau_module(False, False, None, 2)
    real = nbm.fitFreeTau

    def spy(data, *args, **kwargs):
        calls.append(data.shape[0])
        return real(data, *args, **kwargs)

    monkeypatch.setattr(nbm, 'fitFreeTau', spy)
    monkeypatch.setattr(fitters, '_NUMBA_FREE_TAU_BLOCK_BYTES', 8 * N_BINS * 4)
    stack, global_popt = _two_pixel_stack(irf)
    stack = np.repeat(stack, 8, axis=0)
    _fit(irf, stack, global_popt, False, False, True)
    assert len(calls) > 1
    assert max(calls) <= 4
