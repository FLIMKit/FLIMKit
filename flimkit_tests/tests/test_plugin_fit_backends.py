import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

from flimkit import GPU, plugins
from flimkit.FLIM.fitters import fit_per_pixel
from flimkit.GPU import PluginBackend
from flimkit.plugins import loader, registry
from flimkit_tests.mock_data import (
    MOCK_IRF_CENTER,
    MOCK_IRF_FWHM_BINS,
    MOCK_TCSPC_RES,
    generate_synthetic_biexp_decay,
)

N_BINS = 128


@pytest.fixture(autouse=True)
def no_builtin_gpu(monkeypatch):
    monkeypatch.setattr(GPU, 'BUILTIN', ())
    monkeypatch.delenv('FLIMKIT_FIT_BACKEND', raising=False)
    loader.reset()
    plugins.ensure_loaded()
    yield
    for b in plugins.fit_backends():
        registry._fit_backends.pop(b.id, None)


class FreeTauOnly(PluginBackend):

    def __init__(self):
        self.calls = []

    def batch_free_tau_fit(self, stack, irf, tcspc_res, taus_init, tau_min_s, tau_max_s,
                           n_exp, min_photons, correct_pileup, n_sync_px, **kwargs):
        self.calls.append('batch_free_tau_fit')
        ny, nx, _ = stack.shape
        return {'intensity': stack.sum(axis=2), 'from_addon': np.ones((ny, nx))}

    def __repr__(self):
        return 'FreeTauOnly()'


def _write_plugin(tmp_path, body):
    path = tmp_path / 'fast_backend.py'
    path.write_text('from flimkit.plugins import fit_backend\n'
                    'FLIMKIT_PLUGIN_API = 1\n' + body)
    return str(path)


def test_registers_and_sorts_by_priority():
    plugins.register_fit_backend('slow', 'Slow', lambda: None, priority=200)
    plugins.register_fit_backend('Fast', 'Fast', lambda: None, priority=10)
    assert [b.id for b in plugins.fit_backends()] == ['fast', 'slow']
    assert plugins.get_fit_backend('FAST').label == 'Fast'


def test_duplicate_and_builtin_ids_are_refused():
    plugins.register_fit_backend('mine', 'Mine', lambda: None)
    with pytest.raises(plugins.PluginError, match='already registered'):
        plugins.register_fit_backend('mine', 'Again', lambda: None)
    for name in ('auto', 'cpu', 'cuda', 'MLX'):
        with pytest.raises(plugins.PluginError, match='built-in'):
            plugins.register_fit_backend(name, name, lambda: None)


def test_decorator_registers_from_a_plugin_file(tmp_path):
    path = _write_plugin(tmp_path, (
        '@fit_backend("c_demo", "C demo", priority=5)\n'
        'class CDemo:\n'
        '    def batch_fixed_tau(self, *a, **k):\n'
        '        return None\n'))
    result = plugins.load_path(path)
    assert result.ok == True, result.error
    assert result.n_registered == 1
    found = plugins.get_fit_backend('c_demo')
    assert found.priority == 5
    assert found.source == path


def test_a_failed_plugin_rolls_its_backend_back(tmp_path):
    path = _write_plugin(tmp_path, (
        '@fit_backend("doomed", "Doomed")\n'
        'def make():\n'
        '    return None\n'
        'raise RuntimeError("library missing")\n'))
    result = plugins.load_path(path)
    assert result.ok == False
    assert plugins.get_fit_backend('doomed') is None


def test_auto_picks_the_addon_and_tags_it():
    plugins.register_fit_backend('c_free', 'C free-tau', FreeTauOnly)
    backend = GPU.get_backend()
    assert isinstance(backend, FreeTauOnly)
    assert backend.plugin_id == 'c_free'


def test_auto_skips_addons_that_are_unavailable_or_broken(capsys):
    def broken():
        raise OSError('libflim.so: cannot open shared object file')
    plugins.register_fit_backend('no_device', 'No device', lambda: None, priority=1)
    plugins.register_fit_backend('broken', 'Broken', broken, priority=2)
    plugins.register_fit_backend('works', 'Works', FreeTauOnly, priority=3)
    assert isinstance(GPU.get_backend(), FreeTauOnly)
    assert 'broken' in capsys.readouterr().out


def test_an_object_with_no_fit_methods_is_refused(capsys):
    plugins.register_fit_backend('empty', 'Empty', object)
    assert GPU.get_backend() is None
    assert 'none of' in capsys.readouterr().out


def test_by_name_and_environment_override(monkeypatch):
    plugins.register_fit_backend('c_free', 'C free-tau', FreeTauOnly)
    assert isinstance(GPU.get_backend('c_free'), FreeTauOnly)
    monkeypatch.setenv('FLIMKIT_FIT_BACKEND', 'cpu')
    assert GPU.get_backend() is None
    monkeypatch.setenv('FLIMKIT_FIT_BACKEND', 'c_free')
    assert isinstance(GPU.get_backend(), FreeTauOnly)


def test_unknown_name_lists_addons():
    plugins.register_fit_backend('c_free', 'C free-tau', FreeTauOnly)
    with pytest.raises(ValueError, match="'c_free'"):
        GPU.get_backend('nope')


def _irf():
    sigma = MOCK_IRF_FWHM_BINS / 2.3548
    bins = np.arange(N_BINS, dtype=float)
    irf = np.exp(-0.5 * ((bins - MOCK_IRF_CENTER) / sigma) ** 2)
    return irf / irf.sum()


def _fit(backend, free_tau):
    stack = np.stack([np.stack([generate_synthetic_biexp_decay(
        n_bins=N_BINS, tcspc_res=MOCK_TCSPC_RES, tau1_ns=0.5, tau2_ns=3.0,
        a1=0.6, a2=0.4, bg=5.0, peak_counts=3000.0, noise=True)
        for _ in range(3)]) for _ in range(2)]).astype(np.float32)
    return fit_per_pixel(
        stack=stack, tcspc_res=MOCK_TCSPC_RES, n_bins=N_BINS, irf_prompt=_irf(),
        has_tail=False, fit_bg=False, fit_sigma=False,
        global_popt=np.array([0.5e-9, 3.0e-9, 0.6, 0.4, 0.0]), n_exp=2,
        min_photons=50, free_tau=free_tau, gpu_backend=backend)


def test_addon_free_tau_goes_ahead_of_numba():
    backend = plugins.register_fit_backend('c_free', 'C free-tau', FreeTauOnly).create()
    maps = _fit(backend, free_tau=True)
    assert backend.calls == ['batch_free_tau_fit']
    assert 'from_addon' in maps


def test_a_fit_the_addon_does_not_cover_falls_back_to_the_cpu(capsys):
    backend = plugins.register_fit_backend('c_free', 'C free-tau', FreeTauOnly).create()
    maps = _fit(backend, free_tau=False)
    assert backend.calls == []
    assert 'from_addon' not in maps
    assert np.isfinite(maps['tau_mean_amp']).all()
    assert 'does not implement batch_fixed_tau' in capsys.readouterr().out


class FakeBuiltin(PluginBackend):

    def __init__(self):
        self.calls = []

    def batch_fixed_tau(self, stack, *args, **kwargs):
        self.calls.append('batch_fixed_tau')
        return {'intensity': stack.sum(axis=2), 'from_builtin': np.ones(stack.shape[:2])}

    def __repr__(self):
        return 'FakeBuiltin()'

def test_a_free_tau_fit_the_addon_does_not_cover_falls_back(capsys):
    backend = plugins.register_fit_backend('c_fixed', 'C fixed-tau', FakeBuiltin).create()
    maps = _fit(backend, free_tau=True)
    assert backend.calls == []
    assert 'from_builtin' not in maps
    assert np.isfinite(maps['tau_1']).any()
    assert 'does not implement batch_free_tau_fit' in capsys.readouterr().out


def test_a_fit_the_addon_declines_goes_to_the_gpu_backend_it_displaced(monkeypatch):
    monkeypatch.setattr(GPU, 'BUILTIN', ('mlx',))
    monkeypatch.setattr(GPU, '_try_mlx', FakeBuiltin)
    plugins.register_fit_backend('c_free', 'C free-tau', FreeTauOnly)
    backend = GPU.get_backend()
    assert isinstance(backend, FreeTauOnly)
    assert isinstance(backend.fallback_backend, FakeBuiltin)
    maps = _fit(backend, free_tau=False)
    assert backend.fallback_backend.calls == ['batch_fixed_tau']
    assert 'from_builtin' in maps
    assert 'from_addon' in _fit(backend, free_tau=True)


def test_without_addons_auto_still_returns_the_builtin(monkeypatch):
    monkeypatch.setattr(GPU, 'BUILTIN', ('mlx',))
    monkeypatch.setattr(GPU, '_try_mlx', FakeBuiltin)
    assert isinstance(GPU.get_backend(), FakeBuiltin)
