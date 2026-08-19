import numpy as np
import pytest
from flimkit.FLIM.models import (reconvolution_model, dist_reconvolution_model,
                                 _DECost, _DECostDist)
from flimkit.FLIM.fit_tools import (_build_bounds, _pack_p0,
                                    _build_bounds_dist, _pack_p0_dist)
from flimkit.FLIM.fitters import (fit_summed, fit_summed_dist,
                                  fit_per_pixel, fit_per_pixel_dist)
from flimkit.FLIM.bg_tools import tvb_from_decay

N = 256
RES = 1e-10
T = np.arange(N)

def _irf(center=25, sigma=2.2, n=N):
    b = np.arange(n)
    g = np.exp(-0.5 * ((b - center) / sigma) ** 2)
    return g / g.sum()

def _bg_profile(n=N, tau_bins=40.0):
    return tvb_from_decay(np.exp(-np.arange(n) / tau_bins), n)

def _fluor(taus_ns, amps, n_exp, irf):
    p = np.array(list(taus_ns) + list(amps) + [0.0])
    return reconvolution_model(p, RES, N, irf, n_exp, 0.0, False, False, False)

def _noisy_stack(taus_ns, amps, scale, n_exp, ny=8, nx=8, seed=3):
    rng = np.random.default_rng(seed)
    irf = _irf()
    total = _fluor(taus_ns, amps, n_exp, irf) + scale * _bg_profile()
    stack = np.maximum(total, 0)[None, None, :].repeat(ny, 0).repeat(nx, 1)
    return rng.poisson(stack).astype(float)

@pytest.fixture(scope='module')
def gpu_backend():
    try:
        from flimkit.GPU import get_backend
        return get_backend()
    except Exception:
        return None

class TestLoader:
    def test_normalizes_to_unit_area(self):
        raw = 50.0 * np.exp(-T / 40.0) + 5.0
        p = tvb_from_decay(raw, N)
        assert abs(p.sum() - 1.0) < 1e-9
        assert p.size == N

    def test_pads_short_decay_with_zeros(self):
        p = tvb_from_decay((50.0 * np.exp(-T / 40.0))[:200], N)
        assert p.size == N
        assert np.all(p[200:] == 0)

    def test_truncates_long_decay(self):
        p = tvb_from_decay(50.0 * np.exp(-np.arange(N + 40) / 40.0), N)
        assert p.size == N

    def test_unnormalized_preserves_total(self):
        raw = 50.0 * np.exp(-T / 40.0) + 5.0
        p = tvb_from_decay(raw, N, normalize=False)
        assert abs(p.sum() - float(raw.sum())) < 1e-6

    def test_resample_returns_target_length(self):
        raw = 50.0 * np.exp(-T / 40.0)
        p = tvb_from_decay(raw, N, src_tcspc_res=2e-11, dst_tcspc_res=1e-11)
        assert p.size == N

    def test_zero_photons_raises(self):
        with pytest.raises(ValueError):
            tvb_from_decay(np.zeros(N), N)

class TestModel:
    def test_reconv_adds_scaled_profile(self):
        irf = _irf()
        B = _bg_profile()
        p = np.array([2.0e-9, 1000.0, 0.0])
        off = reconvolution_model(p, RES, N, irf, 1, 0.0, False, False, False)
        on = reconvolution_model(np.append(p, 250.0), RES, N, irf, 1, 0.0,
                                 False, False, False, tvb_profile=B, fit_tvb=True)
        assert np.allclose(on, off + 250.0 * B)

    def test_reconv_profile_none_is_backward_compatible(self):
        irf = _irf()
        p = np.array([2.0e-9, 1000.0, 0.0])
        off = reconvolution_model(p, RES, N, irf, 1, 0.0, False, False, False)
        none = reconvolution_model(p, RES, N, irf, 1, 0.0, False, False, False,
                                   tvb_profile=None)
        assert np.allclose(off, none)

    def test_reconv_fixed_scale_path(self):
        irf = _irf()
        B = _bg_profile()
        p = np.array([2.0e-9, 1000.0, 0.0])
        off = reconvolution_model(p, RES, N, irf, 1, 0.0, False, False, False)
        fix = reconvolution_model(p, RES, N, irf, 1, 0.0, False, False, False,
                                  tvb_profile=B, fit_tvb=False, tvb_fixed=0.5)
        assert np.allclose(fix, off + 0.5 * B)

    def test_dist_adds_scaled_profile(self):
        irf = _irf()
        B = _bg_profile()
        p = np.array([2.0e-9, 0.4e-9, 1000.0, 0.0])
        off = dist_reconvolution_model(p, RES, N, irf, 1, 'gaussian', 0.0, False, False)
        on = dist_reconvolution_model(np.append(p, 250.0), RES, N, irf, 1, 'gaussian',
                                      0.0, False, False, tvb_profile=B, fit_tvb=True)
        assert np.allclose(on, off + 250.0 * B)

class TestBoundsPacking:
    def test_reconv_p0_bounds_model_aligned(self):
        irf = _irf()
        B = _bg_profile()
        tmin, tmax, peak = 0.2e-9, 5e-9, 1000.0
        p0 = _pack_p0(2, tmin, tmax, peak, False, True, True, 5.0,
                      fit_tvb=True, tvb_init=33.0)
        lo, hi = _build_bounds(2, tmin, tmax, peak, False, True, True,
                               bg_init=5.0, fit_tvb=True, tvb_init=33.0)
        assert len(p0) == len(lo) == len(hi) == 8
        assert p0[-1] == 33.0
        assert lo[-1] == 0.0
        m_tvb = reconvolution_model(p0, RES, N, irf, 2, 0.0, False, True, True,
                                    tvb_profile=B, fit_tvb=True)
        m_base = reconvolution_model(p0[:-1], RES, N, irf, 2, 0.0, False, True, True)
        assert np.allclose(m_tvb, m_base + 33.0 * B)

    def test_dist_p0_bounds_model_aligned(self):
        irf = _irf()
        B = _bg_profile()
        tmin, tmax, peak = 0.2e-9, 5e-9, 1000.0
        pd = _pack_p0_dist(2, tmin, tmax, peak, True, True, 5.0,
                           fit_tvb=True, tvb_init=22.0)
        ld, hd = _build_bounds_dist(2, tmin, tmax, peak, True, True,
                                    bg_init=5.0, fit_tvb=True, tvb_init=22.0)
        assert len(pd) == len(ld) == len(hd) == 10
        m_tvb = dist_reconvolution_model(pd, RES, N, irf, 2, 'gaussian', 0.0, True, True,
                                         tvb_profile=B, fit_tvb=True)
        m_base = dist_reconvolution_model(pd[:-1], RES, N, irf, 2, 'gaussian', 0.0, True, True)
        assert np.allclose(m_tvb, m_base + 22.0 * B)

    def test_no_tvb_param_vector_unchanged(self):
        p0 = _pack_p0(2, 0.2e-9, 5e-9, 1000.0, False, True, True, 5.0)
        assert len(p0) == 7

class TestCostClasses:
    def test_decost_minimised_by_tvb(self):
        irf = _irf()
        B = _bg_profile()
        p = np.array([2.0e-9, 0.6e-9, 700.0, 300.0, 0.0, 8.0, 400.0])
        decay = reconvolution_model(p, RES, N, irf, 2, 0.0, False, True, False,
                                    tvb_profile=B, fit_tvb=True)
        w = np.sqrt(np.maximum(decay, 1.0))
        c_tvb = _DECost(RES, N, irf, 2, 0.0, False, True, False, np.arange(N), decay, w,
                        tvb_profile=B, fit_tvb=True)(p)
        c_no = _DECost(RES, N, irf, 2, 0.0, False, True, False, np.arange(N), decay, w)(p[:-1])
        assert c_tvb < 1e-6
        assert c_no > 1.0

    def test_decostdist_minimised_by_tvb(self):
        irf = _irf()
        B = _bg_profile()
        pd = np.array([2.0e-9, 0.4e-9, 1000.0, 0.0, 7.0, 350.0])
        dd = dist_reconvolution_model(pd, RES, N, irf, 1, 'gaussian', 0.0, True, False,
                                      tvb_profile=B, fit_tvb=True)
        wd = np.sqrt(np.maximum(dd, 1.0))
        c_tvb = _DECostDist(RES, N, irf, 1, 'gaussian', 0.0, True, False, np.arange(N), dd, wd,
                            tvb_profile=B, fit_tvb=True)(pd)
        c_no = _DECostDist(RES, N, irf, 1, 'gaussian', 0.0, True, False, np.arange(N), dd, wd)(pd[:-1])
        assert c_tvb < 1e-6
        assert c_no > 1.0

class TestSummedFit:
    def test_reconv_recovers_scale_and_taus(self):
        rng = np.random.default_rng(0)
        irf = _irf()
        B = _bg_profile()
        decay = _fluor([2.5e-9, 0.7e-9], [800.0, 400.0], 2, irf) + 300.0 * B
        decay = rng.poisson(np.maximum(decay, 0)).astype(float)
        _, s_on = fit_summed(decay, RES, N, irf, False, True, False, 2, 0.2, 6.0,
                             optimizer='lm_multistart', n_restarts=6,
                             cost_function='poisson', tvb_profile=B, fit_tvb=True)
        _, s_off = fit_summed(decay, RES, N, irf, False, True, False, 2, 0.2, 6.0,
                              optimizer='lm_multistart', n_restarts=6,
                              cost_function='poisson')
        taus = np.sort(s_on['taus_ns'])[::-1]
        assert abs(taus[0] - 2.5) < 0.5
        assert 150.0 < s_on['tvb_scale'] < 480.0
        assert s_on['reduced_chi2'] < s_off['reduced_chi2']

    def test_tvb_scale_key_present_when_off(self):
        rng = np.random.default_rng(1)
        irf = _irf()
        decay = _fluor([2.5e-9, 0.7e-9], [800.0, 400.0], 2, irf)
        decay = rng.poisson(np.maximum(decay, 0)).astype(float)
        _, s = fit_summed(decay, RES, N, irf, False, True, False, 2, 0.2, 6.0,
                          optimizer='lm_multistart', n_restarts=4,
                          cost_function='poisson')
        assert 'tvb_scale' in s
        assert s['tvb_scale'] == 0.0

    def test_dist_recovers_centre_and_scale(self):
        rng = np.random.default_rng(2)
        irf = _irf()
        B = _bg_profile()
        fluor = dist_reconvolution_model(np.array([2.0e-9, 0.4e-9, 1000.0, 0.0]),
                                         RES, N, irf, 1, 'gaussian', 0.0, False, False)
        decay = rng.poisson(np.maximum(fluor + 300.0 * B, 0)).astype(float)
        _, s_on = fit_summed_dist(decay, RES, N, irf, 1, 'gaussian', True, False, 0.2, 6.0,
                                  optimizer='lm_multistart', n_restarts=6,
                                  cost_function='poisson', tvb_profile=B, fit_tvb=True)
        _, s_off = fit_summed_dist(decay, RES, N, irf, 1, 'gaussian', True, False, 0.2, 6.0,
                                   optimizer='lm_multistart', n_restarts=6,
                                   cost_function='poisson')
        assert abs(float(s_on['tau_centers_ns'][0]) - 2.0) < 0.3
        assert 150.0 < s_on['tvb_scale'] < 480.0
        assert s_on['reduced_chi2'] < s_off['reduced_chi2']

class TestPerPixelFixedTau:
    def test_cpu_recovers_and_improves_chi2(self):
        irf = _irf()
        B = _bg_profile()
        stack = _noisy_stack([2.5e-9, 0.7e-9], [800.0, 400.0], 300.0, 2)
        gp = np.array([2.5e-9, 0.7e-9, 800.0, 400.0, 0.0])
        on = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 2,
                           use_gpu=False, tvb_profile=B, fit_tvb=True)
        off = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 2,
                            use_gpu=False)
        assert 'tvb_scale' in on
        assert np.nanmedian(on['chi2_r']) < np.nanmedian(off['chi2_r'])
        assert 150.0 < np.nanmedian(on['tvb_scale']) < 480.0
        assert abs(np.nanmedian(on['tau_mean_amp']) - 1.9) < 0.25

    def test_no_tvb_map_when_off(self):
        irf = _irf()
        stack = _noisy_stack([2.5e-9, 0.7e-9], [800.0, 400.0], 300.0, 2)
        gp = np.array([2.5e-9, 0.7e-9, 800.0, 400.0, 0.0])
        off = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 2, use_gpu=False)
        assert 'tvb_scale' not in off

class TestPerPixelGridScan:
    def test_cpu_recovers_mono_exp(self):
        irf = _irf()
        B = _bg_profile()
        stack = _noisy_stack([2.0e-9], [1000.0], 300.0, 1, seed=7)
        gp = np.array([2.0e-9, 1000.0, 0.0])
        on = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 1,
                           use_gpu=False, tvb_profile=B, fit_tvb=True)
        off = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 1, use_gpu=False)
        assert 'tvb_scale' in on
        assert abs(np.nanmedian(on['tau_1']) - 2.0) < 0.2
        assert 150.0 < np.nanmedian(on['tvb_scale']) < 480.0
        assert np.nanmedian(on['chi2_r']) < np.nanmedian(off['chi2_r'])

    def test_cpu_gpu_parity(self, gpu_backend, monkeypatch):
        if gpu_backend is None:
            pytest.skip('no GPU backend available')
        irf = _irf()
        B = _bg_profile()
        stack = _noisy_stack([2.0e-9], [1000.0], 300.0, 1, ny=10, nx=10, seed=7)
        gp = np.array([2.0e-9, 1000.0, 0.0])
        import flimkit.FLIM.fitters as F
        monkeypatch.setattr(F, '_gpu_backend_cache', None)
        cpu = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 1,
                            use_gpu=False, tvb_profile=B, fit_tvb=True)
        gpu = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 1,
                            use_gpu='auto', gpu_backend=gpu_backend,
                            tvb_profile=B, fit_tvb=True)
        from flimkit.FLIM.fitters import tau_grid_points
        lo = max(2.0 / 20.0, 0.05)
        hi = min(2.0 * 20.0, 45.0)
        step_ratio = (hi / lo) ** (1.0 / (tau_grid_points() - 1))
        one_step = 2.0 * (step_ratio - 1.0)
        tolerance = max(one_step, 0.005 * 2.0) * 1.05
        gap = np.abs(cpu['tau_1'] - gpu['tau_1'])
        d_tau = np.nanmax(gap)
        assert d_tau <= tolerance, (
            f'CPU and GPU differ by {d_tau:.5f} ns, more than the {tolerance:.5f} ns '
            'they are expected to agree within, which is the coarser of the tau grid '
            f'spacing ({one_step:.5f} ns) and half a percent of the lifetime')
        centres = abs(np.nanmedian(cpu['tau_1']) - np.nanmedian(gpu['tau_1']))
        assert centres <= one_step, (
            f'the two paths centre on lifetimes {centres:.5f} ns apart, more than '
            'one grid step, so they are not fitting the same thing')

class TestPerPixelDist:
    def _dist_stack(self, tau_c_ns, width_ns, amp, scale, ny=8, nx=8, seed=4):
        rng = np.random.default_rng(seed)
        irf = _irf()
        fluor = dist_reconvolution_model(np.array([tau_c_ns, width_ns, amp, 0.0]),
                                         RES, N, irf, 1, 'gaussian', 0.0, False, False)
        total = np.maximum(fluor + scale * _bg_profile(), 0)
        return rng.poisson(total[None, None, :].repeat(ny, 0).repeat(nx, 1)).astype(float)

    def test_cpu_unimodal_recovers(self, monkeypatch):
        import flimkit.FLIM.fitters as F
        monkeypatch.setattr(F, '_gpu_backend_cache', None)
        irf = _irf()
        B = _bg_profile()
        stack = self._dist_stack(2.0e-9, 0.4e-9, 1000.0, 300.0)
        gp = np.array([2.0e-9, 0.4e-9, 1000.0, 0.0])
        on = fit_per_pixel_dist(stack, RES, N, irf, gp, 1, 'gaussian',
                                fit_bg=True, fit_sigma=False, gpu_backend=None,
                                tvb_profile=B, fit_tvb=True)
        off = fit_per_pixel_dist(stack, RES, N, irf, gp, 1, 'gaussian',
                                 fit_bg=True, fit_sigma=False, gpu_backend=None)
        assert 'tvb_scale' in on
        assert 'tvb_scale' not in off
        assert abs(np.nanmedian(on['tau_center_1']) - 2.0) < 0.4
        assert 150.0 < np.nanmedian(on['tvb_scale']) < 480.0
        assert np.nanmedian(on['chi2_r']) < np.nanmedian(off['chi2_r'])

    def test_cpu_gpu_parity(self, gpu_backend, monkeypatch):
        if gpu_backend is None:
            pytest.skip('no GPU backend available')
        irf = _irf()
        B = _bg_profile()
        stack = self._dist_stack(2.0e-9, 0.4e-9, 1000.0, 300.0, ny=10, nx=10)
        gp = np.array([2.0e-9, 0.4e-9, 1000.0, 0.0])
        import flimkit.FLIM.fitters as F
        monkeypatch.setattr(F, '_gpu_backend_cache', None)
        cpu = fit_per_pixel_dist(stack, RES, N, irf, gp, 1, 'gaussian',
                                 fit_bg=True, fit_sigma=False, gpu_backend=None,
                                 tvb_profile=B, fit_tvb=True)
        gpu = fit_per_pixel_dist(stack, RES, N, irf, gp, 1, 'gaussian',
                                 fit_bg=True, fit_sigma=False, gpu_backend=gpu_backend,
                                 tvb_profile=B, fit_tvb=True)
        d = np.nanmax(np.abs(cpu['tau_center_1'] - gpu['tau_center_1']))
        assert d < 1e-3


class TestPerPixelFreeTau:
    def test_cpu_improves_chi2(self):
        irf = _irf()
        B = _bg_profile()
        stack = _noisy_stack([2.5e-9, 0.7e-9], [800.0, 400.0], 300.0, 2,
                             ny=4, nx=4, seed=5)
        gp = np.array([2.5e-9, 0.7e-9, 800.0, 400.0, 0.0])
        on = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 2,
                           free_tau=True, use_gpu=False, tvb_profile=B, fit_tvb=True)
        off = fit_per_pixel(stack, RES, N, irf, False, False, False, gp, 2,
                            free_tau=True, use_gpu=False)
        assert 'tvb_scale' in on
        assert np.nanmedian(on['chi2_r']) < np.nanmedian(off['chi2_r'])


class TestExports:
    def test_summary_txt_includes_tvb_scale(self, tmp_path):
        from flimkit.utils.enhanced_outputs import save_fit_summary_txt
        p = tmp_path / 'summary.txt'
        save_fit_summary_txt({'tvb_scale': 123.4, 'chi2r': 1.0}, p, n_exp=2)
        assert 'TVB scale: 123.4' in p.read_text()

    def test_summary_txt_omits_tvb_when_zero(self, tmp_path):
        from flimkit.utils.enhanced_outputs import save_fit_summary_txt
        p = tmp_path / 'summary.txt'
        save_fit_summary_txt({'tvb_scale': 0.0, 'chi2r': 1.0}, p, n_exp=2)
        assert 'TVB scale' not in p.read_text()

    def test_individual_maps_save_tvb_scale_tif(self, tmp_path):
        from flimkit.utils.enhanced_outputs import save_individual_tau_maps
        maps = {'tau_1': np.full((4, 4), 2.0, dtype=np.float32),
                'a1': np.full((4, 4), 1.0, dtype=np.float32),
                'tvb_scale': np.full((4, 4), 50.0, dtype=np.float32)}
        save_individual_tau_maps(maps, tmp_path, roi_name='X', n_exp=1)
        assert (tmp_path / 'X_tvb_scale.tif').exists()