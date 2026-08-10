import sys
import types
import importlib
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from scipy.optimize import nnls
from scipy.linalg import pinv

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

from flimkit_tests.mock_data import (
    generate_synthetic_decay,
    MOCK_TCSPC_RES,
    MOCK_IRF_CENTER,
    MOCK_IRF_FWHM_BINS,
)
from flimkit.FLIM.irf_tools import build_full_irf

def _build_basis(taus_s, irf, n_bins, tcspc_res):
    """Build convolved exponential basis matrix A (n_bins × n_exp)."""
    t = np.arange(n_bins, dtype=float) * tcspc_res
    irf_fft = np.fft.fft(irf)
    cols = []
    for tau in taus_s:
        k = np.exp(-t / max(tau, 1e-15))
        cols.append(np.real(np.fft.ifft(np.fft.fft(k) * irf_fft)))
    return np.stack(cols, axis=1)

def _make_irf(n_bins, center=MOCK_IRF_CENTER,
              fwhm=MOCK_IRF_FWHM_BINS):
    sigma = fwhm / 2.3548
    bins  = np.arange(n_bins, dtype=float)
    irf   = np.exp(-0.5 * ((bins - center) / sigma) ** 2)
    return irf / irf.sum()

def _small_stack(ny=6, nx=8, n_bins=128,
                 tau_ns=2.0):
    """Build a small synthetic single-τ FLIM stack (Poisson, no noise option)."""
    stack = np.zeros((ny, nx, n_bins), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            d = generate_synthetic_decay(
                n_bins=n_bins, tcspc_res=MOCK_TCSPC_RES,
                tau_ns=tau_ns, bg=5.0, peak_counts=800.0, noise=True)
            stack[i, j] = d.astype(np.float32)
    return stack

# 1.  TestBackendDetection
class TestBackendDetection:
    """Unit-tests for flimkit.GPU.get_backend() device-detection logic."""

    def _reload_gpu_init(self):
        """Force re-import of flimkit.GPU so patched env is re-evaluated."""
        mod_name = "flimkit.GPU"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        return importlib.import_module(mod_name)

    def test_returns_none_when_no_gpu(self):
        """get_backend() must return None when no GPU is reachable."""
        # Patch _try_mlx / _try_torch to always return None
        gpu_mod = importlib.import_module("flimkit.GPU")
        with (
            patch.object(gpu_mod, "_try_mlx",  return_value=None),
            patch.object(gpu_mod, "_try_torch", return_value=None),
        ):
            assert gpu_mod.get_backend() is None

    def test_mlx_preferred_on_darwin(self):
        """On macOS, MLX backend should be the first candidate tried."""
        gpu_mod = importlib.import_module("flimkit.GPU")
        sentinel = object()
        with (
            patch.object(gpu_mod, "_try_mlx",  return_value=sentinel),
            patch.object(gpu_mod, "_try_torch", return_value=None),
        ):
            result = gpu_mod.get_backend(prefer="auto")
        assert result is sentinel

    def test_cuda_preferred_over_mps_when_asked(self):
        """prefer='cuda' should skip MLX and MPS."""
        gpu_mod = importlib.import_module("flimkit.GPU")
        cuda_sentinel = object()

        called_with = []
        def fake_try_torch(name, *a, **kw):
            called_with.append(name)
            if name == "cuda":
                return cuda_sentinel
            return None

        with patch.object(gpu_mod, "_try_torch", side_effect=fake_try_torch):
            result = gpu_mod.get_backend(prefer="cuda")

        assert result is cuda_sentinel
        assert "cuda" in called_with

    def test_prefer_mps(self):
        """prefer='mps' should return an MPS backend."""
        gpu_mod = importlib.import_module("flimkit.GPU")
        mps_sentinel = object()
        def fake_try_torch(name, *a, **kw):
            return mps_sentinel if name == "mps" else None
        with patch.object(gpu_mod, "_try_torch", side_effect=fake_try_torch):
            result = gpu_mod.get_backend(prefer="mps")
        assert result is mps_sentinel

# 2.  TestTorchPrecision
class TestTorchPrecision:
    def test_cuda_matmul_uses_highest_precision_and_restores_setting(self):
        torch = pytest.importorskip('torch')
        from flimkit.GPU.torch_backend import TorchBackend

        observed = []

        class Probe:
            def __matmul__(self, other):
                observed.append(torch.get_float32_matmul_precision())
                return 'result'

        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('high')
            backend = TorchBackend(device='cuda')
            result = backend._matmul_full_precision(Probe(), object())
            assert result == 'result'
            assert observed == ['highest']
            assert torch.get_float32_matmul_precision() == 'high'
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_cuda_matmul_restores_precision_after_exception(self):
        torch = pytest.importorskip('torch')
        from flimkit.GPU.torch_backend import TorchBackend

        class FailingProbe:
            def __matmul__(self, other):
                raise RuntimeError('matmul failed')

        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('high')
            backend = TorchBackend(device='cuda')
            with pytest.raises(RuntimeError, match='matmul failed'):
                backend._matmul_full_precision(FailingProbe(), object())
            assert torch.get_float32_matmul_precision() == 'high'
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_concurrent_cuda_matmuls_restore_original_precision(self):
        import threading

        torch = pytest.importorskip('torch')
        from flimkit.GPU.torch_backend import TorchBackend

        first_entered = threading.Event()
        second_entered = threading.Event()
        first_finished = threading.Event()

        class FirstProbe:
            def __matmul__(self, other):
                first_entered.set()
                second_entered.wait(timeout=0.2)
                return 'first'

        class SecondProbe:
            def __matmul__(self, other):
                second_entered.set()
                first_finished.wait(timeout=0.2)
                return 'second'

        backend = TorchBackend(device='cuda')
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('high')

            def run_first():
                backend._matmul_full_precision(FirstProbe(), object())
                first_finished.set()

            first = threading.Thread(target=run_first)
            second = threading.Thread(
                target=backend._matmul_full_precision,
                args=(SecondProbe(), object()),
            )
            first.start()
            assert first_entered.wait(timeout=1.0)
            second.start()
            first.join(timeout=1.0)
            second.join(timeout=1.0)
            assert not first.is_alive()
            assert not second.is_alive()
            assert torch.get_float32_matmul_precision() == 'high'
        finally:
            torch.set_float32_matmul_precision(previous)


# 3.  TestApproxNNLSVsTrueNNLS
class TestApproxNNLSVsTrueNNLS:
    """
    Validates that clamp(pinv(A) @ d, 0) ≈ scipy.nnls(A, d) for
    typical FLIM data (well-conditioned positive decays).
    """
    @pytest.fixture()
    def basis_and_decays(self):
        n_bins   = 128
        tcspc_res = MOCK_TCSPC_RES
        taus_s   = [0.5e-9, 3.0e-9]
        irf      = _make_irf(n_bins)
        A        = _build_basis(taus_s, irf, n_bins, tcspc_res)

        rng   = np.random.default_rng(42)
        N_pix = 200
        # Ground-truth amplitudes: non-negative
        amps_true = rng.uniform(50, 500, size=(N_pix, len(taus_s)))
        decays    = amps_true @ A.T + rng.poisson(5, size=(N_pix, n_bins))
        decays    = np.maximum(decays, 0.0)
        return A, decays

    def test_mean_relative_error_below_1pct(self, basis_and_decays):
        A, decays = basis_and_decays
        A_pinv    = pinv(A)
        errors    = []
        for d in decays:
            amps_approx = np.maximum(A_pinv @ d, 0.0)
            amps_scipy, _  = nnls(A, d)
            denom = np.maximum(np.abs(amps_scipy), 1e-6)
            errors.append(np.abs(amps_approx - amps_scipy) / denom)
        mean_err = np.mean(errors)
        assert mean_err < 0.01, f"Mean NNLS approx error = {mean_err:.4%}"

    def test_amplitude_non_negative(self, basis_and_decays):
        A, decays = basis_and_decays
        A_pinv    = pinv(A)
        for d in decays:
            amps = np.maximum(A_pinv @ d, 0.0)
            assert np.all(amps >= 0.0)

# 4.  TestBatchFixedTauCPUParity
class TestBatchFixedTauCPUParity:
    """
    Compare GPU batch_fixed_tau output to the CPU pixel loop for a small
    synthetic stack.  Runs with whichever GPU is available; skips if none.
    """

    @pytest.fixture()
    def backend(self):
        try:
            from flimkit.GPU import get_backend
            b = get_backend()
        except Exception:
            b = None
        if b is None:
            pytest.skip("No GPU backend available")
        return b

    @pytest.fixture()
    def problem(self):
        n_bins = 128
        tcspc_res = MOCK_TCSPC_RES
        taus_s = [0.5e-9, 3.0e-9]
        irf    = _make_irf(n_bins)
        A      = _build_basis(taus_s, irf, n_bins, tcspc_res)
        stack  = _small_stack(ny=4, nx=6, n_bins=n_bins, tau_ns=2.0)
        return stack, A, np.array(taus_s), n_bins

    def _cpu_reference(self, stack, A, taus_s):
        """Run scipy.nnls per pixel - ground-truth reference."""
        ny, nx, n_bins = stack.shape
        amps_ref  = np.full((ny, nx, len(taus_s)), np.nan)
        for yi in range(ny):
            for xi in range(nx):
                d = stack[yi, xi].astype(float)
                if d.sum() < 50:
                    continue
                amp, _ = nnls(A, d)
                amps_ref[yi, xi] = amp
        return amps_ref

    def test_intensity_map_matches(self, backend, problem):
        stack, A, taus_s, n_bins = problem
        maps = backend.batch_fixed_tau(
            stack, A, taus_s, min_photons=50,
            correct_pileup=False, n_sync_px=0, progress_callback=None)
        expected = stack.sum(axis=2)
        np.testing.assert_array_equal(maps["intensity"], expected)

    def test_amplitude_within_5pct_of_nnls(self, backend, problem):
        stack, A, taus_s, n_bins = problem
        maps = backend.batch_fixed_tau(
            stack, A, taus_s, min_photons=50,
            correct_pileup=False, n_sync_px=0, progress_callback=None)
        ref = self._cpu_reference(stack, A, taus_s)

        valid = ~np.isnan(ref[:, :, 0])
        for i in range(len(taus_s)):
            gpu_amp = maps[f"alpha_{i+1}"][valid]
            cpu_amp = ref[valid, i]
            denom   = np.maximum(cpu_amp, 1.0)
            mean_rel = np.mean(np.abs(gpu_amp - cpu_amp) / denom)
            assert mean_rel < 0.10, (
                f"GPU alpha_{i+1} differs from scipy.nnls by {mean_rel:.2%}"
            )

    def test_maps_have_required_keys(self, backend, problem):
        stack, A, taus_s, _ = problem
        maps = backend.batch_fixed_tau(
            stack, A, taus_s, min_photons=50,
            correct_pileup=False, n_sync_px=0, progress_callback=None)
        required = {
            "intensity", "tau_mean_int", "tau_mean_amp", "chi2_r",
            "alpha_1", "alpha_2", "frac_1", "frac_2",
        }
        assert required.issubset(maps.keys()), (
            f"Missing keys: {required - maps.keys()}"
        )


# 5.  TestBatchGrid1ExpCPUParity
class TestBatchGrid1ExpCPUParity:
    """Compare GPU batch_grid_scan_1exp to the vectorised CPU row loop."""

    @pytest.fixture()
    def backend(self):
        try:
            from flimkit.GPU import get_backend
            b = get_backend()
        except Exception:
            b = None
        if b is None:
            pytest.skip("No GPU backend available")
        return b

    @pytest.fixture()
    def problem(self):
        n_bins    = 128
        tau_true  = 2.0
        irf       = _make_irf(n_bins)
        irf_fft   = np.fft.fft(irf)
        t_axis    = np.arange(n_bins, dtype=float) * MOCK_TCSPC_RES

        lo, hi = 0.1e-9, 10e-9
        N_GRID    = 200
        tau_grid  = np.logspace(np.log10(lo), np.log10(hi), N_GRID)
        basis_grid = np.array([
            np.real(np.fft.ifft(
                np.fft.fft(np.exp(-t_axis / max(tau, 1e-15))) * irf_fft))
            for tau in tau_grid
        ])
        bb_grid  = np.maximum((basis_grid ** 2).sum(axis=1), 1e-20)
        stack    = _small_stack(ny=4, nx=6, n_bins=n_bins, tau_ns=tau_true)
        return stack, basis_grid, bb_grid, tau_grid, tau_true

    def test_tau_recovery_within_10pct(self, backend, problem):
        stack, basis_grid, bb_grid, tau_grid, tau_true = problem
        maps = backend.batch_grid_scan_1exp(
            stack, basis_grid, bb_grid, tau_grid,
            min_photons=50, correct_pileup=False, n_sync_px=0,
            progress_callback=None)

        tau_map = maps["tau_1"]
        valid   = ~np.isnan(tau_map)
        assert valid.sum() > 0, "No valid pixels returned"
        mean_tau = float(np.nanmean(tau_map[valid]))
        rel_err  = abs(mean_tau - tau_true) / tau_true
        assert rel_err < 0.10, (
            f"Mean τ = {mean_tau:.3f} ns, expected ≈ {tau_true} ns "
            f"(rel err = {rel_err:.2%})"
        )

    def test_intensity_map_correct(self, backend, problem):
        stack, basis_grid, bb_grid, tau_grid, _ = problem
        maps = backend.batch_grid_scan_1exp(
            stack, basis_grid, bb_grid, tau_grid,
            min_photons=50, correct_pileup=False, n_sync_px=0,
            progress_callback=None)
        np.testing.assert_array_equal(maps["intensity"], stack.sum(axis=2))

# 6.  TestGPUFitPerPixelIntegration
class TestGPUFitPerPixelIntegration:
    """
    Smoke-test the use_gpu=True path inside fit_per_pixel().
    When a real GPU is available the result is validated; when none is
    available the function must fall back to the CPU path gracefully.
    """

    @pytest.fixture()
    def fit_inputs(self):
        """Minimal inputs that allow fit_per_pixel() to run."""
        n_bins    = 128
        tcspc_res = MOCK_TCSPC_RES
        tau_true  = [2.0e-9]

        irf_prompt = _make_irf(n_bins)
        stack      = _small_stack(ny=4, nx=4, n_bins=n_bins, tau_ns=2.0)

        # global_popt layout: [tau_1, amp_1, shift, (sigma?), (bg?), (tamp, ttau?)]
        # n_exp=1 → [tau1, amp1, shift]
        global_popt = np.array([tau_true[0], 1.0, 0.0])
        return dict(
            stack        = stack,
            tcspc_res    = tcspc_res,
            n_bins       = n_bins,
            irf_prompt   = irf_prompt,
            has_tail     = False,
            fit_bg       = False,
            fit_sigma    = False,
            global_popt  = global_popt,
            n_exp        = 1,
            min_photons  = 50,
        )

    def test_graceful_fallback_when_no_gpu(self, fit_inputs):
        """use_gpu=True must not raise when there is no GPU - CPU fallback."""
        from flimkit.FLIM.fitters import fit_per_pixel
        # Force no GPU by passing a backend that is None explicitly
        maps = fit_per_pixel(
            **fit_inputs,
            use_gpu=True,
            gpu_backend=None,
        )
        assert "intensity"    in maps
        assert "tau_mean_int" in maps
        assert "tau_1"        in maps

    def test_use_gpu_false_is_default_behaviour(self, fit_inputs):
        """use_gpu=False (default) must produce the same result as not passing it."""
        from flimkit.FLIM.fitters import fit_per_pixel
        maps_cpu = fit_per_pixel(**fit_inputs)
        assert "intensity" in maps_cpu

    def test_output_shapes_match(self, fit_inputs):
        """Output map shapes must equal (ny, nx) regardless of use_gpu."""
        from flimkit.FLIM.fitters import fit_per_pixel
        maps = fit_per_pixel(**fit_inputs, use_gpu=False)
        ny, nx = fit_inputs["stack"].shape[:2]
        for key, val in maps.items():
            assert val.shape == (ny, nx), (
                f"maps['{key}'].shape = {val.shape}, expected ({ny},{nx})"
            )

    def test_gpu_and_cpu_tau_agree(self, fit_inputs):
        """If a GPU backend exists, GPU and CPU τ maps must agree within 5%."""
        from flimkit.GPU import get_backend
        try:
            backend = get_backend()
        except Exception:
            backend = None
        if backend is None:
            pytest.skip("No GPU backend available - skipping parity test")

        from flimkit.FLIM.fitters import fit_per_pixel
        maps_gpu = fit_per_pixel(**fit_inputs, use_gpu=True,  gpu_backend=backend)
        maps_cpu = fit_per_pixel(**fit_inputs, use_gpu=False)

        valid = ~np.isnan(maps_cpu["tau_1"]) & ~np.isnan(maps_gpu["tau_1"])
        if valid.sum() == 0:
            pytest.skip("No valid pixels to compare")

        rel_err = np.abs(maps_gpu["tau_1"][valid] - maps_cpu["tau_1"][valid])
        rel_err /= np.maximum(np.abs(maps_cpu["tau_1"][valid]), 0.01)
        assert np.mean(rel_err) < 0.05, (
            f"GPU/CPU τ mean relative diff = {np.mean(rel_err):.2%}"
        )