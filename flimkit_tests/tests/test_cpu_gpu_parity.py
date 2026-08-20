import sys
import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))

from flimkit_tests.mock_data import (
    generate_synthetic_decay,
    generate_synthetic_biexp_decay,
    MOCK_TCSPC_RES,
    MOCK_IRF_CENTER,
    MOCK_IRF_FWHM_BINS,
)
from flimkit.FLIM.irf_tools import build_full_irf
from flimkit.FLIM.fitters import fit_per_pixel


N_BINS = 128
TCSPC_RES = MOCK_TCSPC_RES
MIN_PHOTONS = 50

AMP_TOL   = 0.08
TAU_TOL   = 0.08
TAU1_TOL  = 0.12


def _make_irf(n_bins=N_BINS, center=MOCK_IRF_CENTER, fwhm=MOCK_IRF_FWHM_BINS):
    sigma = fwhm / 2.3548
    bins  = np.arange(n_bins, dtype=float)
    irf   = np.exp(-0.5 * ((bins - center) / sigma) ** 2)
    return irf / irf.sum()


def _synthetic_stack_1exp(ny=6, nx=8, tau_ns=2.0, n_bins=N_BINS):
    stack = np.zeros((ny, nx, n_bins), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            d = generate_synthetic_decay(
                n_bins=n_bins, tcspc_res=TCSPC_RES,
                tau_ns=tau_ns, bg=5.0, peak_counts=800.0, noise=True)
            stack[i, j] = d.astype(np.float32)
    return stack


def _synthetic_stack_2exp(ny=6, nx=8, n_bins=N_BINS):
    stack = np.zeros((ny, nx, n_bins), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            d = generate_synthetic_biexp_decay(
                n_bins=n_bins, tcspc_res=TCSPC_RES,
                tau1_ns=0.5, tau2_ns=3.0,
                a1=0.6, a2=0.4,
                bg=5.0, peak_counts=5000.0, noise=True)
            stack[i, j] = d.astype(np.float32)
    return stack


def _synthetic_stack_3exp(ny=6, nx=8, n_bins=N_BINS):
    """Tri-exponential stack: 0.3 ns / 1.5 ns / 4.0 ns."""
    stack = np.zeros((ny, nx, n_bins), dtype=np.float32)
    t = np.arange(n_bins, dtype=float) * TCSPC_RES
    irf = _make_irf(n_bins)
    irf_fft = np.fft.fft(irf)
    rng = np.random.default_rng(42)

    taus_s = [0.3e-9, 1.5e-9, 4.0e-9]
    amps   = [0.5, 0.3, 0.2]

    for i in range(ny):
        for j in range(nx):
            kernel = sum(a * np.exp(-t / tau) for a, tau in zip(amps, taus_s))
            model  = np.real(np.fft.ifft(np.fft.fft(kernel) * irf_fft))
            model  = model / model.max() * 5000.0 + 5.0
            d = rng.poisson(np.maximum(model, 0)).astype(np.float32)
            stack[i, j] = d
    return stack


def _global_popt_1exp(tau_ns=2.0):
    # [tau1, amp1, shift]
    return np.array([tau_ns * 1e-9, 1.0, 0.0])


def _global_popt_2exp(tau1_ns=0.5, tau2_ns=3.0):
    # [tau1, tau2, amp1, amp2, shift]
    return np.array([tau1_ns * 1e-9, tau2_ns * 1e-9, 0.6, 0.4, 0.0])


def _global_popt_3exp(tau1_ns=0.3, tau2_ns=1.5, tau3_ns=4.0):
    # [tau1, tau2, tau3, amp1, amp2, amp3, shift]
    return np.array([tau1_ns * 1e-9, tau2_ns * 1e-9, tau3_ns * 1e-9,
                     0.5, 0.3, 0.2, 0.0])


def _rel_err(cpu, gpu):
    """Mean relative absolute error over valid (non-NaN) pixels."""
    valid = ~(np.isnan(cpu) | np.isnan(gpu))
    if not valid.any():
        return 0.0
    denom = np.maximum(np.abs(cpu[valid]), 1e-12)
    return float(np.mean(np.abs(cpu[valid] - gpu[valid]) / denom))


@pytest.fixture(scope="module")
def gpu_backend():
    try:
        from flimkit.GPU import get_backend
        b = get_backend()
    except Exception:
        b = None
    if b is None:
        pytest.skip("No GPU backend available - CPU/GPU parity tests skipped")
    return b


def _fit_both(stack, n_exp, global_popt, gpu_backend):
    irf_prompt = _make_irf()
    kwargs = dict(
        stack       = stack,
        tcspc_res   = TCSPC_RES,
        n_bins      = N_BINS,
        irf_prompt  = irf_prompt,
        has_tail    = False,
        fit_bg      = False,
        fit_sigma   = False,
        global_popt = global_popt,
        n_exp       = n_exp,
        min_photons = MIN_PHOTONS,
    )
    cpu = fit_per_pixel(**kwargs, use_gpu=False)
    gpu = fit_per_pixel(**kwargs, gpu_backend=gpu_backend)
    return cpu, gpu


class TestCPUGPUParity1Exp:
    """1-exp: GPU grid-scan vs CPU grid-scan should agree tightly."""

    @pytest.fixture(autouse=True)
    def setup(self, gpu_backend):
        self.stack = _synthetic_stack_1exp(ny=6, nx=8, tau_ns=2.0)
        self.popt  = _global_popt_1exp(tau_ns=2.0)
        self.cpu, self.gpu = _fit_both(self.stack, 1, self.popt, gpu_backend)

    def test_intensity_identical(self):
        np.testing.assert_array_equal(
            self.cpu["intensity"], self.gpu["intensity"],
            err_msg="1-exp: intensity maps differ between CPU and GPU")

    def test_tau1_agrees(self):
        err = _rel_err(self.cpu["tau_1"], self.gpu["tau_1"])
        assert err < TAU1_TOL, (
            f"1-exp: tau_1 CPU/GPU relative error = {err:.2%} (limit {TAU1_TOL:.0%})")

    def test_tau_mean_amp_agrees(self):
        err = _rel_err(self.cpu["tau_mean_amp"], self.gpu["tau_mean_amp"])
        assert err < TAU1_TOL, (
            f"1-exp: tau_mean_amp CPU/GPU relative error = {err:.2%}")

    def test_alpha1_agrees(self):
        err = _rel_err(self.cpu["alpha_1"], self.gpu["alpha_1"])
        assert err < AMP_TOL, (
            f"1-exp: alpha_1 CPU/GPU relative error = {err:.2%}")

    def test_valid_pixel_count_matches(self):
        cpu_valid = (~np.isnan(self.cpu["tau_1"])).sum()
        gpu_valid = (~np.isnan(self.gpu["tau_1"])).sum()
        assert cpu_valid == gpu_valid, (
            f"1-exp: valid pixel count differs - CPU {cpu_valid}, GPU {gpu_valid}")

    def test_no_gpu_pixels_outside_cpu_valid_mask(self):
        cpu_valid = ~np.isnan(self.cpu["tau_1"])
        gpu_valid = ~np.isnan(self.gpu["tau_1"])
        extra = gpu_valid & ~cpu_valid
        assert not extra.any(), (
            f"1-exp: GPU has {extra.sum()} valid pixels not present in CPU result")


class TestCPUGPUParity2Exp:
    """2-exp: GPU batch_fixed_tau vs CPU NNLS pixel loop."""

    @pytest.fixture(autouse=True)
    def setup(self, gpu_backend):
        self.stack = _synthetic_stack_2exp(ny=6, nx=8)
        self.popt  = _global_popt_2exp()
        self.cpu, self.gpu = _fit_both(self.stack, 2, self.popt, gpu_backend)

    def test_intensity_identical(self):
        np.testing.assert_array_equal(
            self.cpu["intensity"], self.gpu["intensity"],
            err_msg="2-exp: intensity maps differ between CPU and GPU")

    def test_alpha1_agrees(self):
        err = _rel_err(self.cpu["alpha_1"], self.gpu["alpha_1"])
        assert err < AMP_TOL, (
            f"2-exp: alpha_1 CPU/GPU relative error = {err:.2%}")

    def test_alpha2_agrees(self):
        err = _rel_err(self.cpu["alpha_2"], self.gpu["alpha_2"])
        assert err < AMP_TOL, (
            f"2-exp: alpha_2 CPU/GPU relative error = {err:.2%}")

    def test_frac1_agrees(self):
        err = _rel_err(self.cpu["frac_1"], self.gpu["frac_1"])
        assert err < AMP_TOL, (
            f"2-exp: frac_1 CPU/GPU relative error = {err:.2%}")

    def test_tau_mean_amp_agrees(self):
        err = _rel_err(self.cpu["tau_mean_amp"], self.gpu["tau_mean_amp"])
        assert err < TAU1_TOL, (
            f"2-exp: tau_mean_amp CPU/GPU relative error = {err:.2%}")

    def test_tau_mean_int_agrees(self):
        err = _rel_err(self.cpu["tau_mean_int"], self.gpu["tau_mean_int"])
        assert err < TAU1_TOL, (
            f"2-exp: tau_mean_int CPU/GPU relative error = {err:.2%}")

    def test_chi2r_agrees(self):
        err = _rel_err(self.cpu["chi2_r"], self.gpu["chi2_r"])
        assert err < TAU1_TOL, (
            f"2-exp: chi2_r CPU/GPU relative error = {err:.2%}")

    def test_required_keys_present(self):
        required = {"intensity", "tau_mean_amp", "tau_mean_int", "chi2_r",
                    "alpha_1", "alpha_2", "frac_1", "frac_2",
                    "tau_1", "tau_2", "a1", "a2"}
        for label, maps in [("CPU", self.cpu), ("GPU", self.gpu)]:
            missing = required - maps.keys()
            assert not missing, f"2-exp {label} missing keys: {missing}"

    def test_valid_pixel_count_matches(self):
        cpu_valid = (~np.isnan(self.cpu["tau_mean_amp"])).sum()
        gpu_valid = (~np.isnan(self.gpu["tau_mean_amp"])).sum()
        assert cpu_valid == gpu_valid, (
            f"2-exp: valid pixel count differs - CPU {cpu_valid}, GPU {gpu_valid}")


class TestCPUGPUParity3Exp:
    """3-exp: GPU batch_fixed_tau vs CPU NNLS pixel loop."""

    @pytest.fixture(autouse=True)
    def setup(self, gpu_backend):
        self.stack = _synthetic_stack_3exp(ny=6, nx=8)
        self.popt  = _global_popt_3exp()
        self.cpu, self.gpu = _fit_both(self.stack, 3, self.popt, gpu_backend)

    def test_intensity_identical(self):
        np.testing.assert_array_equal(
            self.cpu["intensity"], self.gpu["intensity"],
            err_msg="3-exp: intensity maps differ between CPU and GPU")

    def test_alpha1_agrees(self):
        err = _rel_err(self.cpu["alpha_1"], self.gpu["alpha_1"])
        assert err < AMP_TOL, (
            f"3-exp: alpha_1 CPU/GPU relative error = {err:.2%}")

    def test_alpha2_agrees(self):
        err = _rel_err(self.cpu["alpha_2"], self.gpu["alpha_2"])
        assert err < AMP_TOL, (
            f"3-exp: alpha_2 CPU/GPU relative error = {err:.2%}")

    def test_alpha3_agrees(self):
        err = _rel_err(self.cpu["alpha_3"], self.gpu["alpha_3"])
        assert err < AMP_TOL, (
            f"3-exp: alpha_3 CPU/GPU relative error = {err:.2%}")

    def test_frac1_agrees(self):
        err = _rel_err(self.cpu["frac_1"], self.gpu["frac_1"])
        assert err < AMP_TOL, (
            f"3-exp: frac_1 CPU/GPU relative error = {err:.2%}")

    def test_frac2_agrees(self):
        err = _rel_err(self.cpu["frac_2"], self.gpu["frac_2"])
        assert err < AMP_TOL, (
            f"3-exp: frac_2 CPU/GPU relative error = {err:.2%}")

    def test_frac3_agrees(self):
        err = _rel_err(self.cpu["frac_3"], self.gpu["frac_3"])
        assert err < AMP_TOL, (
            f"3-exp: frac_3 CPU/GPU relative error = {err:.2%}")

    def test_tau_mean_amp_agrees(self):
        err = _rel_err(self.cpu["tau_mean_amp"], self.gpu["tau_mean_amp"])
        assert err < TAU1_TOL, (
            f"3-exp: tau_mean_amp CPU/GPU relative error = {err:.2%}")

    def test_tau_mean_int_agrees(self):
        err = _rel_err(self.cpu["tau_mean_int"], self.gpu["tau_mean_int"])
        assert err < TAU1_TOL, (
            f"3-exp: tau_mean_int CPU/GPU relative error = {err:.2%}")

    def test_chi2r_agrees(self):
        err = _rel_err(self.cpu["chi2_r"], self.gpu["chi2_r"])
        assert err < TAU1_TOL, (
            f"3-exp: chi2_r CPU/GPU relative error = {err:.2%}")

    def test_required_keys_present(self):
        required = {"intensity", "tau_mean_amp", "tau_mean_int", "chi2_r",
                    "alpha_1", "alpha_2", "alpha_3",
                    "frac_1", "frac_2", "frac_3",
                    "tau_1", "tau_2", "tau_3"}
        for label, maps in [("CPU", self.cpu), ("GPU", self.gpu)]:
            missing = required - maps.keys()
            assert not missing, f"3-exp {label} missing keys: {missing}"

    def test_valid_pixel_count_matches(self):
        cpu_valid = (~np.isnan(self.cpu["tau_mean_amp"])).sum()
        gpu_valid = (~np.isnan(self.gpu["tau_mean_amp"])).sum()
        assert cpu_valid == gpu_valid, (
            f"3-exp: valid pixel count differs - CPU {cpu_valid}, GPU {gpu_valid}")


class TestCPUGPUParityDivergenceReport:
    """
    Report per-map statistics when CPU and GPU diverge beyond the tolerance.
    These tests print a diagnostic table on failure to make debugging easier.
    """

    @pytest.fixture(autouse=True)
    def setup(self, gpu_backend):
        self.gpu_backend = gpu_backend

    def _report(self, n_exp, cpu, gpu):
        lines = [f"\n{'Map':<20} {'CPU mean':>12} {'GPU mean':>12} {'rel err':>10}"]
        for key in sorted(cpu.keys()):
            if key not in gpu:
                continue
            c, g = cpu[key], gpu[key]
            valid = ~(np.isnan(c) | np.isnan(g))
            if not valid.any():
                continue
            cmean = float(np.nanmean(c[valid]))
            gmean = float(np.nanmean(g[valid]))
            rerr  = _rel_err(c, g)
            lines.append(f"{key:<20} {cmean:>12.4f} {gmean:>12.4f} {rerr:>9.2%}")
        return "\n".join(lines)

    def test_1exp_divergence_report(self):
        stack = _synthetic_stack_1exp()
        popt  = _global_popt_1exp()
        cpu, gpu = _fit_both(stack, 1, popt, self.gpu_backend)
        err = _rel_err(cpu["tau_mean_amp"], gpu["tau_mean_amp"])
        assert err < TAU1_TOL, (
            f"1-exp tau_mean_amp divergence = {err:.2%}\n"
            + self._report(1, cpu, gpu))

    def test_2exp_divergence_report(self):
        stack = _synthetic_stack_2exp()
        popt  = _global_popt_2exp()
        cpu, gpu = _fit_both(stack, 2, popt, self.gpu_backend)
        err = _rel_err(cpu["tau_mean_amp"], gpu["tau_mean_amp"])
        assert err < TAU1_TOL, (
            f"2-exp tau_mean_amp divergence = {err:.2%}\n"
            + self._report(2, cpu, gpu))

    def test_3exp_divergence_report(self):
        stack = _synthetic_stack_3exp()
        popt  = _global_popt_3exp()
        cpu, gpu = _fit_both(stack, 3, popt, self.gpu_backend)
        err = _rel_err(cpu["tau_mean_amp"], gpu["tau_mean_amp"])
        assert err < TAU1_TOL, (
            f"3-exp tau_mean_amp divergence = {err:.2%}\n"
            + self._report(3, cpu, gpu))


# Free-τ backend parity tests
# One-component fits use the same lifetime grid scan on every path.
# Multi-component fits use the shared SciPy solver.
FREE_TAU_TOL     = 0.08
FREE_TAU_CHI2_MULT = 3.0


def _grid_step_rtol(tau_ns):
    from flimkit.FLIM.fitters import tau_grid_points
    lo = max(tau_ns / 20.0, 0.05)
    hi = min(tau_ns * 20.0, 45.0)
    return (hi / lo) ** (1.0 / (tau_grid_points() - 1)) - 1.0


def _fit_both_free_tau(stack, n_exp, global_popt, gpu_backend):
    irf_prompt = _make_irf()
    kwargs = dict(
        stack       = stack,
        tcspc_res   = TCSPC_RES,
        n_bins      = N_BINS,
        irf_prompt  = irf_prompt,
        has_tail    = False,
        fit_bg      = False,
        fit_sigma   = False,
        global_popt = global_popt,
        n_exp       = n_exp,
        min_photons = MIN_PHOTONS,
        free_tau    = True,
    )
    cpu = fit_per_pixel(**kwargs, use_gpu=False)
    gpu = fit_per_pixel(**kwargs, gpu_backend=gpu_backend)
    return cpu, gpu


class TestCPUGPUParityFreeTau:
    """Free-tau per-pixel fitting agrees across backend selection."""

    @pytest.fixture(autouse=True)
    def setup(self, gpu_backend):
        self.gpu_backend = gpu_backend

    def test_1exp_grid_scan_agrees_within_one_grid_step(self):
        stack = _synthetic_stack_1exp(ny=4, nx=4, tau_ns=2.0)
        cpu, gpu = _fit_both_free_tau(
            stack, 1, _global_popt_1exp(), self.gpu_backend)
        step = _grid_step_rtol(2.0)
        np.testing.assert_allclose(
            cpu['tau_1'], gpu['tau_1'], rtol=1.5 * step,
            err_msg=f'one exponential ignores free_tau and grid scans, so the '
                    f'two paths can land on neighbouring grid points; one step '
                    f'is {step:.2e} relative')
        np.testing.assert_allclose(cpu['chi2_r'], gpu['chi2_r'], rtol=0.05)

    # 2-exp free-tau

    def _setup_2exp(self):
        stack = _synthetic_stack_2exp(ny=8, nx=8)
        popt  = _global_popt_2exp()
        return _fit_both_free_tau(stack, 2, popt, self.gpu_backend)

    def test_2exp_free_tau_tau1_agrees(self):
        cpu, gpu = self._setup_2exp()
        err = _rel_err(cpu["tau_1"], gpu["tau_1"])
        assert err < FREE_TAU_TOL, (
            f"free-tau 2-exp: tau_1 CPU/GPU relative error = {err:.2%} "
            f"(limit {FREE_TAU_TOL:.0%})")

    def test_2exp_free_tau_tau2_agrees(self):
        cpu, gpu = self._setup_2exp()
        err = _rel_err(cpu["tau_2"], gpu["tau_2"])
        assert err < FREE_TAU_TOL, (
            f"free-tau 2-exp: tau_2 CPU/GPU relative error = {err:.2%}")

    def test_2exp_free_tau_tau_mean_amp_agrees(self):
        cpu, gpu = self._setup_2exp()
        err = _rel_err(cpu["tau_mean_amp"], gpu["tau_mean_amp"])
        assert err < FREE_TAU_TOL, (
            f"free-tau 2-exp: tau_mean_amp CPU/GPU relative error = {err:.2%}")

    def test_2exp_free_tau_chi2_not_worse(self):
        cpu, gpu = self._setup_2exp()
        valid = ~(np.isnan(cpu["chi2_r"]) | np.isnan(gpu["chi2_r"]))
        if not valid.any():
            pytest.skip("No valid pixels for chi2 comparison")
        cpu_chi2 = float(np.nanmedian(cpu["chi2_r"][valid]))
        gpu_chi2 = float(np.nanmedian(gpu["chi2_r"][valid]))
        assert gpu_chi2 <= cpu_chi2 * FREE_TAU_CHI2_MULT, (
            f"free-tau 2-exp: GPU median χ²_r = {gpu_chi2:.3f} vs "
            f"CPU {cpu_chi2:.3f} (limit = CPU × {FREE_TAU_CHI2_MULT})")

    def test_2exp_free_tau_valid_pixel_count_comparable(self):
        cpu, gpu = self._setup_2exp()
        cpu_valid = (~np.isnan(cpu["tau_mean_amp"])).sum()
        gpu_valid = (~np.isnan(gpu["tau_mean_amp"])).sum()
        assert gpu_valid >= int(cpu_valid * 0.9), (
            f"free-tau 2-exp: GPU fitted {gpu_valid} pixels vs CPU {cpu_valid} "
            f"(must be ≥ 90 % of CPU)")

    def test_2exp_free_tau_required_keys_present(self):
        cpu, gpu = self._setup_2exp()
        required = {"intensity", "tau_mean_amp", "tau_mean_int", "chi2_r",
                    "alpha_1", "alpha_2", "frac_1", "frac_2", "tau_1", "tau_2"}
        for label, maps in [("CPU", cpu), ("GPU", gpu)]:
            missing = required - maps.keys()
            assert not missing, f"free-tau 2-exp {label} missing keys: {missing}"

    # 3-exp free-tau 

    def _setup_3exp(self):
        stack = _synthetic_stack_3exp(ny=8, nx=8)
        popt  = _global_popt_3exp()
        return _fit_both_free_tau(stack, 3, popt, self.gpu_backend)

    def test_3exp_free_tau_tau_mean_amp_agrees(self):
        cpu, gpu = self._setup_3exp()
        err = _rel_err(cpu["tau_mean_amp"], gpu["tau_mean_amp"])
        assert err < FREE_TAU_TOL, (
            f"free-tau 3-exp: tau_mean_amp CPU/GPU relative error = {err:.2%}")

    def test_3exp_free_tau_tau_mean_int_agrees(self):
        cpu, gpu = self._setup_3exp()
        err = _rel_err(cpu["tau_mean_int"], gpu["tau_mean_int"])
        assert err < FREE_TAU_TOL, (
            f"free-tau 3-exp: tau_mean_int CPU/GPU relative error = {err:.2%}")

    def test_3exp_free_tau_chi2_not_worse(self):
        cpu, gpu = self._setup_3exp()
        valid = ~(np.isnan(cpu["chi2_r"]) | np.isnan(gpu["chi2_r"]))
        if not valid.any():
            pytest.skip("No valid pixels for chi2 comparison")
        cpu_chi2 = float(np.nanmedian(cpu["chi2_r"][valid]))
        gpu_chi2 = float(np.nanmedian(gpu["chi2_r"][valid]))
        assert gpu_chi2 <= cpu_chi2 * FREE_TAU_CHI2_MULT, (
            f"free-tau 3-exp: GPU median χ²_r = {gpu_chi2:.3f} vs "
            f"CPU {cpu_chi2:.3f} (limit = CPU × {FREE_TAU_CHI2_MULT})")

    def test_3exp_free_tau_valid_pixel_count_comparable(self):
        cpu, gpu = self._setup_3exp()
        cpu_valid = (~np.isnan(cpu["tau_mean_amp"])).sum()
        gpu_valid = (~np.isnan(gpu["tau_mean_amp"])).sum()
        assert gpu_valid >= int(cpu_valid * 0.9), (
            f"free-tau 3-exp: GPU fitted {gpu_valid} pixels vs CPU {cpu_valid} "
            f"(must be ≥ 90 % of CPU)")

    def test_3exp_free_tau_required_keys_present(self):
        cpu, gpu = self._setup_3exp()
        required = {"intensity", "tau_mean_amp", "tau_mean_int", "chi2_r",
                    "alpha_1", "alpha_2", "alpha_3",
                    "frac_1", "frac_2", "frac_3",
                    "tau_1", "tau_2", "tau_3"}
        for label, maps in [("CPU", cpu), ("GPU", gpu)]:
            missing = required - maps.keys()
            assert not missing, f"free-tau 3-exp {label} missing keys: {missing}"


FIT_START = 8
FIT_END = 112


def _windowed_idx(n_bins=N_BINS):
    idx = np.arange(FIT_START, FIT_END)
    return idx[(idx < 40) | (idx > 52)]


def _fit_both_windowed(stack, n_exp, global_popt, gpu_backend, fit_idx):
    irf_prompt = _make_irf()
    kwargs = dict(
        stack       = stack,
        tcspc_res   = TCSPC_RES,
        n_bins      = N_BINS,
        irf_prompt  = irf_prompt,
        has_tail    = False,
        fit_bg      = False,
        fit_sigma   = False,
        global_popt = global_popt,
        n_exp       = n_exp,
        min_photons = MIN_PHOTONS,
        fit_idx     = fit_idx,
    )
    cpu = fit_per_pixel(**kwargs, use_gpu=False)
    gpu = fit_per_pixel(**kwargs, gpu_backend=gpu_backend)
    return cpu, gpu


class TestCPUGPUParityWindowed1Exp:

    @pytest.fixture(autouse=True)
    def setup(self, gpu_backend):
        self.fit_idx = _windowed_idx()
        stack = _synthetic_stack_1exp(ny=6, nx=8, tau_ns=2.0)
        self.cpu, self.gpu = _fit_both_windowed(
            stack, 1, _global_popt_1exp(tau_ns=2.0), gpu_backend, self.fit_idx)

    def test_the_window_is_actually_narrower(self):
        assert len(self.fit_idx) < N_BINS

    def test_intensity_is_the_full_decay_not_the_window(self):
        np.testing.assert_array_equal(self.cpu['intensity'], self.gpu['intensity'])

    def test_tau_agrees(self):
        assert _rel_err(self.cpu['tau_mean_amp'], self.gpu['tau_mean_amp']) < TAU_TOL

    def test_gpu_window_changes_the_answer(self):
        stack = _synthetic_stack_1exp(ny=6, nx=8, tau_ns=2.0)
        full = fit_per_pixel(
            stack=stack, tcspc_res=TCSPC_RES, n_bins=N_BINS, irf_prompt=_make_irf(),
            has_tail=False, fit_bg=False, fit_sigma=False,
            global_popt=_global_popt_1exp(tau_ns=2.0), n_exp=1,
            min_photons=MIN_PHOTONS, use_gpu=False)
        assert not np.allclose(np.nanmedian(full['chi2_r']),
                               np.nanmedian(self.gpu['chi2_r']))


class TestCPUGPUParityWindowed2Exp:

    @pytest.fixture(autouse=True)
    def setup(self, gpu_backend):
        self.fit_idx = _windowed_idx()
        stack = _synthetic_stack_2exp(ny=6, nx=8)
        self.cpu, self.gpu = _fit_both_windowed(
            stack, 2, _global_popt_2exp(), gpu_backend, self.fit_idx)

    def test_tau_mean_amp_agrees(self):
        assert _rel_err(self.cpu['tau_mean_amp'], self.gpu['tau_mean_amp']) < TAU_TOL

    def test_tau_mean_int_agrees(self):
        assert _rel_err(self.cpu['tau_mean_int'], self.gpu['tau_mean_int']) < TAU_TOL

    def test_amplitudes_agree(self):
        assert _rel_err(self.cpu['alpha_1'], self.gpu['alpha_1']) < AMP_TOL

    def test_chi2_agrees(self):
        assert _rel_err(self.cpu['chi2_r'], self.gpu['chi2_r']) < 0.15


class TestCPUGPUParityWindowedFreeTau:

    @pytest.fixture(autouse=True)
    def setup(self, gpu_backend):
        self.fit_idx = _windowed_idx()
        stack = _synthetic_stack_1exp(ny=4, nx=4, tau_ns=2.0)
        irf_prompt = _make_irf()
        kwargs = dict(
            stack       = stack,
            tcspc_res   = TCSPC_RES,
            n_bins      = N_BINS,
            irf_prompt  = irf_prompt,
            has_tail    = False,
            fit_bg      = False,
            fit_sigma   = False,
            global_popt = _global_popt_1exp(tau_ns=2.0),
            n_exp       = 1,
            min_photons = MIN_PHOTONS,
            fit_idx     = self.fit_idx,
            free_tau    = True,
        )
        self.cpu = fit_per_pixel(**kwargs, use_gpu=False)
        self.gpu = fit_per_pixel(**kwargs, gpu_backend=gpu_backend)

    def test_tau_agrees(self):
        np.testing.assert_allclose(
            self.cpu['tau_1'], self.gpu['tau_1'], rtol=1e-6)

    def test_chi2_agrees(self):
        np.testing.assert_allclose(
            self.cpu['chi2_r'], self.gpu['chi2_r'], rtol=1e-4)

    def test_the_window_changes_the_answer(self):
        full = fit_per_pixel(
            stack=_synthetic_stack_1exp(ny=4, nx=4, tau_ns=2.0),
            tcspc_res=TCSPC_RES, n_bins=N_BINS, irf_prompt=_make_irf(),
            has_tail=False, fit_bg=False, fit_sigma=False,
            global_popt=_global_popt_1exp(tau_ns=2.0), n_exp=1,
            min_photons=MIN_PHOTONS, free_tau=True, use_gpu=False)
        assert not np.allclose(np.nanmedian(full['chi2_r']),
                               np.nanmedian(self.gpu['chi2_r']))
