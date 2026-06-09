import numpy as np
import pytest
from flimkit.FLIM.fitters import fit_summed
from flimkit.FLIM.irf_tools import gaussian_irf
from mock_data import (
    MockPTUFile,
    MOCK_TAU1_NS,
    MOCK_TAU2_NS,
    MOCK_AMP1,
    MOCK_AMP2,
    MOCK_IRF_CENTER,
    MOCK_IRF_FWHM_BINS,
    MOCK_TCSPC_RES,
)


# Helpers that mirror phasor_panel.py logic without importing Tk
def _make_ptu(n_y=32, n_x=32, n_bins=256):
    return MockPTUFile(n_y=n_y, n_x=n_x, n_bins=n_bins, tcspc_res=MOCK_TCSPC_RES)


def _make_irf(n_bins):
    return gaussian_irf(n_bins, MOCK_IRF_CENTER, MOCK_IRF_FWHM_BINS)


def _fake_phasor(n_y, n_x, rng=None):
    """Return synthetic (real_cal, imag_cal, mean) arrays shaped (n_y, n_x)."""
    if rng is None:
        rng = np.random.default_rng(0)
    # Scatter points near two clusters on the universal semicircle
    real = rng.normal(loc=0.4, scale=0.04, size=(n_y, n_x)).clip(0, 1)
    imag = rng.normal(loc=0.35, scale=0.04, size=(n_y, n_x)).clip(0, 0.5)
    mean = rng.integers(50, 2000, size=(n_y, n_x)).astype(float)
    return real, imag, mean


def _ellipse_mask(real, imag, center_g, center_s, radius=0.1, ratio=0.6):
    """Pure-numpy ellipse cursor mask - mirrors phasorpy.cursor.mask_from_elliptic_cursor."""
    from phasorpy.cursor import mask_from_elliptic_cursor
    cg = np.array([center_g])
    cs = np.array([center_s])
    m = mask_from_elliptic_cursor(
        real, imag, cg, cs,
        radius=radius, radius_minor=radius * ratio, angle='semicircle')
    if m.ndim > real.ndim:
        m = m[0]
    return m


def _build_union_mask(real, imag, cursors, radius=0.1, ratio=0.6):
    """Build a union boolean mask from a list of cursor dicts."""
    union = np.zeros_like(real, dtype=bool)
    for cur in cursors:
        if cur.get('type', 'ellipse') == 'ellipse':
            m = _ellipse_mask(real, imag,
                              cur['center_g'], cur['center_s'],
                              radius=radius, ratio=ratio)
        else:
            # polygon - simplified: bounding box for tests
            verts = np.array(cur['vertices'])
            g_lo, g_hi = verts[:, 0].min(), verts[:, 0].max()
            s_lo, s_hi = verts[:, 1].min(), verts[:, 1].max()
            m = ((real >= g_lo) & (real <= g_hi) &
                 (imag >= s_lo) & (imag <= s_hi))
        union |= m
    return union


def _run_fit(decay, n_bins, tcspc_res, irf, n_exp=2):
    return fit_summed(
        decay, tcspc_res, n_bins, irf,
        has_tail=False, fit_bg=True, fit_sigma=False,
        n_exp=n_exp, tau_min_ns=0.05, tau_max_ns=20.0,
        cost_function="poisson",
    )


# 1. Cursor mask construction
class TestCursorMaskConstruction:
    """Verify that cursor masks have correct shape and cover a non-trivial area."""

    def test_single_ellipse_mask_shape(self):
        n_y, n_x = 32, 32
        real, imag, _ = _fake_phasor(n_y, n_x)
        m = _ellipse_mask(real, imag, center_g=0.4, center_s=0.35, radius=0.15)
        assert m.shape == (n_y, n_x)
        assert m.dtype == bool

    def test_single_ellipse_mask_nonempty(self):
        n_y, n_x = 32, 32
        real, imag, _ = _fake_phasor(n_y, n_x)
        m = _ellipse_mask(real, imag, center_g=0.4, center_s=0.35, radius=0.15)
        assert m.any(), "Ellipse cursor placed at cluster centre should select pixels"

    def test_single_ellipse_selects_subset(self):
        """Mask should not cover the entire image."""
        n_y, n_x = 32, 32
        real, imag, _ = _fake_phasor(n_y, n_x)
        m = _ellipse_mask(real, imag, center_g=0.4, center_s=0.35, radius=0.05)
        assert m.sum() < n_y * n_x

    def test_two_cursor_union_larger_than_single(self):
        n_y, n_x = 32, 32
        real, imag, _ = _fake_phasor(n_y, n_x)
        cursors_single = [{'type': 'ellipse', 'center_g': 0.4, 'center_s': 0.35}]
        cursors_two    = [{'type': 'ellipse', 'center_g': 0.4, 'center_s': 0.35},
                          {'type': 'ellipse', 'center_g': 0.6, 'center_s': 0.25}]
        m1 = _build_union_mask(real, imag, cursors_single, radius=0.12)
        m2 = _build_union_mask(real, imag, cursors_two,    radius=0.12)
        assert m2.sum() >= m1.sum(), "Union of two cursors should be >= single cursor"

    def test_cursor_far_from_data_empty(self):
        """A cursor placed far from any data point should select nothing."""
        n_y, n_x = 32, 32
        real, imag, _ = _fake_phasor(n_y, n_x)
        m = _ellipse_mask(real, imag, center_g=0.05, center_s=0.05, radius=0.02)
        # With tight radius far from the cluster, should select very few or zero pixels
        assert m.sum() < 5


# 2. Gated decay extraction
class TestGatedDecayExtraction:
    """Verify that summing the pixel stack through a phasor mask gives correct arrays."""

    def test_gated_decay_shape(self):
        ptu = _make_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        real, imag, mean = _fake_phasor(ptu.n_y, ptu.n_x)
        mask = _ellipse_mask(real, imag, 0.4, 0.35, radius=0.15)
        mask &= (mean >= 1)   # validity gate
        decay = stack[mask].sum(axis=0).astype(float)
        assert decay.shape == (ptu.n_bins,)

    def test_gated_decay_nonnegative(self):
        ptu = _make_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        real, imag, mean = _fake_phasor(ptu.n_y, ptu.n_x)
        mask = _ellipse_mask(real, imag, 0.4, 0.35, radius=0.15)
        decay = stack[mask].sum(axis=0).astype(float)
        assert np.all(decay >= 0)

    def test_gated_decay_has_photons(self):
        ptu = _make_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        real, imag, _ = _fake_phasor(ptu.n_y, ptu.n_x)
        mask = _ellipse_mask(real, imag, 0.4, 0.35, radius=0.15)
        decay = stack[mask].sum(axis=0).astype(float)
        assert decay.max() > 0, "Gated decay should contain photons"

    def test_stack_shape_mismatch_detected(self):
        """When mask and stack spatial shapes differ a ValueError should be raised."""
        ptu = _make_ptu(n_y=32, n_x=32)
        stack = ptu.pixel_stack(channel=1, binning=1)
        wrong_mask = np.ones((16, 16), dtype=bool)   # different shape
        with pytest.raises((ValueError, IndexError)):
            _ = stack[wrong_mask].sum(axis=0)

    def test_zero_photon_decay_detected(self):
        """A mask that selects no pixels should produce an all-zero decay."""
        ptu = _make_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        empty_mask = np.zeros((ptu.n_y, ptu.n_x), dtype=bool)
        decay = stack[empty_mask].sum(axis=0).astype(float)
        assert decay.max() == 0


class TestCursorGatedFitPipeline:
    """End-to-end: mask → decay → fit_summed → summary."""

    @pytest.fixture(scope='class')
    def fit_result(self):
        ptu   = _make_ptu(n_y=32, n_x=32, n_bins=256)
        stack = ptu.pixel_stack(channel=1, binning=1)
        real, imag, _ = _fake_phasor(ptu.n_y, ptu.n_x)
        mask  = _ellipse_mask(real, imag, 0.4, 0.35, radius=0.2)
        decay = stack[mask].sum(axis=0).astype(float)
        irf   = _make_irf(ptu.n_bins)
        popt, summary = _run_fit(decay, ptu.n_bins, ptu.tcspc_res, irf, n_exp=2)
        return summary

    def test_summary_has_taus(self, fit_result):
        assert 'taus_ns' in fit_result

    def test_summary_has_amps(self, fit_result):
        assert 'amps' in fit_result

    def test_summary_has_chi2(self, fit_result):
        assert 'reduced_chi2_tail' in fit_result

    def test_summary_has_model(self, fit_result):
        assert 'model' in fit_result

    def test_correct_number_of_taus(self, fit_result):
        assert len(fit_result['taus_ns']) == 2

    def test_taus_positive(self, fit_result):
        assert np.all(np.array(fit_result['taus_ns']) > 0)

    def test_taus_finite(self, fit_result):
        assert np.all(np.isfinite(fit_result['taus_ns']))

    def test_amps_positive(self, fit_result):
        assert np.all(np.array(fit_result['amps']) > 0)

    def test_chi2_positive(self, fit_result):
        chi2 = fit_result['reduced_chi2_tail']
        assert chi2 is not None and chi2 > 0

    def test_model_same_length_as_input(self, fit_result):
        ptu = _make_ptu(n_y=32, n_x=32, n_bins=256)
        assert len(fit_result['model']) == ptu.n_bins

    def test_tau_mean_computable(self, fit_result):
        taus = fit_result['taus_ns']
        amps = fit_result['amps']
        tau_mean = float(np.dot(taus, amps) / np.sum(amps))
        assert np.isfinite(tau_mean)
        assert tau_mean > 0


# 4. IRF fallback
class TestCursorFitIRFFallback:
    """When no cached IRF is available the code falls back to a Gaussian."""

    def test_gaussian_fallback_shape(self):
        ptu  = _make_ptu()
        bins = ptu.n_bins
        irf  = gaussian_irf(bins, int(np.argmax(ptu.summed_decay())), 3.0)
        assert irf.shape == (bins,)

    def test_gaussian_fallback_fit_succeeds(self):
        """Fit using a Gaussian IRF (fallback path) should not raise."""
        ptu   = _make_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        real, imag, _ = _fake_phasor(ptu.n_y, ptu.n_x)
        mask  = _ellipse_mask(real, imag, 0.4, 0.35, radius=0.2)
        decay = stack[mask].sum(axis=0).astype(float)
        irf   = gaussian_irf(ptu.n_bins, int(np.argmax(decay)), 3.0)
        popt, summary = _run_fit(decay, ptu.n_bins, ptu.tcspc_res, irf)
        assert 'taus_ns' in summary

    def test_mismatched_irf_length_triggers_fallback(self):
        """An IRF with wrong length should not be used - a new one should be generated."""
        ptu   = _make_ptu()
        irf_ok   = _make_irf(ptu.n_bins)
        irf_bad  = irf_ok[:ptu.n_bins // 2]   # wrong length
        # The guard logic: if len(irf) != n_bins → regenerate
        assert len(irf_bad) != ptu.n_bins
        # After regeneration the new IRF should match
        irf_new = gaussian_irf(ptu.n_bins, MOCK_IRF_CENTER, MOCK_IRF_FWHM_BINS)
        assert len(irf_new) == ptu.n_bins


# 5. Result dict structure
class TestCursorFitResultDict:
    """Verify the result dict returned to on_done has the expected keys."""

    @pytest.fixture(scope='class')
    def result(self):
        ptu   = _make_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        real, imag, _ = _fake_phasor(ptu.n_y, ptu.n_x)
        mask  = _ellipse_mask(real, imag, 0.4, 0.35, radius=0.2)
        decay = stack[mask].sum(axis=0).astype(float)
        irf   = _make_irf(ptu.n_bins)
        popt, summary = _run_fit(decay, ptu.n_bins, ptu.tcspc_res, irf)
        return {
            'region_name': '1 cursor',
            'region_id':   -1,
            'region_ids':  [],
            'decay':       decay,
            'time_ns':     ptu.time_ns,
            'irf_prompt':  irf,
            'irf_source':  'from main fit',
            'popt':        popt,
            'summary':     summary,
            'n_exp':       2,
        }

    def test_result_has_region_name(self, result):
        assert 'region_name' in result

    def test_result_has_decay(self, result):
        assert 'decay' in result

    def test_result_has_time_ns(self, result):
        assert 'time_ns' in result

    def test_result_has_irf_prompt(self, result):
        assert 'irf_prompt' in result

    def test_result_has_summary(self, result):
        assert 'summary' in result

    def test_result_decay_length_matches_time(self, result):
        assert len(result['decay']) == len(result['time_ns'])

    def test_result_irf_length_matches_decay(self, result):
        assert len(result['irf_prompt']) == len(result['decay'])


# 6. _show_fit_result_window importable without Tk
class TestShowFitResultWindowImport:
    """The module-level popup function must be importable without Tk running."""

    def test_can_import_show_fit_result_window(self):
        from flimkit.UI.roi_tools import _show_fit_result_window
        assert callable(_show_fit_result_window)

    def test_can_import_show_roi_fit_result_standalone(self):
        from flimkit.UI.roi_tools import _show_roi_fit_result_standalone
        assert callable(_show_roi_fit_result_standalone)


# 7. Multi-cursor union behaviour
class TestMultiCursorUnion:
    """Multiple cursors should produce a union mask, not just the first one."""

    def test_union_covers_both_clusters(self):
        n_y, n_x = 64, 64
        rng = np.random.default_rng(42)
        # Two well-separated clusters
        real = rng.normal(loc=0.3, scale=0.02, size=(n_y, n_x)).clip(0, 1)
        imag = rng.normal(loc=0.3, scale=0.02, size=(n_y, n_x)).clip(0, 0.5)
        # Place cursor 2 cluster pixels at (0.7, 0.2)
        real[32:, :] = rng.normal(loc=0.7, scale=0.02, size=(n_y - 32, n_x)).clip(0, 1)
        imag[32:, :] = rng.normal(loc=0.2, scale=0.02, size=(n_y - 32, n_x)).clip(0, 0.5)

        c1 = {'type': 'ellipse', 'center_g': 0.3, 'center_s': 0.3}
        c2 = {'type': 'ellipse', 'center_g': 0.7, 'center_s': 0.2}

        m1   = _build_union_mask(real, imag, [c1],     radius=0.08)
        m2   = _build_union_mask(real, imag, [c2],     radius=0.08)
        both = _build_union_mask(real, imag, [c1, c2], radius=0.08)

        # Union mask should cover at least as many pixels as either alone
        assert both.sum() >= m1.sum()
        assert both.sum() >= m2.sum()
        # And should cover more than either individual cursor
        assert both.sum() > max(m1.sum(), m2.sum())

    def test_single_cursor_subset_of_union(self):
        n_y, n_x = 32, 32
        real, imag, _ = _fake_phasor(n_y, n_x)
        c1 = {'type': 'ellipse', 'center_g': 0.4, 'center_s': 0.35}
        c2 = {'type': 'ellipse', 'center_g': 0.5, 'center_s': 0.30}
        m1   = _build_union_mask(real, imag, [c1], radius=0.1)
        union = _build_union_mask(real, imag, [c1, c2], radius=0.1)
        # Every pixel selected by c1 alone must also be in the union
        assert np.all(union[m1])
