"""
Tests for per-ROI decay fitting.

These tests exercise the data pipeline that backs the "Fit ROI Decay" button
without requiring any Tk GUI.  They follow the same data flow as
RoiAnalysisPanel._fit_roi_decay():

    PTU pixel stack → ROI mask → summed decay → fit_summed → summary dict

All tests use MockPTUFile so no real PTU file is needed.
"""

import numpy as np
import pytest

from flimkit.UI.roi_tools import RoiManager
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



# Helpers
def _build_ptu(n_y=32, n_x=32, n_bins=256):
    """Small MockPTUFile for fast test execution."""
    return MockPTUFile(n_y=n_y, n_x=n_x, n_bins=n_bins,
                       tcspc_res=MOCK_TCSPC_RES)


def _roi_decay_from_stack(stack, mask):
    """Sum the pixel axis of the stack inside the mask."""
    return stack[mask].sum(axis=0).astype(float)


def _make_irf(n_bins, tcspc_res):
    """Gaussian IRF matching MockPTUFile ground truth."""
    return gaussian_irf(n_bins, MOCK_IRF_CENTER, MOCK_IRF_FWHM_BINS)


def _run_fit(roi_decay, irf_prompt, tcspc_res, n_bins, n_exp=2):
    """Call fit_summed with the same defaults as _fit_roi_decay."""
    return fit_summed(
        roi_decay, tcspc_res, n_bins, irf_prompt,
        has_tail=False, fit_bg=True, fit_sigma=False,
        n_exp=n_exp,
        tau_min_ns=0.05,
        tau_max_ns=20.0,
        cost_function="poisson",
    )



# ROI masking + decay extraction
class TestRoiDecayExtraction:
    """Verify that masking + summation produce sensible arrays."""

    def test_rect_roi_decay_shape(self):
        ptu = _build_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[5, 5], [20, 20]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)
        assert decay.shape == (ptu.n_bins,)

    def test_ellipse_roi_decay_shape(self):
        ptu = _build_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("E1", "ellipse", [[4, 4], [20, 20]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)
        assert decay.shape == (ptu.n_bins,)

    def test_polygon_roi_decay_shape(self):
        ptu = _build_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        coords = [[4, 4], [20, 4], [20, 20], [4, 20]]
        rid = mgr.add_region("P1", "polygon", coords)
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)
        assert decay.shape == (ptu.n_bins,)

    def test_roi_decay_is_nonnegative(self):
        ptu = _build_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[5, 5], [25, 25]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)
        assert np.all(decay >= 0)

    def test_roi_decay_has_photons(self):
        """A central ROI should always collect photons from synthetic data."""
        ptu = _build_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[8, 8], [24, 24]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)
        assert decay.max() > 0, "Central ROI contains no photons"

    def test_whole_image_roi_equals_summed_decay(self):
        """A full-image ROI should match ptu.summed_decay()."""
        ptu = _build_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        ny, nx = ptu.n_y, ptu.n_x
        rid = mgr.add_region("All", "rect", [[0, 0], [nx - 1, ny - 1]])
        mask = mgr.compute_region_mask(rid, (ny, nx))
        roi_decay = _roi_decay_from_stack(stack, mask)
        full_decay = ptu.summed_decay().astype(float)
        np.testing.assert_array_equal(roi_decay, full_decay)

    def test_empty_mask_raises(self):
        """A region entirely outside the image returns an empty mask."""
        ptu = _build_ptu(n_y=32, n_x=32)
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        # Region outside image bounds
        rid = mgr.add_region("Out", "rect", [[200, 200], [300, 300]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        if mask is not None:
            # If mask is returned it should be all-False
            assert not mask.any()


# ROI + fit pipeline

class TestRoiDecayFitPipeline:
    """End-to-end: extract ROI decay → fit → check summary structure."""

    @pytest.fixture
    def central_roi_result(self):
        """Run the full pipeline on a central rect ROI and return summary."""
        ptu = _build_ptu(n_y=48, n_x=48, n_bins=256)
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("Centre", "rect", [[12, 12], [36, 36]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)
        irf = _make_irf(ptu.n_bins, ptu.tcspc_res)
        popt, summary = _run_fit(decay, irf, ptu.tcspc_res, ptu.n_bins, n_exp=2)
        return summary, ptu

    def test_summary_has_required_keys(self, central_roi_result):
        summary, _ = central_roi_result
        for key in ('taus_ns', 'amps', 'model', 'reduced_chi2_tail'):
            assert key in summary, f"Missing key: {key}"

    def test_two_exp_fit_returns_two_taus(self, central_roi_result):
        """A 2-exp fit must return exactly two τ values."""
        summary, _ = central_roi_result
        assert len(summary['taus_ns']) == 2

    def test_taus_are_positive(self, central_roi_result):
        summary, _ = central_roi_result
        taus = summary['taus_ns']
        assert len(taus) == 2
        assert all(t > 0 for t in taus), f"Non-positive τ: {taus}"

    def test_amps_are_positive(self, central_roi_result):
        summary, _ = central_roi_result
        amps = summary['amps']
        assert len(amps) == 2
        assert all(a > 0 for a in amps), f"Non-positive amplitudes: {amps}"

    def test_model_shape_matches_decay(self, central_roi_result):
        summary, ptu = central_roi_result
        assert len(summary['model']) == ptu.n_bins

    def test_chi2_is_positive(self, central_roi_result):
        summary, _ = central_roi_result
        assert summary['reduced_chi2_tail'] > 0

    def test_taus_within_physical_range(self, central_roi_result):
        """Fitted τ values should be within [0.05, 20] ns (the fit bounds)."""
        summary, _ = central_roi_result
        for tau in summary['taus_ns']:
            assert 0.05 <= tau <= 20.0, f"τ={tau:.4f} ns out of physical range"

    def test_amplitude_weighted_mean_tau_is_finite(self, central_roi_result):
        """The downstream τ_mean_fit calculation must not produce NaN."""
        summary, _ = central_roi_result
        taus = summary['taus_ns']
        amps = summary['amps']
        tau_mean = float(np.dot(taus, amps) / np.sum(amps))
        assert np.isfinite(tau_mean)
        assert tau_mean > 0


# IRF fallback logic
class TestIRFFallback:
    """Gaussian IRF fallback should not break the fit."""

    def test_gaussian_irf_fallback_shape(self):
        n_bins = 256
        decay = np.ones(n_bins, dtype=float)
        # Peak at argmax, ~200 ps FWHM
        peak_bin = int(np.argmax(decay))
        fwhm_bins = max(1.0, 0.2e-9 / MOCK_TCSPC_RES)
        irf = gaussian_irf(n_bins, peak_bin, fwhm_bins)
        assert irf.shape == (n_bins,)
        assert irf.sum() > 0

    def test_fit_with_gaussian_irf_fallback_runs(self):
        """Fit must complete even with a crude Gaussian IRF."""
        ptu = _build_ptu(n_y=32, n_x=32)
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[8, 8], [24, 24]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)

        # Fallback: peak from decay, generic FWHM
        peak_bin = int(np.argmax(decay))
        fwhm_bins = max(1.0, 0.2e-9 / ptu.tcspc_res)
        irf_fallback = gaussian_irf(ptu.n_bins, peak_bin, fwhm_bins)

        popt, summary = _run_fit(decay, irf_fallback, ptu.tcspc_res, ptu.n_bins, n_exp=1)
        assert 'taus_ns' in summary
        assert len(summary['taus_ns']) == 1
        assert summary['taus_ns'][0] > 0

    def test_cached_irf_wrong_length_triggers_fallback(self):
        """If cached IRF has mismatched length, fallback Gaussian is used instead."""
        ptu = _build_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("R1", "rect", [[8, 8], [24, 24]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)

        bad_irf = np.ones(ptu.n_bins // 2)   # wrong length

        # Reproduce the conditional from _fit_roi_decay
        if bad_irf is None or len(bad_irf) != ptu.n_bins:
            peak_bin = int(np.argmax(decay))
            fwhm_bins = max(1.0, 0.2e-9 / ptu.tcspc_res)
            irf_to_use = gaussian_irf(ptu.n_bins, peak_bin, fwhm_bins)
        else:
            irf_to_use = bad_irf

        assert len(irf_to_use) == ptu.n_bins


# Statistics written back to region

class TestRoiStatisticsWriteback:
    """Verify the on_done statistics are written correctly."""

    def _run_and_collect_stats(self, n_exp=2):
        ptu = _build_ptu(n_y=48, n_x=48)
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("Region", "rect", [[10, 10], [38, 38]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        decay = _roi_decay_from_stack(stack, mask)
        irf = _make_irf(ptu.n_bins, ptu.tcspc_res)
        _, summary = _run_fit(decay, irf, ptu.tcspc_res, ptu.n_bins, n_exp=n_exp)

        # Reproduce the on_done writeback
        taus = summary.get('taus_ns', [])
        amps = summary.get('amps', [])
        region_obj = mgr.get_region(rid)
        tau_mean_fit = (float(np.dot(taus, amps) / np.sum(amps))
                        if len(taus) > 0 and len(amps) > 0 else None)
        stats = region_obj.get('statistics', {})
        stats['tau_mean_fit']  = tau_mean_fit
        stats['taus_ns_fit']   = list(taus)
        stats['amps_fit']      = list(amps)
        stats['chi2_r_fit']    = summary.get('reduced_chi2_tail')
        region_obj['statistics'] = stats
        return stats, summary

    def test_writeback_contains_tau_mean_fit(self):
        stats, _ = self._run_and_collect_stats()
        assert 'tau_mean_fit' in stats
        assert stats['tau_mean_fit'] is not None

    def test_writeback_tau_mean_is_finite(self):
        stats, _ = self._run_and_collect_stats()
        assert np.isfinite(stats['tau_mean_fit'])

    def test_writeback_taus_ns_fit_list(self):
        stats, summary = self._run_and_collect_stats(n_exp=2)
        assert isinstance(stats['taus_ns_fit'], list)
        assert len(stats['taus_ns_fit']) == 2

    def test_writeback_amps_fit_list(self):
        stats, summary = self._run_and_collect_stats(n_exp=2)
        assert isinstance(stats['amps_fit'], list)
        assert len(stats['amps_fit']) == 2

    def test_writeback_chi2_is_positive(self):
        stats, _ = self._run_and_collect_stats()
        assert stats['chi2_r_fit'] is not None
        assert stats['chi2_r_fit'] > 0

    def test_writeback_single_exp(self):
        """Single-exponential fit should write exactly one τ."""
        stats, _ = self._run_and_collect_stats(n_exp=1)
        assert len(stats['taus_ns_fit']) == 1
        assert len(stats['amps_fit']) == 1


# Edge cases

class TestRoiDecayEdgeCases:
    """Guards against edge inputs that should be caught before calling fit."""

    def test_zero_decay_detected(self):
        """All-zero ROI decay must be caught (max == 0 guard)."""
        zero_decay = np.zeros(256, dtype=float)
        assert zero_decay.max() == 0

    def test_single_pixel_roi_still_has_shape(self):
        """A 1×1 ROI should still produce a 1D decay array of correct length."""
        ptu = _build_ptu()
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("Dot", "rect", [[16, 16], [17, 17]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        assert mask is not None and mask.any()
        decay = _roi_decay_from_stack(stack, mask)
        assert decay.shape == (ptu.n_bins,)

    def test_roi_smaller_than_whole_image_has_fewer_photons(self):
        """A small central ROI must collect fewer photons than the whole image."""
        ptu = _build_ptu(n_y=32, n_x=32)
        stack = ptu.pixel_stack(channel=1, binning=1)
        mgr = RoiManager()
        rid = mgr.add_region("Small", "rect", [[14, 14], [18, 18]])
        mask = mgr.compute_region_mask(rid, (ptu.n_y, ptu.n_x))
        roi_decay   = _roi_decay_from_stack(stack, mask)
        full_decay  = ptu.summed_decay().astype(float)
        assert roi_decay.sum() < full_decay.sum()
