import numpy as np
from flimkit.formats import FLIMFile
from flimkit.FLIM.fitters import fit_summed
from flimkit.FLIM.irf_tools import gaussian_irf

def boxes_to_mask(boxes, shape):
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        xa, xb = sorted((int(round(x0)), int(round(x1))))
        ya, yb = sorted((int(round(y0)), int(round(y1))))
        xa = max(0, xa)
        ya = max(0, ya)
        xb = min(nx - 1, xb)
        yb = min(ny - 1, yb)
        if xb >= xa and yb >= ya:
            mask[ya:yb + 1, xa:xb + 1] = True
    return mask

def fit_roi(ptu_path, boxes, params, irf_cached=None):
    ptu = FLIMFile(ptu_path, verbose=False)
    n_bins = ptu.n_bins
    tcspc_res = ptu.tcspc_res
    stack = ptu.pixel_stack(channel=params.get('channel'), binning=1)
    shape = (stack.shape[0], stack.shape[1])
    mask = boxes_to_mask(boxes, shape)
    if not mask.any():
        raise ValueError('ROI mask is empty - draw a box inside the image.')
    roi_decay = stack[mask].sum(axis=0).astype(float)
    if roi_decay.max() == 0:
        raise ValueError('ROI contains no photons.')
    if irf_cached is not None and len(irf_cached) == n_bins:
        irf_prompt = irf_cached
        irf_source = 'from main fit'
    else:
        decay_peak = int(np.argmax(roi_decay))
        fwhm_bins = max(1.0, 0.2e-9 / tcspc_res)
        irf_prompt = gaussian_irf(n_bins, decay_peak, fwhm_bins)
        irf_source = 'gaussian (no IRF cached)'
    popt, summary = fit_summed(
        roi_decay, tcspc_res, n_bins, irf_prompt,
        has_tail=False, fit_bg=True, fit_sigma=False,
        n_exp=int(params.get('nexp', 2)),
        tau_min_ns=float(params.get('tau_min', 0.145)),
        tau_max_ns=float(params.get('tau_max', 45.0)),
        cost_function=params.get('cost_function', 'poisson'),
    )
    summary['irf_source'] = irf_source
    summary['n_pixels'] = int(mask.sum())
    return roi_decay, summary

def intensity_map(ptu_path, channel=None):
    ptu = FLIMFile(ptu_path, verbose=False)
    stack = ptu.raw_pixel_stack(channel=channel if channel is not None else ptu.photon_channel)
    return stack.sum(axis=-1)
