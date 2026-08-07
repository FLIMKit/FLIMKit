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

def npz_session_path(ptu_path):
    from pathlib import Path
    p = Path(ptu_path)
    return p.parent / f'{p.stem}.roi_session.npz'

def web_fit_path(ptu_path):
    from pathlib import Path
    p = Path(ptu_path)
    return p.parent / f'{p.stem}.web_fit.npz'

def save_web_fit(ptu_path, res):
    g = res.get('global_summary') or {}
    payload = {}
    def _put(k, v):
        if v is not None:
            payload[k] = np.asarray(v)
    _put('time_ns', res.get('time_ns'))
    _put('decay', res.get('decay'))
    _put('model', g.get('model'))
    _put('residuals', g.get('residuals'))
    _put('irf', res.get('irf_prompt'))
    fw = g.get('fit_window_bins')
    if fw is not None:
        payload['fit_window'] = np.asarray(fw)
    if 'decay' not in payload:
        return False
    try:
        np.savez_compressed(str(web_fit_path(ptu_path)), **payload)
        return True
    except Exception:
        return False

def load_web_fit(ptu_path):
    path = web_fit_path(ptu_path)
    if not path.exists():
        return None
    try:
        data = np.load(str(path), allow_pickle=True)
    except Exception:
        return None
    if 'decay' not in data.files:
        return None
    out = {
        'time_ns': np.asarray(data['time_ns']) if 'time_ns' in data.files else None,
        'decay': np.asarray(data['decay']),
        'irf_prompt': np.asarray(data['irf']) if 'irf' in data.files else None,
        'global_summary': {},
    }
    g = out['global_summary']
    if 'model' in data.files:
        g['model'] = np.asarray(data['model'])
    if 'residuals' in data.files:
        g['residuals'] = np.asarray(data['residuals'])
    if 'fit_window' in data.files:
        g['fit_window_bins'] = tuple(int(x) for x in np.asarray(data['fit_window']).tolist())
    return out

def load_npz_session(ptu_path):
    path = npz_session_path(ptu_path)
    if not path.exists():
        return None
    try:
        data = np.load(str(path), allow_pickle=True)
    except Exception:
        return None
    out = {'path': str(path)}
    if 'fov_lifetime_map' in data.files:
        arr = data['fov_lifetime_map']
        out['lifetime_map'] = np.asarray(arr, dtype=float) if arr.ndim == 2 else None
    for k in ('fov_intensity_map', 'intensity'):
        if k in data.files and np.asarray(data[k]).ndim == 2:
            out['intensity_map'] = np.asarray(data[k], dtype=float)
            break
    rows = []
    if all(k in data.files for k in ('summary_params', 'summary_values', 'summary_units')):
        params = data['summary_params'].tolist()
        values = data['summary_values'].tolist()
        units = data['summary_units'].tolist()
        for param, val, unit in zip(params, values, units):
            rows.append({'quantity': str(param), 'value': str(val), 'unit': str(unit)})
    out['rows'] = rows
    if 'fov_n_exp' in data.files:
        try:
            out['n_exp'] = int(data['fov_n_exp'])
        except Exception:
            pass
    out['res'] = _res_from_npz(data)
    return out if (out.get('rows') or out.get('lifetime_map') is not None
                   or out.get('res') is not None) else None

def _res_from_npz(data):
    import json
    files = set(data.files)
    if 'decay' not in files or 'time_ns' not in files:
        return None
    res = {
        'time_ns': np.asarray(data['time_ns'], dtype=float),
        'decay': np.asarray(data['decay'], dtype=float),
    }
    if 'irf_prompt' in files:
        res['irf_prompt'] = np.asarray(data['irf_prompt'], dtype=float)
    g = {}
    if 'global_summary_arr_model' in files:
        g['model'] = np.asarray(data['global_summary_arr_model'], dtype=float)
    if 'global_summary_arr_residuals' in files:
        g['residuals'] = np.asarray(data['global_summary_arr_residuals'], dtype=float)
    if 'global_summary_json' in files:
        try:
            raw = data['global_summary_json']
            raw = raw.item() if raw.ndim == 0 else raw
            gj = json.loads(str(raw))
            fw = gj.get('fit_window_bins')
            if fw is not None:
                g['fit_window_bins'] = tuple(int(x) for x in fw)
        except Exception:
            pass
    res['global_summary'] = g
    return res

def intensity_map(ptu_path, channel=None):
    ptu = FLIMFile(ptu_path, verbose=False)
    stack = ptu.raw_pixel_stack(channel=channel if channel is not None else ptu.photon_channel)
    return stack.sum(axis=-1)

def summed_decay(ptu_path, channel=None):
    ptu = FLIMFile(ptu_path, verbose=False)
    stack = ptu.raw_pixel_stack(channel=channel if channel is not None else ptu.photon_channel)
    decay = stack.sum(axis=(0, 1)).astype(float)
    t = np.arange(len(decay), dtype=float) * ptu.tcspc_res * 1e9
    return t, decay
