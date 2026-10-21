import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm

def load_zstack_display_slices(group_dir, ptu_dir=None, region=None):
    from flimkit.utils.batch_fit import group_zstack_files
    group_dir = Path(group_dir)
    ref = {}
    ref_files = list(group_dir.glob('*_reference_fit.json'))
    if ref_files:
        try:
            ref = json.loads(ref_files[0].read_text())
        except Exception:
            ref = {}
    taus_ns = ref.get('taus_ns', [])
    nexp = ref.get('nexp', len(taus_ns))
    ref_decay = ref_time = ref_model = ref_irf = None
    ref_chi2 = None
    ref_calibrated = ref.get('calibrated_chi2_pearson')
    ref_calibrated_tail = ref.get('calibrated_chi2_tail_pearson')
    npz_path = group_dir / 'reference_decay.npz'
    if npz_path.exists():
        try:
            with np.load(str(npz_path)) as zf:
                ref_decay = zf['decay']
                ref_time = zf['time_ns']
                ref_model = zf['model'] if zf['model'].size > 0 else None
                ref_irf = zf['irf_prompt'] if zf['irf_prompt'].size > 0 else None
                ref_chi2 = float(zf['reduced_chi2_tail'][0]) if zf['reduced_chi2_tail'].size > 0 else None
                if ref_calibrated is None and 'calibrated_chi2_pearson' in zf:
                    ref_calibrated = float(zf['calibrated_chi2_pearson'][0])
                if ref_calibrated_tail is None and 'calibrated_chi2_tail_pearson' in zf:
                    ref_calibrated_tail = float(zf['calibrated_chi2_tail_pearson'][0])
        except Exception:
            pass
    z_to_ptu = {}
    if ptu_dir is not None and Path(ptu_dir).is_dir():
        groups = group_zstack_files(ptu_dir)
        for (r, _t, _s), zsl in groups.items():
            if region is None or r == region:
                z_to_ptu = {z: str(p) for z, p in zsl.items()}
                break
    slices = []
    for slice_dir in sorted(group_dir.glob('z[0-9]*')):
        if not slice_dir.is_dir():
            continue
        try:
            z = int(slice_dir.name[1:])
        except ValueError:
            continue
        def _load(name):
            p = slice_dir / f'{name}.npy'
            return np.load(str(p)) if p.exists() else None
        pixel_maps = {}
        for k in ('tau_mean_int', 'tau_mean_amp', 'alpha_1', 'alpha_2', 'alpha_3',
                  'chi2_r', 'calibrated_chi2_r'):
            m = _load(k)
            if m is not None:
                pixel_maps[k] = m
        global_summary = {'taus_ns': list(taus_ns), 'n_exp': nexp}
        if ref_model is not None:
            global_summary['model'] = ref_model
        if ref_chi2 is not None and ref_chi2 == ref_chi2:
            global_summary['reduced_chi2_tail'] = ref_chi2
        if ref_calibrated is not None and ref_calibrated == ref_calibrated:
            global_summary['calibrated_chi2_pearson'] = ref_calibrated
        if ref_calibrated_tail is not None and ref_calibrated_tail == ref_calibrated_tail:
            global_summary['calibrated_chi2_tail_pearson'] = ref_calibrated_tail
        fit_result = {
            'pixel_maps': pixel_maps,
            'intensity': _load('intensity'),
            'global_summary': global_summary,
        }
        if ref_decay is not None and ref_time is not None:
            fit_result['decay'] = ref_decay
            fit_result['time_ns'] = ref_time
        if ref_irf is not None:
            fit_result['irf_prompt'] = ref_irf
        slices.append({'z': z, 'ptu_path': z_to_ptu.get(z), 'fit_result': fit_result})
    slices.sort(key=lambda d: d['z'])
    return slices

COLORMAPS = {
    'viridis': 'viridis',
    'turbo': 'turbo',
    'jet': 'jet',
    'rainbow': 'rainbow',
    'spectral': 'Spectral_r',
    'plasma': 'plasma',
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    'coolwarm': 'coolwarm',
    'cool': 'cool',
    'hot': 'hot',
    'hsv': 'hsv',
    'twilight': 'twilight',
}

_OPEN_SPAN = {'hsv': (0.0, 0.8), 'twilight': (0.0, 0.5)}

INTENSITY_COLORMAPS = ['inferno', 'gray', 'magma', 'viridis', 'hot', 'bone', 'cividis', 'plasma']

def compute_weighted_lifetime(
    pixel_maps,
    intensity,
    n_exp=2,
    weighting='amplitude',
) -> np.ndarray:
    primary = 'tau_mean_int' if weighting == 'intensity' else 'tau_mean_amp'
    fallback = 'tau_mean_amp' if weighting == 'intensity' else 'tau_mean_int'
    for key in (primary, fallback):
        if key in pixel_maps:
            arr = np.asarray(pixel_maps[key], dtype=np.float32).copy()
            arr[arr == 0] = np.nan
            return arr
    shape = intensity.shape
    num = np.zeros(shape, dtype=np.float64)
    den = np.zeros(shape, dtype=np.float64)
    for i in range(1, n_exp + 1):
        tau_key = f'tau{i}'
        amp_key = f'a{i}'
        if tau_key in pixel_maps and amp_key in pixel_maps:
            tau = np.asarray(pixel_maps[tau_key], dtype=np.float64)
            amp = np.asarray(pixel_maps[amp_key], dtype=np.float64)
            valid = np.isfinite(tau) & np.isfinite(amp)
            if weighting == 'intensity':
                num[valid] += amp[valid] * tau[valid] ** 2
                den[valid] += amp[valid] * tau[valid]
            else:
                num[valid] += amp[valid] * tau[valid]
                den[valid] += amp[valid]
    result = np.full(shape, np.nan, dtype=np.float32)
    mask = den > 0
    result[mask] = (num[mask] / den[mask]).astype(np.float32)
    return result

def apply_color_scale(
    image,
    vmin=None,
    vmax=None,
    gamma=1.0,
    percentile_auto=(2, 98),
):
    valid_mask = ~np.isnan(image)
    valid_pixels = image[valid_mask]
    if vmin is None:
        vmin = np.percentile(valid_pixels, percentile_auto[0]) if valid_pixels.size > 0 else 0
    if vmax is None:
        vmax = np.percentile(valid_pixels, percentile_auto[1]) if valid_pixels.size > 0 else 1
    clipped = np.clip(image, vmin, vmax)
    if vmax > vmin:
        normalized = (clipped - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(clipped)
    if gamma != 1.0:
        normalized = np.power(normalized, 1.0 / gamma)
    normalized[~valid_mask] = np.nan
    return normalized

def get_colormap(name='viridis'):
    cmap_name = COLORMAPS.get(name, name)
    base = plt.get_cmap(cmap_name)
    span = _OPEN_SPAN.get(cmap_name)
    if span is not None:
        cmap = LinearSegmentedColormap.from_list(cmap_name + '_open', base(np.linspace(span[0], span[1], 256)))
    else:
        cmap = base.copy()
    cmap.set_under(cmap(0.0))
    cmap.set_over(cmap(1.0))
    return cmap

def compute_region_stats(
    lifetime_map,
    intensity_map,
    region_mask,
    full_stats=False,
):
    region_lifetime = lifetime_map[region_mask]
    region_intensity = intensity_map[region_mask]
    valid_mask = ~np.isnan(region_lifetime)
    valid_lifetime = region_lifetime[valid_mask]
    valid_intensity = region_intensity[valid_mask]
    stats = {
        'median_tau': float(np.nanmedian(valid_lifetime)) if valid_lifetime.size > 0 else np.nan,
        'mean_amplitude': float(np.mean(valid_intensity)) if valid_intensity.size > 0 else np.nan,
        'photon_count': int(np.sum(valid_intensity)),
        'n_pixels': int(np.sum(valid_mask)),
    }
    if full_stats and valid_lifetime.size > 0:
        stats.update({
            'min_tau': float(np.nanmin(valid_lifetime)),
            'max_tau': float(np.nanmax(valid_lifetime)),
            'std_tau': float(np.nanstd(valid_lifetime)),
            'mean_tau': float(np.nanmean(valid_lifetime)),
            'percentiles': {
                'p25': float(np.nanpercentile(valid_lifetime, 25)),
                'p50': float(np.nanpercentile(valid_lifetime, 50)),
                'p75': float(np.nanpercentile(valid_lifetime, 75)),
                'p90': float(np.nanpercentile(valid_lifetime, 90)),
                'p95': float(np.nanpercentile(valid_lifetime, 95)),
            },
        })
    return stats

def mask_to_rgba(
    mask,
    color=(1.0, 1.0, 1.0),
    alpha=0.3,
):
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = alpha
    return rgba

def compute_residuals(decay, model=None, residuals=None):
    if decay is None:
        return None
    decay = np.asarray(decay, dtype=float)
    if residuals is not None:
        residuals = np.asarray(residuals, dtype=float)
        if residuals.ndim == 1 and len(residuals) == len(decay):
            return residuals
    if model is None:
        return None
    model = np.asarray(model, dtype=float)
    if model.ndim != 1 or len(model) == 0:
        return None
    n = min(len(model), len(decay))
    return (decay[:n] - model[:n]) / np.sqrt(np.maximum(model[:n], 1.0))

def rebuild_summed_model(global_summary, irf_prompt, tcspc_res, n_bins):
    gs = global_summary or {}
    if gs.get('fit_model') not in (None, 'reconvolution'):
        return None
    if float(gs.get('tvb_scale') or 0.0) != 0.0:
        return None
    taus_ns = gs.get('taus_ns')
    amps = gs.get('amps')
    if taus_ns is None or amps is None or irf_prompt is None or not tcspc_res or not n_bins:
        return None
    taus = np.atleast_1d(np.asarray(taus_ns, dtype=float)) * 1e-9
    amps = np.atleast_1d(np.asarray(amps, dtype=float))
    if len(taus) == 0 or len(taus) != len(amps):
        return None
    tcspc_res = float(tcspc_res)
    tail_amp = float(gs.get('tail_amp') or 0.0)
    has_tail = tail_amp > 0
    params = list(taus) + list(amps) + [float(gs.get('irf_shift_bins') or 0.0),
                                        float(gs.get('irf_sigma_bins') or 0.0),
                                        float(gs.get('bg_fit') or 0.0)]
    if has_tail:
        params += [tail_amp, float(gs.get('tail_tau_ns') or 0.0) / (tcspc_res * 1e9)]
    from flimkit.FLIM.models import reconvolution_model
    return reconvolution_model(np.asarray(params), tcspc_res, int(n_bins),
                               np.asarray(irf_prompt, dtype=float), len(taus), 0.0,
                               has_tail, True, True)

def summary_model(global_summary, fit_result):
    gs = global_summary or {}
    model = gs.get('model')
    if isinstance(model, str):
        from flimkit.utils.session import _safe_array_from_json
        model = _safe_array_from_json(model)
    if model is not None and np.asarray(model).size > 0:
        return np.asarray(model, dtype=float)
    fr = fit_result or {}
    try:
        return rebuild_summed_model(gs, fr.get('irf_prompt'), fr.get('tcspc_res'), fr.get('n_bins'))
    except Exception as exc:
        print(f'[Residuals] Could not rebuild the fitted model: {exc}')
        return None
