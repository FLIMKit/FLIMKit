import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm


COLORMAPS = {
    'hsv': 'hsv',
    'viridis': 'viridis',
    'cool': 'cool',
    'hot': 'hot',
    'twilight': 'twilight',
}


def compute_weighted_lifetime(
    pixel_maps,
    intensity,
    n_exp=2,
    weighting='amplitude',
) -> np.ndarray:
    # weighting='amplitude' → Σ(aᵢτᵢ)/Σ(aᵢ);  'intensity' → Σ(aᵢτᵢ²)/Σ(aᵢτᵢ)
    primary  = 'tau_mean_int' if weighting == 'intensity' else 'tau_mean_amp'
    fallback = 'tau_mean_amp' if weighting == 'intensity' else 'tau_mean_int'
    for key in (primary, fallback):
        if key in pixel_maps:
            arr = np.asarray(pixel_maps[key], dtype=np.float32).copy()
            arr[arr == 0] = np.nan
            return arr

    # No precomputed mean map - derive the requested weighting from components.
    # Key format is 'tau1', 'tau2', ... and 'a1', 'a2', ... (no underscore)
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
    # Create working copy, preserve NaN
    valid_mask = ~np.isnan(image)
    valid_pixels = image[valid_mask]

    # Auto-detect range if not provided
    if vmin is None:
        vmin = np.percentile(valid_pixels, percentile_auto[0]) if valid_pixels.size > 0 else 0
    if vmax is None:
        vmax = np.percentile(valid_pixels, percentile_auto[1]) if valid_pixels.size > 0 else 1

    # Clip to range
    clipped = np.clip(image, vmin, vmax)

    # Normalize to [0, 1]
    if vmax > vmin:
        normalized = (clipped - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(clipped)

    # Apply gamma correction
    if gamma != 1.0:
        normalized = np.power(normalized, 1.0 / gamma)

    # Restore NaN
    normalized[~valid_mask] = np.nan

    return normalized


def get_colormap(name='viridis'):
    cmap_name = COLORMAPS.get(name, name)
    return plt.cm.get_cmap(cmap_name)


def compute_region_stats(
    lifetime_map,
    intensity_map,
    region_mask,
    full_stats=False,
):
    region_lifetime = lifetime_map[region_mask]
    region_intensity = intensity_map[region_mask]

    # Filter valid (non-NaN) pixels
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
