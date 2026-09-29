import numpy as np

def tag_pixel_size_um(tags):
    for key in ('ImgHdr_PixResol', 'ImgHdr_PixRes', 'BH_PixelSize_um'):
        try:
            value = float(tags.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None

def field_area_um2(pixel_size_um, shape):
    try:
        size = float(pixel_size_um)
        h, w = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not size > 0 or h <= 0 or w <= 0:
        return None
    return size * size * h * w

def flim_field_area_um2(flim_file):
    size = tag_pixel_size_um(getattr(flim_file, 'tags', {}) or {})
    tags = getattr(flim_file, 'tags', {}) or {}
    shape = (tags.get('ImgHdr_PixY', tags.get('BH_ImageY')), tags.get('ImgHdr_PixX', tags.get('BH_ImageX')))
    return field_area_um2(size, shape)

def xlif_field_area_um2(xlif_path, basename, shape, binning=1):
    try:
        from pathlib import Path
        from flimkit.utils.xml_utils import get_pixel_size
        pixel_size_m, _ = get_pixel_size(Path(xlif_path), basename)
    except Exception as exc:
        print(f'[TIFF] No pixel size from {xlif_path}: {exc}')
        return None
    if not pixel_size_m:
        return None
    return field_area_um2(pixel_size_m * 1e6 * (binning or 1), shape)

def tiff_resolution(image, area_um2):
    try:
        h, w = np.asarray(image).shape[:2]
        size = (float(area_um2) / (h * w)) ** 0.5
    except (TypeError, ValueError, ZeroDivisionError):
        return {}
    if not size > 0:
        return {}
    per_cm = 1e4 / size
    return {'resolution': (per_cm, per_cm), 'resolutionunit': 'CENTIMETER'}

def lifetime_limits(image, vmin=None, vmax=None, percentile_auto=(2, 98)):
    valid = np.asarray(image)[np.isfinite(image)]
    if vmin is None:
        vmin = float(np.percentile(valid, percentile_auto[0])) if valid.size > 0 else 0.0
    if vmax is None:
        vmax = float(np.percentile(valid, percentile_auto[1])) if valid.size > 0 else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)

def colorbar_ticks(lo, hi, gamma=1.0, n=5):
    values = np.linspace(lo, hi, n)
    positions = ((values - lo) / (hi - lo)) ** (1.0 / gamma)
    return positions, values

def save_colorbar(path, cmap, lo, hi, label, gamma=1.0, dpi=200):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(1.3, 4.0), dpi=dpi, facecolor='white')
    ax = fig.add_axes([0.25, 0.05, 0.25, 0.9])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, cmap=cmap, origin='lower', aspect='auto', extent=(0, 1, 0, 1))
    positions, values = colorbar_ticks(lo, hi, gamma)
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_yticks(positions)
    ax.set_yticklabels([f'{v:.3g}' for v in values], color='black')
    ax.tick_params(axis='y', colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.set_ylabel(label, color='black')
    fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
