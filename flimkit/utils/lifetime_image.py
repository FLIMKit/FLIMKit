import numpy as np
from pathlib import Path


def make_lifetime_image(
    canvas,
    output_dir,
    roi_name,
    tau_min_ns=0.0,
    tau_max_ns=5.0,
    smooth_sigma_px=0.0,
    intensity_percentile_lo=0.0,
    intensity_percentile_hi=99.0,
    gamma=0.4,
    dpi=200,
    verbose=True,
    tau_key='tau_mean_amp',
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter

    # Use non-interactive 'Agg' backend for thread-safe file saving
    # This prevents segfaults when matplotlib operations happen in worker threads
    current_backend = matplotlib.get_backend()
    try:
        matplotlib.use('Agg', force=True)
    except Exception:
        pass

    try:
        import tifffile as _tifffile
        _has_tifffile = True
    except ImportError:
        _has_tifffile = False

    from flimkit.configs import FLIM_CMAP

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau_map = np.asarray(canvas[tau_key], dtype=float)
    int_map = np.asarray(canvas['intensity'],    dtype=float)
    _tau_label = {'tau_mean_amp': 'τ_amp', 'tau_mean_int': 'τ_int'}.get(tau_key, 'τ')
    valid   = np.isfinite(tau_map) & (int_map > 0)

    if smooth_sigma_px > 0:
        tau_filled = np.where(valid, tau_map, 0.0)
        w          = valid.astype(float)
        denom      = gaussian_filter(w,          sigma=smooth_sigma_px)
        tau_smooth = np.where(denom > 0.01,
                              gaussian_filter(tau_filled, sigma=smooth_sigma_px) / denom,
                              np.nan)
    else:
        tau_smooth = np.where(valid, tau_map, np.nan)

    if _has_tifffile:
        # TIF uses unsmoothed tau_map (NaN->0 for unfitted pixels).
        # tau_smooth is only for the PNG - it fills in nearby unfitted
        # pixels via NaN-aware Gaussian, which looks fine in the PNG
        # (multiplied to black by int_norm) but appears as blobs in TIF.
        tau_clipped = np.clip(
            np.where(valid, tau_map, tau_min_ns),
            tau_min_ns, tau_max_ns)
        tau_u16 = ((tau_clipped - tau_min_ns) /
                   (tau_max_ns - tau_min_ns + 1e-12) * 65535
                   ).astype(np.uint16)
        tiff_path = output_dir / f"{roi_name}_tau_intensity_weighted.tif"
        _tifffile.imwrite(str(tiff_path), tau_u16)
        if verbose:
            print(f"  τ TIFF → {tiff_path}  "
                  f"(uint16, {tau_min_ns}-{tau_max_ns} ns → 0-65535)")
        
        # Full-range TIF: also use tau_map so unfitted pixels are NaN, not filled.
        tau_full = np.where(valid, tau_map, np.nan).astype(np.float32)
        tiff_full_path = output_dir / f"{roi_name}_tau_intensity_weighted_fullrange.tif"
        _tifffile.imwrite(str(tiff_full_path), tau_full)
        if verbose:
            tau_vals = tau_map[valid]
            tau_fmin = float(np.nanmin(tau_vals))
            tau_fmax = float(np.nanmax(tau_vals))
            print(f"  τ full-range TIFF → {tiff_full_path}  "
                  f"(float32, {tau_fmin:.3f}-{tau_fmax:.3f} ns, unscaled)")

    tau_norm = np.clip(
        (tau_smooth - tau_min_ns) / (tau_max_ns - tau_min_ns + 1e-12),
        0.0, 1.0)

    int_vals = int_map[valid]
    # lo=0 - never clip the low end; dim tissue stays dim but visible.
    # Clipping lo to a non-zero percentile turns the dimmest real tissue black.
    lo = 0.0
    hi = float(np.percentile(int_vals, intensity_percentile_hi))
    hi = max(hi, 1e-6)

    # Hard-clip to hi only - removes outlier bright pixels (tile edges, cell
    # clusters) that would otherwise compress bulk tissue toward zero.
    int_clipped = np.clip(int_map, 0.0, hi)
    int_norm = np.power(
        np.clip(int_clipped / hi, 0.0, 1.0), gamma)
    int_norm[~valid] = 0.0

    if verbose:
        print(f"  intensity  lo=0  hi={hi:.1f}  "
              f"median={np.median(int_vals):.1f}  gamma={gamma}")

    rgb = FLIM_CMAP(tau_norm)[..., :3]
    rgb *= int_norm[..., np.newaxis]
    rgb[~valid] = 0.0
    rgb = np.clip(rgb, 0.0, 1.0)

    png_path = output_dir / f"{roi_name}_tau_intensity_weighted.png"
    plt.imsave(str(png_path), rgb, dpi=dpi)
    if verbose:
        print(f"  lifetime PNG → {png_path}")

    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [1, 0.03]})
    ax.imshow(rgb, interpolation='nearest', aspect='equal')
    ax.set_title(
        f"{roi_name}  {_tau_label} {tau_min_ns}-{tau_max_ns} ns  "
        f"σ={smooth_sigma_px} px  γ={gamma}",
        fontsize=10, color='black')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    sm = plt.cm.ScalarMappable(
        cmap=FLIM_CMAP,
        norm=mcolors.Normalize(tau_min_ns, tau_max_ns))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(f'{_tau_label} (ns)', fontsize=10, color='black')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='black')
    plt.tight_layout(pad=0.3)

    preview_path = output_dir / f"{roi_name}_tau_intensity_weighted_preview.png"
    plt.savefig(str(preview_path), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    if verbose:
        print(f"  preview PNG → {preview_path}")

    if verbose and valid.any():
        tau_v = tau_map[valid]
        print(f"  τ_amp  median={np.nanmedian(tau_v):.3f}  "
              f"mean={np.nanmean(tau_v):.3f}  "
              f"p5={np.nanpercentile(tau_v, 5):.3f}  "
              f"p95={np.nanpercentile(tau_v, 95):.3f} ns  "
              f"n={valid.sum():,}")

    return png_path

def make_component_rgb_tiff(
    canvas,
    output_dir,
    roi_name,
    n_exp,
    intensity_percentile_hi=99.0,
    verbose=True,
):
    try:
        import tifffile as _tifffile
    except ImportError:
        raise ImportError('tifffile is required - pip install tifffile')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intensity = np.asarray(canvas['intensity'], dtype=float)
    H, W = intensity.shape
    channels = []

    for k in range(1, n_exp + 1):
        amp_key = f'a{k}'
        amp = canvas.get(amp_key)
        if amp is None:
            channels.append(np.zeros((H, W), dtype=np.uint16))
            continue

        amp = np.asarray(amp, dtype=float)
        fitted = np.isfinite(amp) & (intensity > 0)

        # Express as amplitude fraction (composition) to remove intensity bias
        total_amp = np.zeros((H, W), dtype=float)
        for j in range(1, n_exp + 1):
            a_j = canvas.get(f'a{j}')
            if a_j is not None:
                total_amp += np.where(np.isfinite(a_j), a_j, 0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.where((fitted) & (total_amp > 0), amp / total_amp, np.nan)

        # Scale to uint16 - clip outliers at hi percentile
        vals = frac[np.isfinite(frac)]
        if vals.size > 0:
            hi = float(np.percentile(vals, intensity_percentile_hi))
            hi = max(hi, 1e-9)
            ch_u16 = np.where(
                np.isfinite(frac),
                np.clip(frac / hi, 0.0, 1.0) * 65535,
                0.0
            ).astype(np.uint16)
        else:
            ch_u16 = np.zeros((H, W), dtype=np.uint16)

        channels.append(ch_u16)

    # Pad to 3 channels (RGB) with zeros for unused channels
    while len(channels) < 3:
        channels.append(np.zeros((H, W), dtype=np.uint16))

    # Stack as (H, W, 3) RGB
    rgb_u16 = np.stack(channels[:3], axis=-1)

    labels = ['R=τ₁', 'G=τ₂', 'B=τ₃']
    channel_info = '  '.join(labels[:n_exp])

    tiff_path = output_dir / f"{roi_name}_component_rgb.tif"
    _tifffile.imwrite(str(tiff_path), rgb_u16, photometric='rgb')

    if verbose:
        print(f"  component RGB TIFF → {tiff_path}")
        print(f"    {channel_info}  (amplitude fraction, uint16)")

    return tiff_path