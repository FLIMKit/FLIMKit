from pathlib import Path

import numpy as np

def _hex_to_rgb01(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

def _filtered_phasor(session):
    real = np.asarray(session['real_cal'], dtype=float).squeeze()
    imag = np.asarray(session['imag_cal'], dtype=float).squeeze()
    mean = np.asarray(session['mean'], dtype=float).squeeze()
    spec = session.get('phasor_filter') or {}
    method = spec.get('method') or 'none'
    if method == 'none':
        return real, imag, mean
    from flimkit.phasor.filters import phasor_filter
    kwargs = {}
    if method == 'gaussian':
        kwargs['sigma'] = float(spec.get('sigma', 1.0))
    elif method == 'median':
        kwargs['size'] = int(spec.get('size', 3))
    elif method != 'wavelet':
        kwargs['sigma'] = float(spec.get('sigma', 1.0))
        kwargs['size'] = int(spec.get('size', 3))
    try:
        real, imag = phasor_filter(real.copy(), imag.copy(), method, mean=mean, **kwargs)
    except Exception as exc:
        print(f'[Phasor Export] Could not re-apply the saved {method} filter, plotting unfiltered: {exc}')
    return real, imag, mean

def _cursor_radii(params):
    r = float(params.get('radius', 0.05))
    return r, float(params.get('radius_minor', r * 0.6))

def _cursor_masks(real, imag, valid, cursors, params):
    from matplotlib.path import Path as MplPath
    from phasorpy.cursor import mask_from_elliptic_cursor
    r, r_min = _cursor_radii(params)
    masks = []
    for cur in cursors:
        if cur.get('type', 'ellipse') == 'poly':
            pts = np.column_stack([real.ravel(), imag.ravel()])
            m = MplPath(np.asarray(cur['vertices'], dtype=float)).contains_points(pts).reshape(real.shape)
        else:
            m = mask_from_elliptic_cursor(real, imag, np.array([cur['center_g']]), np.array([cur['center_s']]),
                                          radius=r, radius_minor=r_min, angle='semicircle')
            if m.ndim > real.ndim:
                m = m[0]
        masks.append(m & valid)
    return masks

def _cursor_labels(real, imag, masks, frequency):
    from phasorpy.lifetime import phasor_to_apparent_lifetime
    labels = []
    for i, m in enumerate(masks):
        n_px = int(m.sum())
        label = f'C{i + 1}: {n_px} px'
        if n_px and frequency:
            tau_phi, _ = phasor_to_apparent_lifetime(real[m], imag[m], frequency)
            label += f', median τφ {float(np.nanmedian(tau_phi)):.2f} ns'
        labels.append(label)
    return labels

def _draw_cursors(ax, cursors, params, labels):
    from matplotlib.patches import Ellipse, Polygon
    r, r_min = _cursor_radii(params)
    for i, cur in enumerate(cursors):
        col = cur['color']
        if cur.get('type', 'ellipse') == 'poly':
            verts = cur['vertices']
            ax.add_patch(Polygon(verts, closed=True, facecolor=col, alpha=0.18,
                                 edgecolor=col, linewidth=2, linestyle='--', zorder=8))
            cx = float(np.mean([v[0] for v in verts]))
            cy = float(np.mean([v[1] for v in verts]))
        else:
            cx, cy = cur['center_g'], cur['center_s']
            ang = float(np.degrees(np.arctan2(cy, cx - 0.5) + np.pi / 2.0))
            ax.add_patch(Ellipse((cx, cy), 2 * r, 2 * r_min, angle=ang, facecolor=col,
                                 alpha=0.18, edgecolor=col, linewidth=2, linestyle='--', zorder=8))
        ax.plot(cx, cy, 'o', color=col, ms=7, zorder=10, label=labels[i])
        ax.text(cx + 0.015, cy + 0.015, f'C{i + 1}', color=col, fontsize=10,
                fontweight='bold', zorder=11)

def _new_figure(size, dpi):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=size, dpi=dpi, facecolor='white')
    FigureCanvasAgg(fig)
    return fig

def _plot_title(title, label):
    return f'{title}  -  {label}' if title else label

def savePhasorPlots(session, output_dir, stem, min_photons=0.01, dpi=150):
    from matplotlib.patches import Patch
    from phasorpy.cursor import pseudo_color
    from phasorpy.plot import PhasorPlot
    real, imag, mean = _filtered_phasor(session)
    valid = (mean >= min_photons) & np.isfinite(real) & np.isfinite(imag)
    if not valid.any():
        raise ValueError(f'No pixels above {min_photons} photons to plot')
    frequency = float(session.get('frequency') or 0.0)
    cursors = session.get('cursors') or []
    params = session.get('params') or {}
    masks = _cursor_masks(real, imag, valid, cursors, params)
    labels = _cursor_labels(real, imag, masks, frequency)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = f'Phasor ({frequency:.1f} MHz)' if frequency else 'Phasor'
    spec = session.get('phasor_filter') or {}
    if (spec.get('method') or 'none') != 'none':
        label += f", {spec['method']} filter"
    fig = _new_figure((7, 5), dpi)
    ax = fig.add_subplot(111)
    pp = PhasorPlot(ax=ax, frequency=frequency or None)
    pp.hist2d(real[valid], imag[valid], cmap='inferno', bins=256)
    _draw_cursors(ax, cursors, params, labels)
    if cursors:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(_plot_title(stem, label), fontsize=11)
    plot_file = output_dir / f'{stem}_phasor.png'
    fig.savefig(plot_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    written = [plot_file]
    disp = session.get('display_image')
    disp = np.asarray(disp if disp is not None else session['mean'], dtype=float).squeeze()
    if disp.ndim != 2 or min(disp.shape) < 2:
        print('[Phasor Export] Point measurement, so there is no phasor image to export')
        return written
    fig = _new_figure((7, 6), dpi)
    ax = fig.add_subplot(111)
    if cursors:
        colors = np.array([_hex_to_rgb01(c['color']) for c in cursors])
        ax.imshow(pseudo_color(*masks, intensity=disp, colors=colors),
                  origin='upper', interpolation='nearest')
        handles = [Patch(facecolor=c['color'], edgecolor=c['color'], label=labels[i])
                   for i, c in enumerate(cursors)]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  fontsize=8, framealpha=0.9, borderaxespad=0)
        ax.set_title(_plot_title(stem, 'Phasor cursor overlay'), fontsize=11)
    else:
        im = ax.imshow(disp, cmap='inferno', origin='upper', interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity (photons)')
        ax.set_title(_plot_title(stem, 'Phasor image (no cursors placed)'), fontsize=11)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    image_file = output_dir / f'{stem}_phasor_image.png'
    fig.savefig(image_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    written.append(image_file)
    return written
