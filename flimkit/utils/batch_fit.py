import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from flimkit.formats import FLIMFile
from ..FLIM.fit_tools import find_irf_peak_bin
from ..FLIM.irf_tools import (
    gaussian_irf_from_fwhm, estimate_irf_from_decay_parametric, machine_irf_prompt,
)
from ..configs import (
    MACHINE_IRF_SIGMA_MAX_FULL,
    IRF_FWHM, IRF_BINS, IRF_FIT_WIDTH,
    TAU_DISPLAY_MIN, TAU_DISPLAY_MAX,
)

_STACK_MAPS = ['intensity', 'tau_mean_amp', 'alpha_1', 'alpha_2',
               'bound_fraction', 'chi2_r', 'calibrated_chi2_r']

_TL_FILENAME_RE = re.compile(
    r'^(?P<region>.+?)_t(?P<t>\d+)(?:_s(?P<s>\d+))?(?:_z(?P<z>\d+))?\.ptu$',
    re.IGNORECASE
)
_ZS_FILENAME_RE = re.compile(
    r'^(?P<region>.+?)(?:_t(?P<t>\d+))?(?:_s(?P<s>\d+))?_z(?P<z>\d+)\.ptu$',
    re.IGNORECASE
)

def parse_timelapse_filename(fname):
    m = _TL_FILENAME_RE.match(Path(fname).name)
    if not m:
        return None
    region = m.group('region')
    t = int(m.group('t'))
    s = int(m.group('s')) if m.group('s') is not None else 0
    z = int(m.group('z')) if m.group('z') is not None else 0
    return region, t, s, z

def group_timelapse_files(ptu_dir):
    groups = defaultdict(lambda: defaultdict(dict))
    ptu_dir = Path(ptu_dir)
    for p in sorted(ptu_dir.glob('*.ptu')):
        parsed = parse_timelapse_filename(p.name)
        if parsed is None:
            continue
        region, t, s, z = parsed
        groups[(region, z)][t][s] = p
    return {k: dict(v) for k, v in groups.items()}

def parse_zstack_filename(fname):
    m = _ZS_FILENAME_RE.match(Path(fname).name)
    if not m:
        return None
    region = m.group('region')
    t = int(m.group('t')) if m.group('t') is not None else 0
    s = int(m.group('s')) if m.group('s') is not None else 0
    z = int(m.group('z'))
    return region, t, s, z

def group_zstack_files(ptu_dir):
    groups = defaultdict(dict)
    ptu_dir = Path(ptu_dir)
    for p in sorted(ptu_dir.glob('*.ptu')):
        parsed = parse_zstack_filename(p.name)
        if parsed is None:
            continue
        region, t, s, z = parsed
        groups[(region, t, s)][z] = p
    return {k: dict(v) for k, v in groups.items()}

def zstack_group_label(region, t, s):
    if t == 0 and s == 0:
        return region
    return f'{region}_t{t:04d}_s{s}'

def pool_decays(frame_positions, channel=None):
    pooled = None
    tcspc_res = None
    n_bins = None
    for _t, positions in sorted(frame_positions.items()):
        for _s, ptu_path in sorted(positions.items()):
            ptu = FLIMFile(str(ptu_path), verbose=False)
            d = ptu.summed_decay(channel=channel).astype(np.float64)
            if pooled is None:
                pooled = d.copy()
                tcspc_res = ptu.tcspc_res
                n_bins = ptu.n_bins
            else:
                n = min(d.size, pooled.size)
                pooled = pooled[:n] + d[:n]
                n_bins = n
    return pooled, tcspc_res, n_bins

def build_irf(decay, tcspc_res, n_bins, args):
    from ..FLIM.irf_tools import estimate_irf_from_decay_raw
    irf_peak_bin = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    sigma_max = MACHINE_IRF_SIGMA_MAX_FULL
    estimate_irf = getattr(args, 'estimate_irf', 'gaussian')
    if estimate_irf in ('machine_irf', 'machine_irf_sigma_full', 'machine_irf_sigma_half'):
        irf_prompt, _strategy, has_tail, fit_bg, fit_sigma, sigma_max = machine_irf_prompt(
            getattr(args, 'machine_irf', None), n_bins, irf_peak_bin, estimate_irf)
    elif estimate_irf == 'raw':
        irf_prompt = estimate_irf_from_decay_raw(
            decay, tcspc_res, n_bins,
            n_irf_bins=getattr(args, 'irf_bins', IRF_BINS))
        has_tail = True
        fit_sigma = True
        fit_bg = True
    elif estimate_irf == 'parametric':
        irf_prompt = estimate_irf_from_decay_parametric(
            decay, tcspc_res, n_bins,
            fit_window_width_ns=getattr(args, 'irf_fit_width', IRF_FIT_WIDTH))
        has_tail = True
        fit_sigma = True
        fit_bg = True
    else:
        fwhm_ns = getattr(args, 'irf_fwhm', None) or IRF_FWHM or (tcspc_res * 1e9)
        irf_prompt = gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, decay_peak_bin)
        has_tail = False
        fit_sigma = False
        fit_bg = True
    return irf_prompt, has_tail, fit_bg, fit_sigma, sigma_max

def compute_redox_metrics(pixel_maps, n_exp, compute_bound_fraction=False):
    out = {}
    a1 = pixel_maps.get('a1') if pixel_maps.get('a1') is not None else pixel_maps.get('alpha_1')
    a2 = pixel_maps.get('a2') if pixel_maps.get('a2') is not None else pixel_maps.get('alpha_2')
    if compute_bound_fraction and n_exp >= 2 and a1 is not None and a2 is not None:
        total = a1 + a2
        with np.errstate(invalid='ignore', divide='ignore'):
            out['bound_fraction'] = np.where(total > 0, a2 / total, np.nan).astype(np.float32)
    tau_mean = pixel_maps.get('tau_mean_amp')
    if tau_mean is not None:
        out['tau_mean'] = tau_mean.astype(np.float32)
    return out

def compute_intensity_redox_ratio(intensity_s0, intensity_s1):
    total = intensity_s0.astype(float) + intensity_s1.astype(float)
    return np.where(total > 0, intensity_s1.astype(float) / total, np.nan).astype(np.float32)

def make_synthetic_popt(ref_taus_ns, n_exp, n_bins, irf_peak_bin,
                        fit_sigma, fit_bg, has_tail):
    taus = [t * 1e-9 for t in ref_taus_ns if t is not None]
    while len(taus) < n_exp:
        taus.append((taus[-1] if taus else 1e-9) * 2.0)
    taus = taus[:n_exp]
    amps = [1.0 / n_exp] * n_exp
    parts = taus + amps + [float(irf_peak_bin)]
    if fit_sigma:
        parts.append(0.0)
    if fit_bg:
        parts.append(0.0)
    if has_tail:
        parts += [0.0, 1.0]
    return np.array(parts, dtype=float)

def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Not serializable: {type(obj)}')

def save_json(path, data):
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)

def save_series_csv(series, path, index_name='t', drop_keys=('path',)):
    indices = sorted(series.keys())
    if not indices:
        return
    drop = set(drop_keys)
    scalar_keys = [k for k in sorted({k for row in series.values() for k in row})
                   if k not in drop]
    lines = [','.join([index_name] + scalar_keys)]
    for idx in indices:
        row = series[idx]
        vals = [str(idx)] + [str(row.get(k, '')) for k in scalar_keys]
        lines.append(','.join(vals))
    Path(path).write_text('\n'.join(lines))

def resolve_tau_display_range(taus_ns, args):
    tau_lo = getattr(args, 'tau_display_min', None)
    tau_hi = getattr(args, 'tau_display_max', None)
    if tau_lo is None:
        tau_lo = TAU_DISPLAY_MIN
    if tau_hi is None:
        tau_hi = TAU_DISPLAY_MAX
    if tau_lo is not None and tau_hi is not None:
        return float(tau_lo), float(tau_hi)
    taus_valid = [t for t in taus_ns if t == t]
    if taus_valid:
        pad = 0.25 * (max(taus_valid) - min(taus_valid) + 1e-9)
        auto_lo = max(0.0, min(taus_valid) - pad)
        auto_hi = max(taus_valid) + pad
    else:
        auto_lo = getattr(args, 'tau_min', 0.0)
        auto_hi = getattr(args, 'tau_max', 5.0)
    return (float(tau_lo) if tau_lo is not None else float(auto_lo),
            float(tau_hi) if tau_hi is not None else float(auto_hi))

def save_tile_lifetime_txt(path, taus_ns, pixel_maps):
    tau_map = pixel_maps.get('tau_mean_int')
    valid = (np.isfinite(tau_map) & (tau_map > 0)) if tau_map is not None else None
    chi_map = pixel_maps.get('chi2_r')
    chi_valid = (np.isfinite(chi_map) & (chi_map > 0)) if chi_map is not None else None
    lines = ['Per-tile lifetime export']
    for i, tau in enumerate(taus_ns):
        lines.append(f'tau{i+1}_ns = {float(tau):.6f}')
    if valid is not None and valid.any():
        tau_vals = tau_map[valid]
        lines.append(f'tau_mean_int_mean_ns = {float(np.mean(tau_vals)):.6f}')
        lines.append(f'tau_mean_int_median_ns = {float(np.median(tau_vals)):.6f}')
        lines.append(f'tau_mean_int_std_ns = {float(np.std(tau_vals)):.6f}')
        n_pixels_fitted = int(valid.sum())
    else:
        lines.append('tau_mean_int_mean_ns = nan')
        lines.append('tau_mean_int_median_ns = nan')
        lines.append('tau_mean_int_std_ns = nan')
        n_pixels_fitted = 0
    lines.append(f'n_pixels_fitted = {n_pixels_fitted}')
    if chi_valid is not None and chi_valid.any():
        lines.append(f'chi2_r_mean = {float(np.mean(chi_map[chi_valid])):.6f}')
    else:
        lines.append('chi2_r_mean = nan')
    Path(path).write_text('\n'.join(lines) + '\n')

def save_map_stacks(slice_dirs, out_dir, out_prefix, maps=None):
    maps = maps or _STACK_MAPS
    out_dir = Path(out_dir)
    for map_name in maps:
        frames = []
        for d in slice_dirs:
            p = Path(d) / f'{map_name}.npy'
            if p.exists():
                frames.append(np.load(str(p)))
            else:
                frames = []
                break
        if frames:
            arr = np.stack(frames, axis=0)
            out_path = out_dir / f'{out_prefix}_{map_name}_stack.npy'
            np.save(str(out_path), arr)
            print(f'  Saved stack: {out_prefix} {map_name} {arr.shape}  → {out_path.name}')

def plot_metric_summary(series_map, output_path, group_label, x_label):
    labels = sorted(series_map.keys())
    all_x = sorted({x for s in series_map.values() for x in s})
    if not all_x:
        return
    tau_colors = ['#2196F3', '#64B5F6', '#0D47A1', '#42A5F5']
    bound_colors = ['#E91E63', '#F48FB1', '#880E4F', '#EC407A']
    def _vals(series, key):
        return [series.get(x, {}).get(key, np.nan) for x in all_x]
    has_tau = any(v == v
                  for s in series_map.values() for v in _vals(s, 'tau_mean_mean'))
    has_bound = any(v == v
                    for s in series_map.values() for v in _vals(s, 'bound_fraction_mean'))
    n_plots = int(has_tau) + int(has_bound)
    if n_plots == 0:
        return
    from flimkit.utils.plotting import _EXPORT_RC
    plt.rcParams.update(_EXPORT_RC)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    multi = len(labels) > 1
    def _plot_metric(ax, key, ylabel, colors):
        for i, lab in enumerate(labels):
            vals = _vals(series_map[lab], key)
            c = colors[i % len(colors)]
            lw = 1.0 if multi else 1.5
            ax.plot(all_x, vals, 'o-', color=c, linewidth=lw, label=str(lab), alpha=0.8)
        if multi:
            mean_vals = [float(np.nanmean([series_map[lab].get(x, {}).get(key, np.nan)
                                           for lab in labels])) for x in all_x]
            ax.plot(all_x, mean_vals, 'o-', color='black', linewidth=2.0,
                    label='mean', zorder=5)
            ax.legend(fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{group_label}')
        ax.grid(True, alpha=0.3)
    idx = 0
    if has_tau:
        _plot_metric(axes[idx], 'tau_mean_mean', 'Mean τ (ns)', tau_colors)
        idx += 1
    if has_bound:
        _plot_metric(axes[idx], 'bound_fraction_mean',
                     'Bound fraction  α₂/(α₁+α₂)', bound_colors)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved summary plot: {Path(output_path).name}')
