import re
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from ..PTU.reader import PTUFile
from ..FLIM.fitters import fit_summed, fit_per_pixel
from ..FLIM.fit_tools import find_irf_peak_bin, estimate_bg
from ..FLIM.irf_tools import gaussian_irf_from_fwhm, estimate_irf_from_decay_parametric
from ..configs import (
    MIN_PHOTONS_PERPIX, Optimizer, lm_restarts, de_population, de_maxiter, n_workers,
    MACHINE_IRF_FIT_BG, MACHINE_IRF_FIT_SIGMA, MACHINE_IRF_FIT_TAIL,
    MACHINE_IRF_SIGMA_MAX_FULL, MACHINE_IRF_SIGMA_MAX_HALF,
    IRF_FWHM, IRF_BINS, IRF_FIT_WIDTH,
)

_FILENAME_RE = re.compile(
    r'^(?P<region>.+?)_t(?P<t>\d+)(?:_s(?P<s>\d+))?(?:_z(?P<z>\d+))?\.ptu$',
    re.IGNORECASE
)


def parse_timelapse_filename(fname):
    m = _FILENAME_RE.match(Path(fname).name)
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

def _pool_decays(frame_positions, channel=None):
    pooled = None
    tcspc_res = None
    n_bins = None
    for _t, positions in sorted(frame_positions.items()):
        for _s, ptu_path in sorted(positions.items()):
            ptu = PTUFile(str(ptu_path), verbose=False)
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

def _build_irf(decay, tcspc_res, n_bins, args):
    from ..FLIM.irf_tools import estimate_irf_from_decay_raw
    irf_peak_bin = find_irf_peak_bin(decay)
    decay_peak_bin = int(np.argmax(decay))
    sigma_max = MACHINE_IRF_SIGMA_MAX_FULL
    estimate_irf = getattr(args, 'estimate_irf', 'gaussian')
    if estimate_irf in ('machine_irf', 'machine_irf_sigma_full', 'machine_irf_sigma_half'):
        from ..interactive import _load_machine_irf_prompt
        irf_prompt, _ = _load_machine_irf_prompt(
            getattr(args, 'machine_irf', None), n_bins, irf_peak_bin)
        has_tail = MACHINE_IRF_FIT_TAIL
        fit_sigma = MACHINE_IRF_FIT_SIGMA
        fit_bg = MACHINE_IRF_FIT_BG
        if estimate_irf == 'machine_irf_sigma_full':
            fit_sigma = True
            sigma_max = MACHINE_IRF_SIGMA_MAX_FULL
        elif estimate_irf == 'machine_irf_sigma_half':
            fit_sigma = True
            sigma_max = MACHINE_IRF_SIGMA_MAX_HALF
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
    """ORR = I_s1 / (I_s0 + I_s1)"""
    total = intensity_s0.astype(float) + intensity_s1.astype(float)
    return np.where(total > 0, intensity_s1.astype(float) / total, np.nan).astype(np.float32)

def _make_synthetic_popt(ref_taus_ns, n_exp, n_bins, irf_peak_bin,
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

def _save_json(path, data):
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)

def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Not serializable: {type(obj)}')

def _save_timeseries_csv(time_series, path):
    timepoints = sorted(time_series.keys())
    if not timepoints:
        return
    scalar_keys = [k for k in sorted({k for ts in time_series.values() for k in ts})
                   if k not in ('path',)]
    lines = [','.join(['t'] + scalar_keys)]
    for t in timepoints:
        row = time_series[t]
        vals = [str(t)] + [str(row.get(k, '')) for k in scalar_keys]
        lines.append(','.join(vals))
    Path(path).write_text('\n'.join(lines))


def _save_4d_stacks(frame_positions, group_dir, group_label):
    maps_to_stack = ['intensity', 'tau_mean_amp', 'alpha_1', 'alpha_2',
                     'bound_fraction', 'chi2_r']
    sorted_t = sorted(frame_positions.keys())
    all_s = sorted({s for positions in frame_positions.values() for s in positions})
    for s in all_s:
        for map_name in maps_to_stack:
            frames = []
            for t in sorted_t:
                p = group_dir / f't{t:04d}' / f's{s}' / f'{map_name}.npy'
                if p.exists():
                    frames.append(np.load(str(p)))
                else:
                    frames = []
                    break
            if frames:
                arr = np.stack(frames, axis=0)
                out_path = group_dir / f'{group_label}_s{s}_{map_name}_stack.npy'
                np.save(str(out_path), arr)
                print(f'  Saved stack: s{s} {map_name} {arr.shape}  → {out_path.name}')


def plot_timelapse_summary(per_position_series, output_path, group_label):
    all_s = sorted(per_position_series.keys())
    all_t = sorted({t for s_data in per_position_series.values() for t in s_data})
    if not all_t:
        return
    pos_colors_tau   = ['#2196F3', '#64B5F6', '#0D47A1', '#42A5F5']
    pos_colors_bound = ['#E91E63', '#F48FB1', '#880E4F', '#EC407A']
    def _vals(s_data, key):
        return [s_data.get(t, {}).get(key, np.nan) for t in all_t]
    has_tau   = any(v == v
                    for s_data in per_position_series.values()
                    for v in _vals(s_data, 'tau_mean_mean'))
    has_bound = any(v == v
                    for s_data in per_position_series.values()
                    for v in _vals(s_data, 'bound_fraction_mean'))
    n_plots = int(has_tau) + int(has_bound)
    if n_plots == 0:
        return
    from flimkit.utils.plotting import _EXPORT_RC
    plt.rcParams.update(_EXPORT_RC)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0
    if has_tau:
        ax = axes[ax_idx]
        for i, s in enumerate(all_s):
            vals = _vals(per_position_series[s], 'tau_mean_mean')
            c = pos_colors_tau[i % len(pos_colors_tau)]
            lw = 1.0 if len(all_s) > 1 else 1.5
            ax.plot(all_t, vals, 'o-', color=c, linewidth=lw,
                    label=f's{s}', alpha=0.8)
        if len(all_s) > 1:
            mean_vals = [float(np.nanmean([
                per_position_series[s].get(t, {}).get('tau_mean_mean', np.nan)
                for s in all_s])) for t in all_t]
            ax.plot(all_t, mean_vals, 'o-', color='black', linewidth=2.0,
                    label='mean', zorder=5)
            ax.legend(fontsize=8)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Mean τ (ns)')
        ax.set_title(f'{group_label}')
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    if has_bound:
        ax = axes[ax_idx]
        for i, s in enumerate(all_s):
            vals = _vals(per_position_series[s], 'bound_fraction_mean')
            c = pos_colors_bound[i % len(pos_colors_bound)]
            lw = 1.0 if len(all_s) > 1 else 1.5
            ax.plot(all_t, vals, 'o-', color=c, linewidth=lw,
                    label=f's{s}', alpha=0.8)
        if len(all_s) > 1:
            mean_vals = [float(np.nanmean([
                per_position_series[s].get(t, {}).get('bound_fraction_mean', np.nan)
                for s in all_s])) for t in all_t]
            ax.plot(all_t, mean_vals, 'o-', color='black', linewidth=2.0,
                    label='mean', zorder=5)
            ax.legend(fontsize=8)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Bound fraction  α₂/(α₁+α₂)')
        ax.set_title(f'{group_label}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved summary plot: {Path(output_path).name}')

def _iter_effective_groups(groups, pool_positions, output_dir):
    for (region, z), frame_positions in groups.items():
        if pool_positions:
            group_label = f'{region}_z{z}'
            group_dir   = output_dir / group_label
            yield group_label, group_dir, frame_positions
        else:
            all_s = sorted({s for pos in frame_positions.values() for s in pos})
            for s in all_s:
                group_label = f'{region}_z{z}_s{s}'
                group_dir   = output_dir / group_label
                fp_single = {
                    t: {s: pos[s]}
                    for t, pos in frame_positions.items()
                    if s in pos
                }
                yield group_label, group_dir, fp_single

def fit_timelapse(ptu_dir, output_dir, args,
                  ref_tau1_ns=None, ref_tau2_ns=None, ref_tau3_ns=None,
                  channel=None,
                  pool_positions=False,
                  compute_bound_fraction=False,
                  progress_callback=None,
                  cancel_event=None):
    ptu_dir = Path(ptu_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = group_timelapse_files(ptu_dir)
    if not groups:
        raise ValueError(
            f'No timelapse PTU files found in {ptu_dir}. '
            'Expected pattern: region_tX[_sY][_zZ].ptu')
    print(f'\nTimelapse groups found:')
    for (region, z), frame_positions in groups.items():
        n_t   = len(frame_positions)
        all_s = sorted({s for pos in frame_positions.values() for s in pos})
        t_list = sorted(frame_positions)
        print(f'  region={region}  z={z}  →  {n_t} timepoints '
              f'(t={t_list[0]}…{t_list[-1]})  ×  {len(all_s)} position(s) '
              f'(s={all_s})')
    print(f'  Position handling: {"pooled (shared τ)" if pool_positions else "independent (per-position τ)"}')
    total_frames = sum(
        sum(len(pos) for pos in fp.values())
        for fp in groups.values())
    step = 0
    all_results = {}
    for group_label, group_dir, frame_positions in _iter_effective_groups(
            groups, pool_positions, output_dir):
        if cancel_event is not None and cancel_event.is_set():
            print('  Cancelled.')
            break
        group_dir.mkdir(parents=True, exist_ok=True)
        all_s = sorted({s for pos in frame_positions.values() for s in pos})
        n_t   = len(frame_positions)
        print(f'\n{"="*60}')
        print(f'  GROUP: {group_label}  '
              f'({n_t} timepoints × {len(all_s)} position(s))')
        print(f'{"="*60}')
        n_ts = sum(len(pos) for pos in frame_positions.values())
        print(f'\n[1] Pooling photons across {n_ts} PTU files…')
        pooled_decay, tcspc_res, n_bins = _pool_decays(frame_positions, channel=channel)
        print(f'    Total photons: {pooled_decay.sum():,.0f}')
        irf_prompt, has_tail, fit_bg, fit_sigma, sigma_max = _build_irf(
            pooled_decay, tcspc_res, n_bins, args)
        ref_taus = [ref_tau1_ns, ref_tau2_ns, ref_tau3_ns][:args.nexp]
        use_supplied = all(r is not None for r in ref_taus)
        if use_supplied:
            print(f'\n[2] Using user-supplied τ:  '
                  + '  '.join(f'τ{i+1}={t} ns' for i, t in enumerate(ref_taus)))
            irf_peak_bin = int(find_irf_peak_bin(pooled_decay))
            global_popt = _make_synthetic_popt(
                ref_taus, args.nexp, n_bins,
                irf_peak_bin, fit_sigma, fit_bg, has_tail)
            taus_ns = np.array(ref_taus)
            global_summary = {
                'taus_ns': taus_ns,
                'amps': np.ones(args.nexp) / args.nexp,
                'tau_mean_amp_ns': float(np.mean(taus_ns)),
            }
        else:
            print(f'\n[2] Fitting reference τ from pooled decay  ({args.nexp}-exp)…')
            t0 = time.time()
            global_popt, global_summary = fit_summed(
                pooled_decay, tcspc_res, n_bins,
                irf_prompt, has_tail, fit_bg, fit_sigma,
                args.nexp, args.tau_min, args.tau_max,
                optimizer=getattr(args, 'optimizer', 'de'),
                n_restarts=getattr(args, 'restarts', lm_restarts),
                de_popsize=getattr(args, 'de_population', de_population),
                de_maxiter=getattr(args, 'de_maxiter', de_maxiter),
                workers=getattr(args, 'workers', n_workers),
                polish=not getattr(args, 'no_polish', False),
                cost_function=getattr(args, 'cost_function', 'poisson'),
                sigma_max=sigma_max,
            )
            print(f'    Reference fit: {time.time() - t0:.1f} s')
            taus_ns = global_summary.get('taus_ns', global_popt[:args.nexp] * 1e9)
        for i, tau in enumerate(taus_ns):
            print(f'    τ{i+1} = {tau:.4f} ns  (locked for all frames)')
        _save_json(group_dir / f'{group_label}_reference_fit.json', {
            'taus_ns': list(taus_ns),
            'nexp': args.nexp,
            'tau_min_ns': args.tau_min,
            'tau_max_ns': args.tau_max,
            'total_pooled_photons': float(pooled_decay.sum()),
            'tcspc_res_s': float(tcspc_res),
            'n_bins': int(n_bins),
            'n_timepoints': n_t,
            'positions': all_s,
            'estimate_irf': getattr(args, 'estimate_irf', 'gaussian'),
            'user_supplied_tau': use_supplied,
        })
        print(f'\n[3] Per-frame per-position fitting  (α free, τ locked)…')
        per_pos_series = {s: {} for s in all_s}
        for t, positions in sorted(frame_positions.items()):
            if cancel_event is not None and cancel_event.is_set():
                break
            frame_dir = group_dir / f't{t:04d}'
            frame_dir.mkdir(exist_ok=True)
            for s, ptu_path in sorted(positions.items()):
                if cancel_event is not None and cancel_event.is_set():
                    break
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_frames)
                print(f'\n  t={t}  s={s}: {ptu_path.name}')
                t_start = time.time()
                ptu = PTUFile(str(ptu_path), verbose=False)
                pixel_stack = ptu.raw_pixel_stack(channel=channel)
                if pixel_stack.shape[2] != n_bins:
                    nb = pixel_stack.shape[2]
                    if nb > n_bins:
                        pixel_stack = pixel_stack[:, :, :n_bins]
                    else:
                        pixel_stack = np.pad(
                            pixel_stack, ((0, 0), (0, 0), (0, n_bins - nb)))
                pixel_maps = fit_per_pixel(
                    pixel_stack.astype(np.float32),
                    tcspc_res, n_bins,
                    irf_prompt, has_tail, fit_bg, fit_sigma,
                    global_popt, args.nexp,
                    min_photons=getattr(args, 'min_photons', MIN_PHOTONS_PERPIX),
                    tau_min_ns=args.tau_min,
                    tau_max_ns=args.tau_max,
                    correct_pileup=getattr(args, 'correct_pileup', False),
                    n_sync=getattr(ptu, 'n_records', 0),
                    progress_callback=None,
                    free_tau=False,
                )
                redox = compute_redox_metrics(pixel_maps, args.nexp,
                                              compute_bound_fraction=compute_bound_fraction)
                pos_dir = frame_dir / f's{s}'
                pos_dir.mkdir(exist_ok=True)
                intensity = pixel_maps.get('intensity', pixel_stack.sum(axis=2))
                np.save(str(pos_dir / 'intensity.npy'), intensity.astype(np.float32))
                for map_name in ('alpha_1', 'alpha_2', 'alpha_3', 'tau_mean_amp',
                                 'tau_mean_int', 'chi2_r'):
                    if pixel_maps.get(map_name) is not None:
                        np.save(str(pos_dir / f'{map_name}.npy'),
                                pixel_maps[map_name].astype(np.float32))
                for map_name, arr in redox.items():
                    np.save(str(pos_dir / f'{map_name}.npy'), arr)
                stats = {'t': t, 's': s, 'path': str(ptu_path)}
                tau_map = redox.get('tau_mean')
                if tau_map is None:
                    tau_map = pixel_maps.get('tau_mean_amp')
                if tau_map is not None:
                    valid = tau_map[np.isfinite(tau_map) & (tau_map > 0)]
                    stats['tau_mean_mean'] = float(np.mean(valid)) if valid.size > 0 else float('nan')
                    stats['tau_mean_std']  = float(np.std(valid))  if valid.size > 0 else float('nan')
                if 'bound_fraction' in redox:
                    bf = redox['bound_fraction']
                    valid_bf = bf[np.isfinite(bf)]
                    stats['bound_fraction_mean'] = (
                        float(np.mean(valid_bf)) if valid_bf.size > 0 else float('nan'))
                    stats['bound_fraction_std'] = (
                        float(np.std(valid_bf)) if valid_bf.size > 0 else float('nan'))
                n_fitted = int(np.sum(
                    np.isfinite(pixel_maps.get('tau_mean_amp', np.array([np.nan])))))
                stats['n_pixels_fitted'] = n_fitted
                for i, tau in enumerate(taus_ns):
                    stats[f'tau{i+1}_ns'] = float(tau)
                def _map_mean(name, require_positive=False):
                    m = pixel_maps.get(name)
                    if m is None:
                        return None
                    ok = np.isfinite(m)
                    if require_positive:
                        ok = ok & (m > 0)
                    vals = m[ok]
                    return float(np.mean(vals)) if vals.size > 0 else float('nan')
                for i in range(args.nexp):
                    am = _map_mean(f'alpha_{i+1}')
                    if am is not None:
                        stats[f'alpha_{i+1}_mean'] = am
                chi_mean = _map_mean('chi2_r', require_positive=True)
                if chi_mean is not None:
                    stats['chi2_r_mean'] = chi_mean
                elapsed = time.time() - t_start
                print(f'    τ_mean={stats.get("tau_mean_mean", float("nan")):.4f} ns  '
                      f'bound_frac={stats.get("bound_fraction_mean", float("nan")):.4f}  '
                      f'n_px={n_fitted:,}  ({elapsed:.1f} s)')
                per_pos_series[s][t] = stats
        if getattr(args, 'save_stack', True):
            print(f'\n[4] Saving 4D stacks…')
            _save_4d_stacks(frame_positions, group_dir, group_label)
        if not getattr(args, 'no_plots', False):
            print(f'\n[5] Saving summary plot…')
            plot_timelapse_summary(
                per_pos_series,
                group_dir / f'{group_label}_timeseries.png',
                group_label,
            )
        for s, s_series in per_pos_series.items():
            _save_timeseries_csv(
                s_series,
                group_dir / f'{group_label}_s{s}_timeseries.csv',
            )
            print(f'  Saved CSV: {group_label}_s{s}_timeseries.csv')
        _json_keys = (['tau_mean_mean', 'tau_mean_std',
                       'bound_fraction_mean', 'bound_fraction_std',
                       'n_pixels_fitted', 'chi2_r_mean']
                      + [f'tau{i+1}_ns' for i in range(args.nexp)]
                      + [f'alpha_{i+1}_mean' for i in range(args.nexp)])
        _save_json(group_dir / f'{group_label}_timeseries.json', {
            'positions': all_s,
            'timepoints': sorted(frame_positions.keys()),
            'per_position': {
                str(s): {
                    't': sorted(s_data.keys()),
                    **{k: [s_data.get(t, {}).get(k) for t in sorted(s_data)]
                       for k in _json_keys}
                }
                for s, s_data in per_pos_series.items()
            },
        })
        all_results[(region, z)] = {
            'group_dir': str(group_dir),
            'n_timepoints': n_t,
            'positions': all_s,
            'taus_ns': list(taus_ns),
            'per_position_series': per_pos_series,
        }
        print(f'\n  Group {group_label} done.')
    print(f'\n{"="*60}')
    print(f'  TIMELAPSE COMPLETE  →  {output_dir}')
    print(f'{"="*60}\n')
    return all_results