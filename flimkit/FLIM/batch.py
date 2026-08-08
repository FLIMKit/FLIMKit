import time
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from flimkit.formats import FLIMFile
from ..FLIM.fitters import fit_summed, fit_per_pixel
from ..FLIM.fit_tools import find_irf_peak_bin
from ..utils.lifetime_image import make_lifetime_image, make_component_rgb_tiff
from ..utils.plotting import plot_summed
from ..configs import (
    MIN_PHOTONS_PERPIX, lm_restarts, de_population, de_maxiter, n_workers,
)
from ..utils.batch_fit import (
    group_timelapse_files, group_zstack_files, zstack_group_label,
    pool_decays, build_irf, make_synthetic_popt, compute_redox_metrics,
    save_json, save_series_csv, resolve_tau_display_range,
    save_tile_lifetime_txt, save_map_stacks, plot_metric_summary,
)

def _save_4d_stacks(frame_positions, group_dir, group_label):
    sorted_t = sorted(frame_positions.keys())
    all_s = sorted({s for pos in frame_positions.values() for s in pos})
    for s in all_s:
        save_map_stacks([group_dir / f't{t:04d}' / f's{s}' for t in sorted_t],
                        group_dir, f'{group_label}_s{s}')

def _iter_effective_groups(groups, pool_positions, output_dir):
    for (region, z), frame_positions in groups.items():
        if pool_positions:
            group_label = f'{region}_z{z}'
            group_dir = output_dir / group_label
            yield group_label, group_dir, frame_positions
        else:
            all_s = sorted({s for pos in frame_positions.values() for s in pos})
            for s in all_s:
                group_label = f'{region}_z{z}_s{s}'
                group_dir = output_dir / group_label
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
        n_t = len(frame_positions)
        all_s = sorted({s for pos in frame_positions.values() for s in pos})
        t_list = sorted(frame_positions)
        print(f'  region={region}  z={z}  →  {n_t} timepoints '
              f'(t={t_list[0]}…{t_list[-1]})  ×  {len(all_s)} position(s) '
              f'(s={all_s})')
    print(f'  Position handling: {'pooled (shared τ)' if pool_positions else 'independent (per-position τ)'}')
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
        n_t = len(frame_positions)
        print(f'\n{'='*60}')
        print(f'  GROUP: {group_label}  '
              f'({n_t} timepoints × {len(all_s)} position(s))')
        print(f'{'='*60}')
        n_ts = sum(len(pos) for pos in frame_positions.values())
        print(f'\n[1] Pooling photons across {n_ts} PTU files…')
        pooled_decay, tcspc_res, n_bins = pool_decays(frame_positions, channel=channel)
        print(f'    Total photons: {pooled_decay.sum():,.0f}')
        irf_prompt, has_tail, fit_bg, fit_sigma, sigma_max = build_irf(
            pooled_decay, tcspc_res, n_bins, args)
        ref_taus = [ref_tau1_ns, ref_tau2_ns, ref_tau3_ns][:args.nexp]
        use_supplied = all(r is not None for r in ref_taus)
        if use_supplied:
            print(f'\n[2] Using user-supplied τ:  '
                  + '  '.join(f'τ{i+1}={t} ns' for i, t in enumerate(ref_taus)))
            irf_peak_bin = int(find_irf_peak_bin(pooled_decay))
            global_popt = make_synthetic_popt(
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
        tau_disp_min, tau_disp_max = resolve_tau_display_range(taus_ns, args)
        print(f'    Lifetime display range (fixed for all timepoints): '
              f'{tau_disp_min:.3f}-{tau_disp_max:.3f} ns')
        save_json(group_dir / f'{group_label}_reference_fit.json', {
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
            'calibrated_chi2_pearson': global_summary.get(
                'calibrated_chi2_pearson'),
            'calibrated_chi2_tail_pearson': global_summary.get(
                'calibrated_chi2_tail_pearson'),
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
                ptu = FLIMFile(str(ptu_path), verbose=False)
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
                    n_sync=getattr(ptu, 'n_sync', None),
                    fit_idx=global_summary.get('fit_idx'),
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
                                 'tau_mean_int', 'chi2_r', 'calibrated_chi2_r'):
                    if pixel_maps.get(map_name) is not None:
                        np.save(str(pos_dir / f'{map_name}.npy'),
                                pixel_maps[map_name].astype(np.float32))
                for map_name, arr in redox.items():
                    np.save(str(pos_dir / f'{map_name}.npy'), arr)
                roi_name = f'{group_label}_t{t:04d}_s{s}'
                if getattr(args, 'save_lifetime', True):
                    try:
                        make_lifetime_image(
                            canvas=pixel_maps, output_dir=pos_dir, roi_name=roi_name,
                            tau_min_ns=tau_disp_min, tau_max_ns=tau_disp_max,
                            intensity_percentile_hi=95, tau_key='tau_mean_int', verbose=False,
                        )
                    except Exception as exc:
                        print(f'    Warning: lifetime image export failed for {roi_name}: {exc}')
                    finally:
                        plt.close('all')
                    try:
                        save_tile_lifetime_txt(
                            pos_dir / f'{roi_name}_lifetime.txt', taus_ns, pixel_maps)
                    except Exception as exc:
                        print(f'    Warning: lifetime .txt export failed for {roi_name}: {exc}')
                    try:
                        tile_decay = pixel_stack.sum(axis=(0, 1)).astype(np.float64)
                        tile_popt, tile_summary = fit_summed(
                            tile_decay, tcspc_res, n_bins,
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
                        plot_summed(
                            tile_decay, tile_summary, ptu, None,
                            args.nexp, getattr(args, 'estimate_irf', 'gaussian'),
                            str(pos_dir / roi_name), irf_prompt=irf_prompt,
                        )
                    except Exception as exc:
                        print(f'    Warning: per-tile detail fit plot failed for {roi_name}: {exc}')
                    finally:
                        plt.close('all')
                if getattr(args, 'save_rgb', True):
                    try:
                        make_component_rgb_tiff(
                            canvas=pixel_maps, output_dir=pos_dir, roi_name=roi_name,
                            n_exp=args.nexp, intensity_percentile_hi=95, verbose=False,
                        )
                    except Exception as exc:
                        print(f'    Warning: component RGB TIFF export failed for {roi_name}: {exc}')
                if getattr(args, 'save_intensity', True):
                    try:
                        import tifffile as _tifffile
                        int_max_disp = getattr(args, 'intensity_display_max', None)
                        i_max = float(int_max_disp) if int_max_disp is not None \
                            else float(np.percentile(intensity[intensity > 0], 99.0)
                                       if (intensity > 0).any() else 1.0)
                        i_max = max(i_max, 1e-6)
                        intensity_u16 = np.clip(
                            intensity.astype(np.float64) / i_max * 65535, 0, 65535
                        ).astype(np.uint16)
                        _tifffile.imwrite(str(pos_dir / f'{roi_name}_intensity.tif'), intensity_u16)
                    except Exception as exc:
                        print(f'    Warning: intensity TIFF export failed for {roi_name}: {exc}')
                if getattr(args, 'save_ind', False):
                    try:
                        from ..utils.enhanced_outputs import save_individual_tau_maps
                        save_individual_tau_maps(
                            pixel_maps, pos_dir, roi_name=roi_name, n_exp=args.nexp)
                    except Exception as exc:
                        print(f'    Warning: individual component map export failed for {roi_name}: {exc}')
                stats = {'t': t, 's': s, 'path': str(ptu_path)}
                tau_map = redox.get('tau_mean')
                if tau_map is None:
                    tau_map = pixel_maps.get('tau_mean_amp')
                if tau_map is not None:
                    valid = tau_map[np.isfinite(tau_map) & (tau_map > 0)]
                    stats['tau_mean_mean'] = float(np.mean(valid)) if valid.size > 0 else float('nan')
                    stats['tau_mean_std'] = float(np.std(valid)) if valid.size > 0 else float('nan')
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
                print(f'    τ_mean={stats.get('tau_mean_mean', float('nan')):.4f} ns  '
                      f'bound_frac={stats.get('bound_fraction_mean', float('nan')):.4f}  '
                      f'n_px={n_fitted:,}  ({elapsed:.1f} s)')
                per_pos_series[s][t] = stats
        if getattr(args, 'save_stack', True):
            print(f'\n[4] Saving 4D stacks…')
            _save_4d_stacks(frame_positions, group_dir, group_label)
        if not getattr(args, 'save_npy', True):
            for t in frame_positions:
                for s in frame_positions[t]:
                    pos_dir = group_dir / f't{t:04d}' / f's{s}'
                    for f_ in pos_dir.glob('*.npy'):
                        try:
                            f_.unlink(missing_ok=True)
                        except Exception as exc:
                            print(f'    Warning: could not remove {f_}: {exc}')
        if not getattr(args, 'no_plots', False):
            print(f'\n[5] Saving summary plot…')
            plot_metric_summary(
                {f's{s}': v for s, v in per_pos_series.items()},
                group_dir / f'{group_label}_timeseries.png',
                group_label, 'Timepoint')
        for s, s_series in per_pos_series.items():
            save_series_csv(
                s_series,
                group_dir / f'{group_label}_s{s}_timeseries.csv',
                index_name='t', drop_keys=('path',))
            print(f'  Saved CSV: {group_label}_s{s}_timeseries.csv')
        _json_keys = (['tau_mean_mean', 'tau_mean_std',
                       'bound_fraction_mean', 'bound_fraction_std',
                       'n_pixels_fitted', 'chi2_r_mean']
                      + [f'tau{i+1}_ns' for i in range(args.nexp)]
                      + [f'alpha_{i+1}_mean' for i in range(args.nexp)])
        save_json(group_dir / f'{group_label}_timeseries.json', {
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
    print(f'\n{'='*60}')
    print(f'  TIMELAPSE COMPLETE  →  {output_dir}')
    print(f'{'='*60}\n')
    return all_results

def fit_zstack(ptu_dir, output_dir, args,
               ref_tau1_ns=None, ref_tau2_ns=None, ref_tau3_ns=None,
               channel=None,
               compute_bound_fraction=False,
               progress_callback=None,
               cancel_event=None):
    ptu_dir = Path(ptu_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = group_zstack_files(ptu_dir)
    if not groups:
        raise ValueError(
            f'No z-stack PTU files found in {ptu_dir}. '
            'Expected pattern: region_zX.ptu (optionally region_tX_sY_zX.ptu)')
    print(f'\nZ-stack groups found:')
    for (region, t, s), zslices in groups.items():
        z_list = sorted(zslices)
        print(f'  region={region}  t={t}  s={s}  →  {len(z_list)} slices '
              f'(z={z_list[0]}…{z_list[-1]})')
    total_slices = sum(len(z) for z in groups.values())
    step = 0
    all_results = {}
    for (region, t, s), zslices in groups.items():
        if cancel_event is not None and cancel_event.is_set():
            print('  Cancelled.')
            break
        group_label = zstack_group_label(region, t, s)
        group_dir = output_dir / group_label
        group_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = group_dir / 'lifetime_preview'
        preview_dir.mkdir(exist_ok=True)
        n_z = len(zslices)
        print(f'\n{'='*60}')
        print(f'  STACK: {group_label}  ({n_z} z-slices)')
        print(f'{'='*60}')
        print(f'\n[1] Pooling photons across {n_z} z-slices  (one FOV)…')
        pool_input = {z: {0: p} for z, p in zslices.items()}
        pooled_decay, tcspc_res, n_bins = pool_decays(pool_input, channel=channel)
        print(f'    Total photons: {pooled_decay.sum():,.0f}')
        irf_prompt, has_tail, fit_bg, fit_sigma, sigma_max = build_irf(
            pooled_decay, tcspc_res, n_bins, args)
        ref_taus = [ref_tau1_ns, ref_tau2_ns, ref_tau3_ns][:args.nexp]
        use_supplied = all(r is not None for r in ref_taus)
        if use_supplied:
            print(f'\n[2] Using user-supplied τ:  '
                  + '  '.join(f'τ{i+1}={t_} ns' for i, t_ in enumerate(ref_taus)))
            irf_peak_bin = int(find_irf_peak_bin(pooled_decay))
            global_popt = make_synthetic_popt(
                ref_taus, args.nexp, n_bins,
                irf_peak_bin, fit_sigma, fit_bg, has_tail)
            taus_ns = np.array(ref_taus)
            global_summary = {'taus_ns': taus_ns, 'n_exp': args.nexp}
        else:
            print(f'\n[2] Fitting reference τ from pooled z-stack  ({args.nexp}-exp)…')
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
            print(f'    τ{i+1} = {tau:.4f} ns  (locked for whole stack)')
        tau_disp_min, tau_disp_max = resolve_tau_display_range(taus_ns, args)
        print(f'    Lifetime display range (fixed for all slices): '
              f'{tau_disp_min:.3f}-{tau_disp_max:.3f} ns')
        save_json(group_dir / f'{group_label}_reference_fit.json', {
            'taus_ns': list(taus_ns),
            'nexp': args.nexp,
            'tau_min_ns': args.tau_min,
            'tau_max_ns': args.tau_max,
            'total_pooled_photons': float(pooled_decay.sum()),
            'tcspc_res_s': float(tcspc_res),
            'n_bins': int(n_bins),
            'n_slices': n_z,
            'z_slices': sorted(zslices),
            'estimate_irf': getattr(args, 'estimate_irf', 'gaussian'),
            'user_supplied_tau': use_supplied,
            'calibrated_chi2_pearson': global_summary.get(
                'calibrated_chi2_pearson'),
            'calibrated_chi2_tail_pearson': global_summary.get(
                'calibrated_chi2_tail_pearson'),
        })
        n_ref = min(len(pooled_decay), n_bins)
        ref_time_ns = (np.arange(n_ref) + 0.5) * tcspc_res * 1e9
        ref_model = global_summary.get('model')
        np.savez(
            str(group_dir / 'reference_decay.npz'),
            decay=np.asarray(pooled_decay[:n_ref], dtype=np.float64),
            time_ns=ref_time_ns.astype(np.float64),
            model=(np.asarray(ref_model[:n_ref], dtype=np.float64)
                   if ref_model is not None else np.array([], dtype=np.float64)),
            irf_prompt=(np.asarray(irf_prompt[:n_ref], dtype=np.float64)
                        if irf_prompt is not None else np.array([], dtype=np.float64)),
            taus_ns=np.asarray(list(taus_ns), dtype=np.float64),
            reduced_chi2_tail=np.asarray(
                [global_summary.get('reduced_chi2_tail', float('nan'))], dtype=np.float64),
            calibrated_chi2_pearson=np.asarray(
                [global_summary.get('calibrated_chi2_pearson', float('nan'))],
                dtype=np.float64),
            calibrated_chi2_tail_pearson=np.asarray(
                [global_summary.get('calibrated_chi2_tail_pearson', float('nan'))],
                dtype=np.float64),
        )
        print(f'\n[3] Per-slice per-pixel fitting  (α free, τ locked)…')
        z_series = {}
        for z, ptu_path in sorted(zslices.items()):
            if cancel_event is not None and cancel_event.is_set():
                break
            step += 1
            if progress_callback is not None:
                progress_callback(step, total_slices)
            print(f'\n  z={z}: {ptu_path.name}')
            t_start = time.time()
            slice_dir = group_dir / f'z{z:04d}'
            slice_dir.mkdir(exist_ok=True)
            ptu = FLIMFile(str(ptu_path), verbose=False)
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
                n_sync=getattr(ptu, 'n_sync', None),
                fit_idx=global_summary.get('fit_idx'),
                progress_callback=None,
                free_tau=False,
            )
            redox = compute_redox_metrics(pixel_maps, args.nexp,
                                          compute_bound_fraction=compute_bound_fraction)
            intensity = pixel_maps.get('intensity', pixel_stack.sum(axis=2))
            np.save(str(slice_dir / 'intensity.npy'), intensity.astype(np.float32))
            for map_name in ('alpha_1', 'alpha_2', 'alpha_3', 'tau_mean_amp',
                             'tau_mean_int', 'chi2_r', 'calibrated_chi2_r'):
                if pixel_maps.get(map_name) is not None:
                    np.save(str(slice_dir / f'{map_name}.npy'),
                            pixel_maps[map_name].astype(np.float32))
            for map_name, arr in redox.items():
                np.save(str(slice_dir / f'{map_name}.npy'), arr)
            roi_name = f'{group_label}_z{z:04d}'
            if getattr(args, 'save_lifetime', True):
                try:
                    png_path = make_lifetime_image(
                        canvas=pixel_maps, output_dir=slice_dir, roi_name=roi_name,
                        tau_min_ns=tau_disp_min, tau_max_ns=tau_disp_max,
                        intensity_percentile_hi=95, tau_key='tau_mean_int', verbose=False,
                    )
                    if png_path is not None and Path(png_path).exists():
                        shutil.copyfile(str(png_path), str(preview_dir / f'z{z:04d}.png'))
                except Exception as exc:
                    print(f'    Warning: lifetime image export failed for {roi_name}: {exc}')
                finally:
                    plt.close('all')
                try:
                    save_tile_lifetime_txt(
                        slice_dir / f'{roi_name}_lifetime.txt', taus_ns, pixel_maps)
                except Exception as exc:
                    print(f'    Warning: lifetime .txt export failed for {roi_name}: {exc}')
                try:
                    tile_decay = pixel_stack.sum(axis=(0, 1)).astype(np.float64)
                    tile_popt, tile_summary = fit_summed(
                        tile_decay, tcspc_res, n_bins,
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
                    plot_summed(
                        tile_decay, tile_summary, ptu, None,
                        args.nexp, getattr(args, 'estimate_irf', 'gaussian'),
                        str(slice_dir / roi_name), irf_prompt=irf_prompt,
                    )
                except Exception as exc:
                    print(f'    Warning: per-slice detail fit plot failed for {roi_name}: {exc}')
                finally:
                    plt.close('all')
            if getattr(args, 'save_rgb', True):
                try:
                    make_component_rgb_tiff(
                        canvas=pixel_maps, output_dir=slice_dir, roi_name=roi_name,
                        n_exp=args.nexp, intensity_percentile_hi=95, verbose=False,
                    )
                except Exception as exc:
                    print(f'    Warning: component RGB TIFF export failed for {roi_name}: {exc}')
            if getattr(args, 'save_intensity', True):
                try:
                    import tifffile as _tifffile
                    int_max_disp = getattr(args, 'intensity_display_max', None)
                    i_max = float(int_max_disp) if int_max_disp is not None \
                        else float(np.percentile(intensity[intensity > 0], 99.0)
                                   if (intensity > 0).any() else 1.0)
                    i_max = max(i_max, 1e-6)
                    intensity_u16 = np.clip(
                        intensity.astype(np.float64) / i_max * 65535, 0, 65535
                    ).astype(np.uint16)
                    _tifffile.imwrite(str(slice_dir / f'{roi_name}_intensity.tif'), intensity_u16)
                except Exception as exc:
                    print(f'    Warning: intensity TIFF export failed for {roi_name}: {exc}')
            if getattr(args, 'save_ind', False):
                try:
                    from ..utils.enhanced_outputs import save_individual_tau_maps
                    save_individual_tau_maps(
                        pixel_maps, slice_dir, roi_name=roi_name, n_exp=args.nexp)
                except Exception as exc:
                    print(f'    Warning: individual component map export failed for {roi_name}: {exc}')
            stats = {'z': z, 'path': str(ptu_path)}
            tau_map = redox.get('tau_mean')
            if tau_map is None:
                tau_map = pixel_maps.get('tau_mean_amp')
            if tau_map is not None:
                valid = tau_map[np.isfinite(tau_map) & (tau_map > 0)]
                stats['tau_mean_mean'] = float(np.mean(valid)) if valid.size > 0 else float('nan')
                stats['tau_mean_std'] = float(np.std(valid)) if valid.size > 0 else float('nan')
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
            print(f'    τ_mean={stats.get('tau_mean_mean', float('nan')):.4f} ns  '
                  f'bound_frac={stats.get('bound_fraction_mean', float('nan')):.4f}  '
                  f'n_px={n_fitted:,}  ({elapsed:.1f} s)')
            z_series[z] = stats
        if getattr(args, 'save_stack', True):
            print(f'\n[4] Saving (Z, H, W) stacks…')
            save_map_stacks(
                [group_dir / f'z{z:04d}' for z in sorted(zslices)],
                group_dir, group_label)
        if not getattr(args, 'save_npy', True):
            for z in zslices:
                slice_dir = group_dir / f'z{z:04d}'
                for f_ in slice_dir.glob('*.npy'):
                    try:
                        f_.unlink(missing_ok=True)
                    except Exception as exc:
                        print(f'    Warning: could not remove {f_}: {exc}')
        if not getattr(args, 'no_plots', False):
            print(f'\n[5] Saving z-series plot…')
            plot_metric_summary(
                {group_label: z_series},
                group_dir / f'{group_label}_zseries.png', group_label, 'Z-slice')
        save_series_csv(
            z_series, group_dir / f'{group_label}_zseries.csv',
            index_name='z', drop_keys=('path', 'z'))
        print(f'  Saved CSV: {group_label}_zseries.csv')
        _json_keys = (['tau_mean_mean', 'tau_mean_std',
                       'bound_fraction_mean', 'bound_fraction_std',
                       'n_pixels_fitted', 'chi2_r_mean']
                      + [f'tau{i+1}_ns' for i in range(args.nexp)]
                      + [f'alpha_{i+1}_mean' for i in range(args.nexp)])
        save_json(group_dir / f'{group_label}_zseries.json', {
            'z_slices': sorted(z_series.keys()),
            **{k: [z_series.get(z, {}).get(k) for z in sorted(z_series)]
               for k in _json_keys},
        })
        all_results[group_label] = {
            'group_dir': str(group_dir),
            'preview_dir': str(preview_dir),
            'n_slices': n_z,
            'taus_ns': list(taus_ns),
            'z_series': z_series,
        }
        print(f'\n  Stack {group_label} done.')
    print(f'\n{'='*60}')
    print(f'  Z-STACK COMPLETE  →  {output_dir}')
    print(f'{'='*60}\n')
    return all_results
