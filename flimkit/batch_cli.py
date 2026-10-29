import argparse
import csv
import gc
from pathlib import Path

from .configs import (
    n_exp, Tau_min, Tau_max, binning_factor, MIN_PHOTONS_PERPIX, lm_restarts,
    de_population, de_maxiter, n_workers, IRF_BINS, IRF_FIT_WIDTH, IRF_FWHM,
    channels, INTENSITY_THRESHOLD, TAU_DISPLAY_MIN, TAU_DISPLAY_MAX,
    MACHINE_IRF_DEFAULT_PATH)


def _resolve_out_dir(ptu_dir, output_dir):
    out = Path(output_dir) if output_dir else Path(ptu_dir) / 'save'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _append_row(csv_path, row):
    new_file = Path(csv_path).exists() == False
    with open(csv_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file == True:
            writer.writeheader()
        writer.writerow(row)


def _error_row(key, name, exc):
    return {key: name, 'status': f'ERROR: {type(exc).__name__}: {str(exc)[:80]}'}


def _batch_parser(description, input_flag, input_help, need_ptu_dir=True):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(input_flag, required=True, help=input_help)
    if need_ptu_dir == True:
        ap.add_argument('--ptu-dir', required=True)
    ap.add_argument('--output-dir', default=None, help='Default: <ptu-dir>/save')
    ap.add_argument('--estimate-irf', default='machine_irf',
                    choices=['parametric', 'raw', 'machine_irf', 'machine_irf_sigma_full', 'machine_irf_sigma_half'])
    ap.add_argument('--machine-irf', default=str(MACHINE_IRF_DEFAULT_PATH))
    ap.add_argument('--nexp', type=int, default=n_exp, choices=[1, 2, 3])
    ap.add_argument('--tau-min', type=float, default=Tau_min)
    ap.add_argument('--tau-max', type=float, default=Tau_max)
    ap.add_argument('--binning', type=int, default=binning_factor)
    ap.add_argument('--min-photons', type=int, default=MIN_PHOTONS_PERPIX)
    ap.add_argument('--optimizer', default='de', choices=['lm_multistart', 'de'])
    ap.add_argument('--restarts', type=int, default=lm_restarts)
    ap.add_argument('--de-population', type=int, default=de_population)
    ap.add_argument('--de-maxiter', type=int, default=de_maxiter)
    ap.add_argument('--workers', type=int, default=n_workers)
    ap.add_argument('--channel', type=int, default=channels)
    ap.add_argument('--irf-fwhm', type=float, default=IRF_FWHM)
    ap.add_argument('--intensity-threshold', type=int, default=INTENSITY_THRESHOLD)
    ap.add_argument('--correct-pileup', action='store_true')
    ap.add_argument('--tau-display-min', type=float, default=TAU_DISPLAY_MIN)
    ap.add_argument('--tau-display-max', type=float, default=TAU_DISPLAY_MAX)
    ap.add_argument('--intensity-display-min', type=float, default=None)
    ap.add_argument('--intensity-display-max', type=float, default=None)
    ap.add_argument('--tau-weighting', default='amp', choices=['amp', 'int'])
    ap.add_argument('--fit-t0', action='store_true')
    return ap


def _common_args(a):
    return dict(
        ptu_dir=a.ptu_dir, estimate_irf=a.estimate_irf, machine_irf=a.machine_irf,
        irf=None, irf_xlsx=None, irf_xlsx_dir=None, irf_xlsx_map=None,
        irf_bins=IRF_BINS, irf_fit_width=IRF_FIT_WIDTH, irf_fwhm=a.irf_fwhm,
        nexp=a.nexp, tau_min=a.tau_min, tau_max=a.tau_max, mode='both',
        binning=a.binning, min_photons=a.min_photons, optimizer=a.optimizer,
        restarts=a.restarts, de_population=a.de_population, de_maxiter=a.de_maxiter,
        workers=a.workers, no_polish=False, channel=a.channel, no_plots=True,
        cell_mask=False, intensity_threshold=a.intensity_threshold,
        correct_pileup=a.correct_pileup, no_xlsx_irf=True, debug_xlsx=False,
        print_config=False, xlsx=None, dist_type='discrete', dist_n_components=1,
        fit_t0=a.fit_t0, tvb_ptu=None, tvb_channel=None,
        tau_display_min=a.tau_display_min, tau_display_max=a.tau_display_max,
        intensity_display_min=a.intensity_display_min, intensity_display_max=a.intensity_display_max,
        tau_weighting=a.tau_weighting)


def _run_tile_batch(a):
    from .interactive import _run_tile_fit
    xlif_files = sorted(Path(a.xlif_dir).glob('*.xlif'))
    if len(xlif_files) == 0:
        raise ValueError(f'No .xlif files found in: {a.xlif_dir}')
    out_dir = _resolve_out_dir(a.ptu_dir, a.output_dir)
    csv_path = out_dir / 'batch_roi_fit_summary.csv'
    n_total = len(xlif_files)
    for idx, xlif_path in enumerate(xlif_files):
        name = xlif_path.stem
        roi_out = out_dir / name.replace(' ', '_')
        roi_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*50}\n  [{idx + 1}/{n_total}] {name}\n{'='*50}")
        try:
            args = argparse.Namespace(
                **_common_args(a), xlif=str(xlif_path), output_dir=str(roi_out),
                ptu_basename=name, rotate_tiles=a.rotate_tiles,
                register_tiles=a.register_tiles, reg_max_shift_px=a.reg_max_shift, out=None)
            result = _run_tile_fit(args)
            summary = result['global_summary'] if result else {}
            row = {'roi': name, 'status': 'OK', **summary}
            del result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row = _error_row('roi', name, exc)
        gc.collect()
        _append_row(csv_path, row)
    print(f'\nBatch complete. CSV: {csv_path}')
    return csv_path


def _run_fov_batch(a):
    from .interactive import _run_flim_fit
    from .formats import supported_extensions
    ptu_files = sorted(p for e in supported_extensions() for p in Path(a.fov_dir).glob('*' + e))
    if len(ptu_files) == 0:
        raise ValueError(f'No FLIM files found in: {a.fov_dir}')
    out_dir = _resolve_out_dir(a.ptu_dir, a.output_dir)
    csv_path = out_dir / 'batch_fov_fit_summary.csv'
    n_total = len(ptu_files)
    for idx, ptu_path in enumerate(ptu_files):
        name = ptu_path.stem
        fov_out = out_dir / name.replace(' ', '_')
        fov_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*50}\n  [{idx + 1}/{n_total}] {name}\n{'='*50}")
        try:
            args = argparse.Namespace(
                **_common_args(a), ptu=str(ptu_path), out=str(fov_out / name),
                irf_align='steepest_rise', irf_shift_bins=2)
            result = _run_flim_fit(args)
            summary = result.get('global_summary', {})
            row = {'fov': name, 'status': 'OK',
                   'pileup_pct': result.get('pileup_pct') if result.get('pileup_pct') is not None else '',
                   'count_rate_mhz': result.get('count_rate_mhz') if result.get('count_rate_mhz') is not None else '',
                   **summary}
            del result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row = _error_row('fov', name, exc)
        gc.collect()
        _append_row(csv_path, row)
    print(f'\nBatch FOV complete. CSV: {csv_path}')
    return csv_path


def _inquire_common(a):
    import inquirer
    from .interactive import yes_no_question
    a.ptu_dir = input('Enter directory containing PTU files: ').strip()
    if Path(a.ptu_dir).is_dir() == False:
        raise ValueError(f'PTU directory not found: {a.ptu_dir}')
    a.output_dir = input('Enter output directory (Enter for <ptu-dir>/save): ').strip() or None
    a.estimate_irf = inquirer.prompt([inquirer.List(
        'm', message='IRF method',
        choices=['machine_irf', 'machine_irf_sigma_half', 'machine_irf_sigma_full', 'parametric', 'raw'])])['m']
    a.machine_irf = str(MACHINE_IRF_DEFAULT_PATH)
    if a.estimate_irf.startswith('machine_irf'):
        a.machine_irf = input(f'Machine IRF .npy (Enter for default: {a.machine_irf}): ').strip() or a.machine_irf
        if Path(a.machine_irf).exists() == False:
            print('  Warning: machine IRF file not found, falling back to parametric')
            a.estimate_irf = 'parametric'
    a.nexp = int(inquirer.prompt([inquirer.List(
        'n', message='Number of exponential components', choices=['1', '2', '3'], default='2')])['n'])
    val = input('Min photon intensity per pixel (Enter for none): ').strip() if yes_no_question('Apply intensity threshold?') == 'y' else ''
    a.intensity_threshold = int(val) if val else None
    a.correct_pileup = yes_no_question('Apply pile-up correction?') == 'y'
    return a


def _ask_float(label, default):
    val = input(f'  {label} (Enter for {default}): ').strip()
    return float(val) if val else default


def _ask_int(label, default):
    val = input(f'  {label} (Enter for {default}): ').strip()
    return int(val) if val else default


def _inquire_display(a):
    import inquirer
    from .interactive import yes_no_question
    a.tau_weighting = 'int' if inquirer.prompt([inquirer.List(
        'w', message='Lifetime weighting for the tau image',
        choices=['amplitude', 'intensity'])])['w'] == 'intensity' else 'amp'
    if yes_no_question('Set lifetime display range for tau images?') == 'y':
        a.tau_display_min = _ask_float('Min tau (ns)', None)
        a.tau_display_max = _ask_float('Max tau (ns)', None)
    if yes_no_question('Set intensity display range?') == 'y':
        a.intensity_display_min = _ask_float('Min intensity (photons)', None)
        a.intensity_display_max = _ask_float('Max intensity (photons)', None)
    return a


def _inquire_expert(a):
    import inquirer
    from .interactive import yes_no_question
    if yes_no_question('Set expert fit settings?') == 'n':
        return a
    a.tau_min = _ask_float('Min tau for fit (ns)', a.tau_min)
    a.tau_max = _ask_float('Max tau for fit (ns)', a.tau_max)
    a.binning = _ask_int('Binning factor', a.binning)
    a.min_photons = _ask_int('Min photons per pixel', a.min_photons)
    a.optimizer = inquirer.prompt([inquirer.List(
        'o', message='Optimizer', choices=['de', 'lm_multistart'], default=a.optimizer)])['o']
    a.restarts = _ask_int('LM restarts', a.restarts)
    a.de_population = _ask_int('DE population', a.de_population)
    a.de_maxiter = _ask_int('DE max iterations', a.de_maxiter)
    a.workers = _ask_int('Workers', a.workers)
    a.channel = _ask_int('Channel', a.channel)
    a.irf_fwhm = _ask_float('IRF FWHM (ns)', a.irf_fwhm)
    return a


def tileBatchFit(interactive=False):
    if interactive == True:
        from .interactive import yes_no_question
        a = argparse.Namespace(
            tau_min=Tau_min, tau_max=Tau_max, binning=binning_factor,
            min_photons=MIN_PHOTONS_PERPIX, optimizer='de', restarts=lm_restarts,
            de_population=de_population, de_maxiter=de_maxiter, workers=n_workers,
            channel=channels, irf_fwhm=IRF_FWHM, tau_display_min=TAU_DISPLAY_MIN,
            tau_display_max=TAU_DISPLAY_MAX, reg_max_shift=120, intensity_display_min=None,
            intensity_display_max=None, tau_weighting='amp', fit_t0=False)
        print('\nBatch tiled ROI fit: one fit per .xlif in a folder')
        a.xlif_dir = input('Enter directory containing XLIF files: ').strip()
        if Path(a.xlif_dir).is_dir() == False:
            raise ValueError(f'XLIF directory not found: {a.xlif_dir}')
        _inquire_common(a)
        a.rotate_tiles = yes_no_question('Apply 90 degree clockwise rotation to tiles?') == 'y'
        a.register_tiles = yes_no_question('Register tile columns by phase correlation?') == 'y'
        if a.register_tiles == True:
            a.reg_max_shift = _ask_int('Max registration shift (px)', a.reg_max_shift)
        _inquire_display(a)
        _inquire_expert(a)
    else:
        ap = _batch_parser('Batch per-tile FLIM fit over every .xlif in a folder', '--xlif-dir',
                           'Folder of .xlif files')
        ap.add_argument('--no-rotate', action='store_true')
        ap.add_argument('--no-register', action='store_true')
        ap.add_argument('--reg-max-shift', type=int, default=120)
        a = ap.parse_args()
        a.rotate_tiles = a.no_rotate == False
        a.register_tiles = a.no_register == False
    return _run_tile_batch(a)


def fovBatchFit(interactive=False):
    if interactive == True:
        a = argparse.Namespace(
            tau_min=Tau_min, tau_max=Tau_max, binning=binning_factor,
            min_photons=MIN_PHOTONS_PERPIX, optimizer='de', restarts=lm_restarts,
            de_population=de_population, de_maxiter=de_maxiter, workers=n_workers,
            channel=channels, irf_fwhm=IRF_FWHM, tau_display_min=TAU_DISPLAY_MIN,
            tau_display_max=TAU_DISPLAY_MAX, intensity_display_min=None,
            intensity_display_max=None, tau_weighting='amp', fit_t0=False)
        print('\nBatch single-FOV fit: one fit per FLIM file in a folder')
        _inquire_common(a)
        _inquire_expert(a)
        a.fov_dir = a.ptu_dir
    else:
        ap = _batch_parser('Batch single-FOV FLIM fit over every FLIM file in a folder', '--fov-dir',
                           'Folder of FLIM files', need_ptu_dir=False)
        a = ap.parse_args()
        a.ptu_dir = a.fov_dir
    return _run_fov_batch(a)
