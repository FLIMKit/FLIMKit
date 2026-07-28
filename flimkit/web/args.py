import argparse
from pathlib import Path
from flimkit.UI.utils import _C

def _opt_float(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _opt_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

MACHINE_METHODS = ('machine_irf', 'machine_irf_sigma_half', 'machine_irf_sigma_full')

def build_fov_args(vals):
    cfg = _C()
    ptu = (vals.get('ptu') or '').strip()
    a = argparse.Namespace()
    a.ptu = ptu
    a.xlsx = (vals.get('xlsx') or '').strip() or None
    a.debug_xlsx = False
    a.print_config = False
    method = vals.get('irf_method', 'machine_irf')
    a.irf = None
    a.irf_xlsx = None
    a.no_xlsx_irf = True
    a.estimate_irf = method
    if method in MACHINE_METHODS:
        a.machine_irf = str(cfg['MACHINE_IRF_DEFAULT_PATH'])
    else:
        a.machine_irf = None
    a.irf_bins = cfg['IRF_BINS']
    a.irf_fit_width = cfg['IRF_FIT_WIDTH']
    a.irf_fwhm = _opt_float(vals.get('irf_fwhm')) if vals.get('irf_fwhm') not in (None, '') else cfg['IRF_FWHM']
    model = vals.get('model', 'discrete')
    a.dist_type = model
    if model in ('discrete', 'tail'):
        a.nexp = _opt_int(vals.get('nexp'), cfg['n_exp'])
        a.dist_n_components = 1
    else:
        a.nexp = 2
        a.dist_n_components = _opt_int(vals.get('ncomp'), 1)
    a.tau_min = float(vals.get('tau_min', cfg['Tau_min']))
    a.tau_max = float(vals.get('tau_max', cfg['Tau_max']))
    a.mode = vals.get('mode', cfg['D_mode'])
    a.binning = _opt_int(vals.get('binning'), cfg['binning_factor'])
    a.min_photons = _opt_int(vals.get('min_photons'), cfg['MIN_PHOTONS_PERPIX'])
    a.optimizer = vals.get('optimizer') or cfg['Optimizer']
    a.cost_function = vals.get('cost_function', 'poisson')
    a.restarts = _opt_int(vals.get('restarts'), cfg['lm_restarts'])
    a.de_population = _opt_int(vals.get('de_population'), cfg['de_population'])
    a.de_maxiter = _opt_int(vals.get('de_maxiter'), cfg['de_maxiter'])
    a.workers = _opt_int(vals.get('workers'), cfg['n_workers'])
    a.no_polish = False
    _chan = (vals.get('channel') or '').strip()
    a.channel = int(_chan) if _chan.isdigit() else cfg['channels']
    a.out = out_path_for(ptu, vals.get('out') or cfg['OUT_NAME'])
    a.no_plots = True
    a.cell_mask = bool(vals.get('cell_mask', False))
    a.correct_pileup = bool(vals.get('correct_pileup', False))
    a.intensity_threshold = _opt_float(vals.get('threshold'))
    a.irf_align = vals.get('irf_align', 'steepest_rise')
    a.irf_shift_bins = _opt_int(vals.get('irf_shift_bins'), 2)
    a.align_irf = bool(vals.get('align_irf', False))
    a.free_tau_perpixel = bool(vals.get('free_tau', False))
    a.fit_start_ns = _opt_float(vals.get('fit_start_ns'))
    a.fit_end_ns = _opt_float(vals.get('fit_end_ns'))
    a.exclude_ns = (vals.get('exclude_ns') or '').strip() or None
    a.fit_t0 = bool(vals.get('fit_t0', False))
    a.tvb_ptu = (vals.get('tvb_ptu') or '').strip() or None
    a.tvb_channel = None
    return a

def out_path_for(ptu, out_raw):
    if Path(out_raw).parent == Path('.'):
        return str(Path(ptu).parent / out_raw)
    return out_raw
