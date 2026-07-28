import argparse
from pathlib import Path
from flimkit.UI.utils import _C

def build_fov_args(vals):
    cfg = _C()
    ptu = (vals.get('ptu') or '').strip()
    a = argparse.Namespace()
    a.ptu = ptu
    a.xlsx = None
    a.debug_xlsx = False
    a.print_config = False
    irf_source = vals.get('irf_source', 'machine')
    a.irf = None
    a.irf_xlsx = None
    a.no_xlsx_irf = True
    if irf_source == 'machine':
        a.estimate_irf = 'machine_irf'
        a.machine_irf = str(cfg['MACHINE_IRF_DEFAULT_PATH'])
    else:
        a.estimate_irf = 'parametric'
        a.machine_irf = None
    a.irf_bins = cfg['IRF_BINS']
    a.irf_fit_width = cfg['IRF_FIT_WIDTH']
    a.irf_fwhm = cfg['IRF_FWHM']
    model = vals.get('model', 'discrete')
    a.dist_type = model
    if model in ('discrete', 'tail'):
        a.nexp = int(vals.get('nexp', cfg['n_exp']))
        a.dist_n_components = 1
    else:
        a.nexp = 2
        a.dist_n_components = int(vals.get('ncomp', 1))
    a.tau_min = float(vals.get('tau_min', cfg['Tau_min']))
    a.tau_max = float(vals.get('tau_max', cfg['Tau_max']))
    a.mode = vals.get('mode', cfg['D_mode'])
    a.binning = cfg['binning_factor']
    a.min_photons = cfg['MIN_PHOTONS_PERPIX']
    a.optimizer = cfg['Optimizer']
    a.restarts = cfg['lm_restarts']
    a.de_population = cfg['de_population']
    a.de_maxiter = cfg['de_maxiter']
    a.workers = cfg['n_workers']
    a.no_polish = False
    a.channel = cfg['channels']
    a.out = out_path_for(ptu, vals.get('out') or cfg['OUT_NAME'])
    a.no_plots = True
    a.cell_mask = False
    a.correct_pileup = bool(vals.get('correct_pileup', False))
    a.intensity_threshold = None
    a.irf_align = 'steepest_rise'
    a.irf_shift_bins = 2
    a.tvb_ptu = None
    a.tvb_channel = None
    return a

def out_path_for(ptu, out_raw):
    if Path(out_raw).parent == Path('.'):
        return str(Path(ptu).parent / out_raw)
    return out_raw
