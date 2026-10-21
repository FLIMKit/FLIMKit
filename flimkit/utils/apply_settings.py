import copy
import json
from pathlib import Path
import numpy as np

def target_args(src_args, target_ptu, target_xlsx=None, irf_method='irf_xlsx'):
    a = copy.copy(src_args)
    target = Path(target_ptu)
    a.ptu = str(target)
    a.xlsx = target_xlsx or None
    if irf_method == 'irf_xlsx':
        if not target_xlsx:
            raise ValueError(target.name + ' has no LAS X export, which its IRF method needs')
        a.irf_xlsx = target_xlsx
    src_ptu = Path(str(src_args.ptu))
    src_out = Path(str(src_args.out))
    if src_out.parent == src_ptu.parent:
        a.out = str(target.parent / target.stem)
    else:
        a.out = str(src_out.parent / target.stem)
    return a

def _scalar(val):
    if isinstance(val, np.ndarray) and val.ndim == 0:
        val = val.item()
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    return val

def read_session_settings(session_path):
    form_state = None
    color_scale = None
    try:
        data = np.load(session_path, allow_pickle=True)
    except Exception:
        return form_state, color_scale
    if 'form_state_json' in data.files:
        try:
            form_state = json.loads(_scalar(data['form_state_json']))
        except Exception:
            pass
    if 'fov_color_scale' in data.files:
        try:
            color_scale = json.loads(_scalar(data['fov_color_scale']))
        except Exception:
            pass
    return form_state, color_scale

def fov_targets(project, source_ptu):
    src = str(Path(source_ptu).resolve()) if source_ptu else ''
    found = []
    for stem, rec in project.sorted_scans():
        if rec.scan_type != 'fov':
            continue
        if str(Path(rec.source_path).resolve()) == src:
            continue
        found.append((stem, rec))
    return found

def settings_summary(form_state, color_scale, int_scale=None):
    fs = form_state or {}
    cs = color_scale or {}
    ics = int_scale or {}
    model = fs.get('fit_model_fov', 'discrete')
    if model in ('discrete', 'tail'):
        comps = str(fs.get('nexp_fov', '')) + '-exp'
    else:
        comps = str(fs.get('ncomp_dist_fov', '')) + ' component'
    window = str(fs.get('tau_min_fov', '') or 'default') + ' to ' + str(fs.get('tau_max_fov', '') or 'default') + ' ns'
    lo = cs.get('vmin')
    hi = cs.get('vmax')
    rng = 'auto' if lo is None and hi is None else (('auto' if lo is None else '%.2f' % lo) + ' to ' + ('auto' if hi is None else '%.2f' % hi) + ' ns')
    lines = [
        'Model: ' + model + ', ' + comps,
        'IRF: ' + str(fs.get('irf_method', 'irf_xlsx')),
        'Fit window: ' + window,
        'FLIM display: ' + rng + ', gamma ' + str(cs.get('gamma', 1.0)) + ', ' + str(cs.get('cmap', 'viridis')),
    ]
    ilo = ics.get('vmin')
    ihi = ics.get('vmax')
    irng = ('0' if ilo is None else '%g' % ilo) + ' to ' + ('99th percentile' if ihi is None else '%g' % ihi)
    lines.append('Intensity display: ' + irng + ', ' + str(ics.get('cmap', 'inferno')))
    return '\n'.join(lines)

def fitted_fovs(project):
    if project is None:
        return []
    return [(stem, rec) for stem, rec in project.sorted_scans()
            if rec.scan_type == 'fov' and rec.has_session]

def _text_list(val):
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if not isinstance(val, (list, tuple)):
        return []
    return [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in val]

def write_fit_summary(fit_result, output_path, name=None):
    params = _text_list(fit_result.get('summary_params'))
    values = _text_list(fit_result.get('summary_values'))
    units = _text_list(fit_result.get('summary_units'))
    if not params or len(params) != len(values):
        return False
    units = units if len(units) == len(params) else [''] * len(params)
    lines = ['FLIMKit fit summary']
    if name:
        lines.append('File: ' + str(name))
    strategy = _scalar(fit_result.get('strategy'))
    if strategy:
        lines.append('IRF: ' + str(strategy))
    decay = fit_result.get('decay')
    if isinstance(decay, np.ndarray) and decay.size:
        lines.append(f'Total photons: {float(decay.sum()):,.0f}')
    lines.append('')
    width = max(len(p) for p in params)
    for p, v, u in zip(params, values, units):
        lines.append((p.ljust(width) + '  ' + v + (' ' + u if u else '')).rstrip())
    Path(output_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return True
