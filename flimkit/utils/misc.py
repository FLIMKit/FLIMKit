import os
import logging

def setup_loggers(log_dir: str = '.', log_prefix: str = 'run'):
    os.makedirs(log_dir, exist_ok=True)
    loggers = {}
    
    # Main run logger
    run_logger = logging.getLogger('run')
    run_logger.setLevel(logging.INFO)
    run_fh = logging.FileHandler(os.path.join(log_dir, f'{log_prefix}.log'))
    run_fh.setLevel(logging.INFO)
    run_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    run_fh.setFormatter(run_formatter)
    run_logger.handlers = [run_fh]
    loggers['run'] = run_logger

    # Error logger
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    error_fh = logging.FileHandler(os.path.join(log_dir, 'error.log'))
    error_fh.setLevel(logging.ERROR)
    error_fh.setFormatter(run_formatter)
    error_logger.handlers = [error_fh]
    loggers['error'] = error_logger

    # Warning logger
    warning_logger = logging.getLogger('warning')
    warning_logger.setLevel(logging.WARNING)
    warning_fh = logging.FileHandler(os.path.join(log_dir, 'warning.log'))
    warning_fh.setLevel(logging.WARNING)
    warning_fh.setFormatter(run_formatter)
    warning_logger.handlers = [warning_fh]
    loggers['warning'] = warning_logger

    return loggers

def print_summary(summary: dict, strategy: str, n_exp: int):
    s = summary
    tcspc_res = s['tcspc_res']
    is_dist = 'tau_centers_ns' in s
    is_tail = s.get('fit_model') == 'tail'

    print(f"\n{'─'*60}")
    if is_tail:
        print(f"  Fit: {n_exp}-exp tail (no reconvolution)")
        print(f"{'─'*60}")
        for i, (tau, amp, frac, inten, ifrac) in enumerate(
                zip(s['taus_ns'], s['amps'], s['fractions'],
                    s['intensities'], s['intensity_fractions'])):
            print(f"  τ{i+1} = {tau:8.4f} ns   A{i+1} = {amp:.3e}   "
                  f"f{i+1} = {frac:.4f}   I{i+1} = {inten:.3e} ({ifrac:.4f})")
        print(f"  A_sum = {s['a_sum']:.3e}   I_sum = {s['i_sum']:.3e} cts")
    elif is_dist:
        width_label = 'σ' if s['dist_type'] == 'gaussian' else 'Γ'
        print(f"  Fit: {s['n_components']}-component {s['dist_type']} dist | IRF: {strategy}")
        print(f"{'─'*60}")
        for i, (tau_c, w, fwhm, amp, frac) in enumerate(
                zip(s['tau_centers_ns'], s['widths_ns'], s['fwhms_ns'],
                    s['amps'], s['fractions'])):
            print(f"  τ_c{i+1} = {tau_c:7.4f} ns   {width_label}{i+1} = {w:.4f} ns   "
                  f"FWHM = {fwhm:.4f} ns   f{i+1} = {frac:.4f}")
    else:
        print(f"  Fit: {n_exp}-exp | IRF: {strategy}")
        print(f"{'─'*60}")
        for i, (tau, amp, frac) in enumerate(
                zip(s['taus_ns'], s['amps'], s['fractions'])):
            print(f"  τ{i+1} = {tau:8.4f} ns   α{i+1} = {amp:.3e}   f{i+1} = {frac:.4f}")

    print(f"  τ_mean (amplitude-weighted)  = {s['tau_mean_amp_ns']:.4f} ns")
    print(f"  τ_mean (intensity-weighted)  = {s['tau_mean_int_ns']:.4f} ns")
    print(f"  bg (fitted, Tail Offset)     = {s['bg_fit']:.2f} cts/bin")
    if is_tail:
        print(f"  t0 (lifetime offset)         = {s['t0_ns']:.4f} ns")
        print(f"  Fit window                   = {s['fit_window_ns'][0]:.2f}-"
              f"{s['fit_window_ns'][1]:.2f} ns")
    else:
        print(f"  IRF shift                    = {s['irf_shift_bins']:.3f} bins "
              f"({s['irf_shift_bins'] * tcspc_res * 1e12:.1f} ps)")
        print(f"  IRF σ (prompt broadening)    = {s['irf_sigma_bins']:.3f} bins")
        print(f"  IRF FWHM (effective)         = {s['irf_fwhm_eff_ns']:.4f} ns")
    if not is_dist and not is_tail and s.get('tail_amp', 0) > 0:
        print(f"  IRF tail amp                 = {s['tail_amp']:.4f}")
        print(f"  IRF tail τ                   = {s['tail_tau_ns']:.3f} ns")
        if s['tail_tau_ns'] > 18:
            print(f"   tail τ near upper bound - consider acquiring a scatter PTU")
    p_val = s.get('p_val')
    p_str = f", p={p_val:.4f}" if p_val is not None else ""
    print(f"  χ²_r = {s['reduced_chi2']:.4f}  "
          f"(χ²={s['chi2']:.1f}, DoF={s['dof']}{p_str})  [full window, Neyman]")
    print(f"  χ²_r = {s['reduced_chi2_pearson']:.4f}  "
          f"[full window, Pearson]")
    if not is_tail:
        print(f"  χ²_r = {s['reduced_chi2_tail']:.4f}  "
              f"(tail only, t>{s['tail_start_bin']*tcspc_res*1e9:.2f} ns)  [Neyman]")
        print(f"  χ²_r = {s['reduced_chi2_tail_pearson']:.4f}  "
              f"(tail only, t>{s['tail_start_bin']*tcspc_res*1e9:.2f} ns)  [Pearson]")
    if p_val is not None:
        flag = '' if 0.001 < p_val < 0.999 else ''
        print(f"  {flag} Optimizer: {s['optimizer_msg']}")
    else:
        print(f"  Optimizer: {s['optimizer_msg']}")

def check_full_path(path):
    if os.path.isabs(path) == True:
        return path
    else:
        if os.path.isabs(os.getcwd() + '/' + path) == True:
            return os.getcwd() + '/' + path
        else:
            raise Exception('Path not found')