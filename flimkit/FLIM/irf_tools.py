import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib
from datetime import timezone
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
from ..formats.PTU.reader import PTUFile, read_pck
from ..formats import FLIMFile
from ..utils.xlsx_tools import load_xlsx
from ..configs import MACHINE_IRF_DIR as _DEFAULT_MACHINE_IRF_DIR
from ..configs import (
    MACHINE_IRF_FIT_BG, MACHINE_IRF_FIT_SIGMA, MACHINE_IRF_FIT_TAIL,
    MACHINE_IRF_SIGMA_MAX_FULL, MACHINE_IRF_SIGMA_MAX_HALF,
)

def machine_irf_prompt(machine_irf_path, n_bins, align_bin, variant='machine_irf',
                       align_label=None):
    from ..interactive import _load_machine_irf_prompt
    irf_prompt, strategy = _load_machine_irf_prompt(machine_irf_path, n_bins, align_bin)
    if align_label:
        strategy += f' align={align_label}'
    has_tail = MACHINE_IRF_FIT_TAIL
    fit_bg = MACHINE_IRF_FIT_BG
    fit_sigma = MACHINE_IRF_FIT_SIGMA
    sigma_max = MACHINE_IRF_SIGMA_MAX_FULL
    if variant == 'machine_irf_sigma_full':
        fit_sigma = True
        sigma_max = MACHINE_IRF_SIGMA_MAX_FULL
        strategy += ' + σ≤{:.1f}'.format(sigma_max)
    elif variant == 'machine_irf_sigma_half':
        fit_sigma = True
        sigma_max = MACHINE_IRF_SIGMA_MAX_HALF
        strategy += ' + σ≤{:.1f}'.format(sigma_max)
    return irf_prompt, strategy, has_tail, fit_bg, fit_sigma, sigma_max

try:
    matplotlib.use('Agg', force=True)
except Exception:
    pass  

def _leading_edge_crossing(arr: np.ndarray, frac: float = 0.5) -> int:
    arr = np.asarray(arr, dtype=float)
    peak = int(np.argmax(arr))
    thr = float(arr[peak]) * frac
    above = np.where(arr[:peak + 1] >= thr)[0]
    return int(above[0]) if len(above) else 0

def _max_slope_bin(arr: np.ndarray) -> int:
    arr = np.asarray(arr, dtype=float)
    if arr.size < 2:
        return 0
    return int(np.argmax(np.diff(arr)))

def _extract_landmarks(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    arr = np.maximum(arr, 0.0)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return {
        'peak': int(np.argmax(arr)),
        'halfmax': _leading_edge_crossing(arr, 0.5),
        'onset10': _leading_edge_crossing(arr, 0.1),
        'slope': _max_slope_bin(arr),
    }

def discover_ptu_xlsx_pairs(folder: str | Path) -> list[tuple[str, Path, Path]]:
    base = Path(folder)
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {base}")
    pairs: list[tuple[str, Path, Path]] = []
    for ptu_path in sorted(base.glob('*.ptu')):
        if ptu_path.name.startswith('._'):
            continue
        name = ptu_path.stem
        xlsx_path = base / f"{name}.xlsx"
        if xlsx_path.exists() and not xlsx_path.name.startswith('._'):
            pairs.append((name, ptu_path, xlsx_path))
    return pairs

def build_machine_irf_from_folder(
    folder: str | Path,
    align_anchor: str = 'peak',
    reducer: str = 'median',
    save: bool = False,
    confirm_save: bool = False,
    output_name: str = 'machine_irf_default',
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    if align_anchor not in {'peak', 'halfmax', 'onset10', 'slope'}:
        raise ValueError('align_anchor must be one of: peak, halfmax, onset10, slope')
    if reducer not in {'median', 'mean'}:
        raise ValueError('reducer must be one of: median, mean')
    pairs = discover_ptu_xlsx_pairs(folder)
    if len(pairs) < 2:
        raise ValueError('Need at least 2 PTU/XLSX pairs to build a machine IRF.')
    irfs = []
    peaks = []
    nbins_all = []
    tcspc_all = []
    for name, ptu_path, xlsx_path in pairs:
        ptu_f = FLIMFile(str(ptu_path), verbose=False)
        xlsx = load_xlsx(str(xlsx_path))
        irf = irf_from_xlsx(xlsx, ptu_f.n_bins, ptu_f.tcspc_res)
        irfs.append(irf)
        peaks.append(int(np.argmax(irf)))
        nbins_all.append(ptu_f.n_bins)
        tcspc_all.append(ptu_f.tcspc_res)
    common_nbins = int(min(nbins_all))
    irfs = [v[:common_nbins] for v in irfs]
    marks = [_extract_landmarks(v) for v in irfs]
    ref_anchor = int(np.median([m[align_anchor] for m in marks]))
    aligned = np.zeros((len(irfs), common_nbins), dtype=float)
    for i, (irf, m) in enumerate(zip(irfs, marks)):
        shift = ref_anchor - int(m[align_anchor])
        aligned[i] = np.roll(irf, shift)
    if reducer == 'median':
        machine_irf = np.median(aligned, axis=0)
    else:
        machine_irf = aligned.mean(axis=0)
    machine_irf = np.maximum(machine_irf, 0.0)
    s = machine_irf.sum()
    if s <= 0:
        raise ValueError('Machine IRF aggregation produced all zeros.')
    machine_irf /= s
    out_paths = None
    if save:
        if not confirm_save:
            raise RuntimeError(
                'Save requested but confirm_save=False. '
                'Set confirm_save=True after explicit user confirmation.'
            )
        if output_dir is None:
            output_dir = _DEFAULT_MACHINE_IRF_DIR
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"{output_name}.npy"
        csv_path = out_dir / f"{output_name}.csv"
        meta_path = out_dir / f"{output_name}_meta.json"
        np.save(npy_path, machine_irf.astype(np.float64))
        np.savetxt(csv_path, machine_irf.astype(np.float64), delimiter=',')
        meta = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_folder': str(Path(folder).resolve()),
            'n_pairs': len(pairs),
            'pair_names': [name for name, _, _ in pairs],
            'align_anchor': align_anchor,
            'reducer': reducer,
            'common_nbins': common_nbins,
            'tcspc_res_ns_mean': float(np.mean(tcspc_all) * 1e9),
            'machine_landmarks': _extract_landmarks(machine_irf),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        out_paths = {
            'npy': str(npy_path),
            'csv': str(csv_path),
            'meta_json': str(meta_path),
        }
    meta = {
        'n_pairs': len(pairs),
        'pair_names': [name for name, _, _ in pairs],
        'align_anchor': align_anchor,
        'reducer': reducer,
        'common_nbins': common_nbins,
        'landmarks': _extract_landmarks(machine_irf),
        'peak_bins_before_alignment': peaks,
    }
    if verbose:
        print(f"Machine IRF built from {len(pairs)} pairs")
        print(f"  anchor={align_anchor}, reducer={reducer}, n_bins={common_nbins}")
        print(f"  landmarks={meta['landmarks']}")
        if out_paths:
            print('  saved:')
            for _, p in out_paths.items():
                print(f"    {p}")
    return {
        'irf': machine_irf,
        'pairs': pairs,
        'metadata': meta,
        'save_paths': out_paths,
    }

def gaussian_irf_from_fwhm(n_bins: int,
                            tcspc_res: float,
                            fwhm_ns: float,
                            peak_bin: int) -> np.ndarray:
    t   = np.arange(n_bins, dtype=float) * tcspc_res * 1e9
    t0 = peak_bin * tcspc_res * 1e9
    irf = np.exp(-(t - t0)**2 * 4.0 * np.log(2) / fwhm_ns**2)
    return irf / irf.sum()

def irf_from_scatter_ptu(path: str, ptu_ref: PTUFile,
                         channel: int | None = None) -> np.ndarray:
    scatter = FLIMFile(path, verbose=False)
    decay   = scatter.summed_decay(channel=channel)
    s = decay.sum()
    if s == 0 and channel is not None:
        decay = scatter.summed_decay(channel=None)
        s = decay.sum()
        if s > 0:
            print(f"  Scatter PTU has no photons on channel {channel}; "
                  f"using auto-detected channel {scatter.photon_channel} instead")
    if s == 0:
        raise ValueError(f"Scatter PTU {path!r} has no photons.")
    n = ptu_ref.n_bins
    if decay.size < n:
        decay = np.concatenate([decay, np.zeros(n - decay.size)])
    else:
        decay = decay[:n]
    s = decay.sum()
    if s == 0:
        raise ValueError(f"Scatter PTU {path!r} has no photons in the first {n} bins.")
    print(f"  IRF from scatter PTU: {s:,.0f} photons")
    return decay / s

def irf_from_pck(path: str, n_bins: int, channel=None) -> np.ndarray:
    hist, tags = read_pck(path)
    if channel is None:
        channel = int(np.argmax(hist.sum(axis=1)))
    if channel >= hist.shape[0]:
        raise ValueError(f'Channel {channel} not in .pck (has {hist.shape[0]} channel(s))')
    full = hist[channel].astype(float)
    peak = int(np.argmax(full))
    if peak >= n_bins:
        raise ValueError(f'.pck IRF peak (bin {peak}) is beyond the {n_bins}-bin target grid; '
                         f'the .pck ({full.size} bins) and the data have different TCSPC resolution.')
    if full.size < n_bins:
        irf = np.concatenate([full, np.zeros(n_bins - full.size)])
    else:
        irf = full[:n_bins]
    s = irf.sum()
    if s == 0:
        raise ValueError(f'.pck IRF channel {channel} has no counts.')
    print(f'  IRF from .pck: channel {channel}, {int(s):,} photons, peak bin {peak}')
    return irf / s

def align_irf_to_bin(irf_prompt: np.ndarray, target_bin: int,
                     n_bins: int) -> tuple[np.ndarray, int]:
    current = int(np.argmax(irf_prompt))
    shift = int(target_bin) - current
    if shift == 0:
        return irf_prompt, 0
    x = np.arange(n_bins, dtype=float)
    shifted = np.interp(x - shift, x, irf_prompt, left=0.0, right=0.0)
    s = shifted.sum()
    if s > 0:
        shifted = shifted / s
    return shifted, shift

def irf_from_measured_file(path: str, ptu_ref: PTUFile,
                           channel: int | None = None) -> np.ndarray:
    if str(path).lower().endswith('.pck'):
        return irf_from_pck(path, ptu_ref.n_bins, channel=channel)
    return irf_from_scatter_ptu(path, ptu_ref, channel=channel)

def irf_from_xlsx_analytical(xlsx: dict, n_bins: int, tcspc_res: float,
                              verbose: bool = True) -> tuple[np.ndarray, dict]:
    if xlsx['irf_t'] is None or xlsx['irf_c'] is None:
        raise ValueError('XLSX does not contain IRF columns.')
    t_pts = np.array(xlsx['irf_t'], dtype=float)
    c_pts = np.maximum(np.array(xlsx['irf_c'], dtype=float), 0.0)
    mask = c_pts > c_pts.max() * 1e-3
    if mask.sum() < 3:
        raise ValueError('Fewer than 3 non-negligible IRF points in xlsx - '
                         'cannot fit analytical model.')
    t_fit = t_pts[mask]
    c_fit = c_pts[mask]
    t0_guess = t_pts[np.argmax(c_pts)]

    def _model(t, t0, fwhm, tail_amp, tail_tau, A):
        gauss = np.exp(-4.0 * np.log(2) * (t - t0)**2 / fwhm**2)
        tail = np.where(t >= t0,
                         tail_amp * np.exp(-(t - t0) / np.maximum(tail_tau, 0.01)),
                         0.0)
        return A * (gauss + tail)
    try:
        popt, _ = curve_fit(
            _model, t_fit, c_fit,
            p0 = [t0_guess, 0.15,  0.05, 0.5,  c_pts.max()],
            bounds=([t0_guess - 0.5, 0.05, 0.0,  0.05, 0],
                    [t0_guess + 0.5, 0.5,  2.0,  10.0, c_pts.max() * 2]),
            maxfev=20000
        )
        t0, fwhm, tail_amp, tail_tau, A = popt
    except Exception as e:
        raise RuntimeError(f"Analytical IRF fit failed: {e}. "
                           f"Try --irf-xlsx with a higher-count IRF export.") from e
    tcspc_ns = tcspc_res * 1e9
    t_full    = np.arange(n_bins, dtype=float) * tcspc_ns
    irf_full = np.maximum(_model(t_full, t0, fwhm, tail_amp, tail_tau, A), 0.0)
    s = irf_full.sum()
    if s == 0:
        raise ValueError('Analytical IRF evaluates to zero on bin grid.')
    irf_norm = irf_full / s
    params = dict(t0_ns=t0, fwhm_ns=fwhm, tail_amp=tail_amp, tail_tau_ns=tail_tau)
    if verbose:
        print(f"  Analytical IRF fit (FLIM microscope model):")
        print(f"    t0       = {t0:.4f} ns  (bin {t0/tcspc_ns:.2f})")
        print(f"    FWHM     = {fwhm*1000:.2f} ps")
        print(f"    tail_amp = {tail_amp:.4f}")
        print(f"    tail_tau = {tail_tau:.4f} ns")
        above = np.where(irf_norm >= irf_norm.max() / 2)[0]
        fwhm_meas = (above[-1] - above[0]) * tcspc_ns if len(above) > 1 else fwhm
        print(f"    FWHM (measured on grid) = {fwhm_meas*1000:.2f} ps")
        print(f"    Peak bin = {np.argmax(irf_norm)}")
    return irf_norm, params

def irf_from_xlsx(xlsx: dict, n_bins: int, tcspc_res: float) -> np.ndarray:
    if xlsx['irf_t'] is None or xlsx['irf_c'] is None:
        raise ValueError('XLSX does not contain IRF columns.')
    tcspc_ns = tcspc_res * 1e9
    t_full     = np.arange(n_bins, dtype=float) * tcspc_ns
    t_pts = np.array(xlsx['irf_t'], dtype=float)
    c_pts = np.array(xlsx['irf_c'], dtype=float)
    c_pts = np.maximum(c_pts, 0.0)
    order = np.argsort(t_pts)
    t_pts, c_pts = t_pts[order], c_pts[order]
    irf_interp = np.interp(t_full, t_pts, c_pts, left=0.0, right=0.0)
    s = irf_interp.sum()
    if s == 0:
        raise ValueError('xlsx IRF is all zeros after interpolation.')
    return irf_interp / s

def gaussian_irf(n_bins: int, peak_bin: int, fwhm_bins: float) -> np.ndarray:
    bins  = np.arange(n_bins, dtype=float)
    sigma = fwhm_bins / 2.3548
    irf = np.exp(-0.5 * ((bins - peak_bin) / sigma)**2)
    return irf / irf.sum()

def reconstruct_irf_from_decay(decay: np.ndarray,
                                tcspc_res: float,
                                n_bins: int,
                                noise_floor: float = 50,
                                noise_frac: float = 0.001,
                                max_bap: int = 2,
                                verbose: bool = False) -> np.ndarray:
    decay = np.asarray(decay, dtype=float)
    if decay.size < 3 or decay.max() <= 0:
        raise ValueError('Decay histogram is empty or all zeros.')
    peak_idx = int(np.argmax(decay))
    peak_val = decay[peak_idx]
    threshold = max(noise_floor, noise_frac * peak_val)
    start_idx = peak_idx
    while start_idx > 0 and decay[start_idx - 1] > threshold:
        start_idx -= 1
    cut_idx = peak_idx
    prev_val = peak_val
    for i in range(1, max_bap + 1):
        next_idx = peak_idx + i
        if next_idx >= n_bins:
            break
        next_val = decay[next_idx]
        if next_val <= 0:
            break
        cut_idx = next_idx
        prev_val = next_val
    irf_full = np.zeros(n_bins, dtype=float)
    for src in range(start_idx, cut_idx + 1):
        if 0 <= src < n_bins:
            irf_full[src] = decay[src]
    total = irf_full.sum()
    if total == 0:
        raise ValueError('Reconstructed IRF has zero counts - check decay quality.')
    irf_norm = irf_full / total
    if verbose:
        tcspc_ns = tcspc_res * 1e9
        bap = cut_idx - peak_idx
        n_rising = peak_idx - start_idx
        above = np.where(irf_norm >= irf_norm.max() / 2)[0]
        fwhm = (above[-1] - above[0]) * tcspc_ns if len(above) > 1 else tcspc_ns
        print(f"  IRF reconstructed from decay rising edge:")
        print(f"    Peak bin (decay)  = {peak_idx}  →  IRF peak bin = {peak_idx}")
        print(f"    Rising edge       = {n_rising} bins")
        print(f"    Bins after peak   = {bap}")
        print(f"    IRF extent        = bins {start_idx}..{cut_idx}  "
              f"({cut_idx - start_idx + 1} bins)")
        print(f"    FWHM (grid)       = {fwhm * 1000:.1f} ps")
    return irf_norm

def estimate_irf_from_decay_raw(decay, tcspc_res, n_bins,
                                n_irf_bins=21, bg_est_pre=5) -> np.ndarray:
    peak_bin = int(np.argmax(decay))
    bg_end = max(0, peak_bin - bg_est_pre)
    bg = float(np.median(decay[:bg_end])) if bg_end > 0 \
                else float(np.median(decay[-30:]))
    decay_sub = np.maximum(decay - bg, 0.0)
    half = n_irf_bins // 2
    start = max(0, peak_bin - half)
    end = min(n_bins, peak_bin + half + 1)
    irf_raw = decay_sub[start:end].copy()
    total = irf_raw.sum()
    if total == 0:
        raise ValueError('Extracted IRF region has zero counts.')
    irf_full          = np.zeros(n_bins, dtype=float)
    irf_full[start:end] = irf_raw / total
    return irf_full

def _irf_parametric(t, t0, amplitude):
    return amplitude * (t / t0) * np.exp(-t / t0)

def estimate_irf_from_decay_parametric(decay, tcspc_res, n_bins,
                                       fit_window_width_ns=1.5,
                                       bg_est_pre=5) -> np.ndarray:
    peak_bin = int(np.argmax(decay))
    bg_end = max(0, peak_bin - bg_est_pre)
    bg = float(np.median(decay[:bg_end])) if bg_end > 0 \
                else float(np.median(decay[-30:]))
    decay_sub = np.maximum(decay - bg, 0.0)
    time_ns = np.arange(n_bins) * tcspc_res * 1e9
    t_peak_ns = time_ns[peak_bin]
    start_ns = max(0, t_peak_ns - fit_window_width_ns / 2)
    end_ns = min(time_ns[-1], t_peak_ns + fit_window_width_ns / 2)
    sb        = np.searchsorted(time_ns, start_ns, side='left')
    eb        = np.searchsorted(time_ns, end_ns,   side='right')
    if eb - sb < 3:
        raise ValueError('Fit window too narrow.')
    t_fit = time_ns[sb:eb] - time_ns[sb]
    y_fit = decay_sub[sb:eb]
    pk = np.argmax(y_fit)
    try:
        popt, _ = curve_fit(_irf_parametric, t_fit, y_fit,
                             p0=[t_fit[pk]/2.0 if pk > 0 else 1.0, y_fit[pk]],
                             bounds=([0.01, 0], [10.0, np.inf]))
        t0, amp = popt
    except Exception as e:
        print(f"Parametric fit failed: {e}, falling back to raw extraction.")
        return estimate_irf_from_decay_raw(decay, tcspc_res, n_bins)
    t_full_ns = time_ns - time_ns[sb]
    irf_full = np.maximum(_irf_parametric(t_full_ns, t0, amp), 0.0)
    total = irf_full.sum()
    return irf_full / total if total > 0 else np.zeros(n_bins)

def build_full_irf(irf_prompt: np.ndarray,
                   shift_bins: float,
                   sigma_bins: float,
                   tail_amp:   float,
                   tail_tau_bins: float,
                   n_bins:     int) -> np.ndarray:
    peak_bin = int(np.argmax(irf_prompt))
    bins     = np.arange(n_bins, dtype=float)
    tail = np.where(
        bins >= peak_bin,
        tail_amp * np.exp(-(bins - peak_bin) / max(tail_tau_bins, 0.1)),
        0.0
    )
    irf_aug = irf_prompt + tail
    s = irf_aug.sum()
    if s > 0:
        irf_aug /= s
    x_orig      = np.arange(n_bins, dtype=float)
    irf_shifted = np.interp(x_orig - shift_bins, x_orig, irf_aug,
                            left=0.0, right=0.0)
    if sigma_bins > 0.05:
        irf_shifted = gaussian_filter1d(irf_shifted, sigma=sigma_bins)
        s2 = irf_shifted.sum()
        if s2 > 0:
            irf_shifted /= s2
    return irf_shifted

def _fwhm_ns(irf: np.ndarray, tcspc_res: float) -> float:
    pk = irf.max()
    if pk <= 0:
        return np.nan
    above = np.where(irf >= pk / 2)[0]
    if len(above) > 1:
        return (above[-1] - above[0]) * tcspc_res * 1e9
    integral = irf.sum() * tcspc_res * 1e9
    fwhm_est = integral * np.sqrt(4 * np.log(2) / np.pi)
    return float(fwhm_est)

def compare_irfs(irf_estimated:  np.ndarray,
                 xlsx:           dict | None,
                 tcspc_res:      float,
                 n_bins:         int,
                 strategy:       str,
                 out_prefix:     str) -> dict | None:
    t_ns = np.arange(n_bins, dtype=float) * tcspc_res * 1e9
    irf_xlsx_embedded = None
    if xlsx is not None and xlsx.get('irf_t') is not None and xlsx.get('irf_c') is not None:
        irf_raw = np.zeros(n_bins)
        for t, c in zip(xlsx['irf_t'], xlsx['irf_c']):
            idx = int(round(t / (tcspc_res * 1e9)))
            if 0 <= idx < n_bins:
                irf_raw[idx] += max(c, 0.0)
        s = irf_raw.sum()
        if s > 0:
            irf_xlsx_embedded = irf_raw / s
    if irf_xlsx_embedded is None:
        print('  IRF comparison skipped - no xlsx IRF available.')
        return None
    est = irf_estimated / irf_estimated.sum()
    ref = irf_xlsx_embedded / irf_xlsx_embedded.sum()
    peak_est_bin = int(np.argmax(est))
    peak_ref_bin = int(np.argmax(ref))
    shift_bins = peak_est_bin - peak_ref_bin
    x = np.arange(n_bins, dtype=float)
    est_aligned = np.interp(x + shift_bins, x, est, left=0.0, right=0.0)
    s = est_aligned.sum()
    if s > 0:
        est_aligned /= s

    def _metrics(a, b, label):
        support = (a > 1e-8) | (b > 1e-8)
        a_s, b_s = a[support], b[support]
        if len(a_s) > 1 and a_s.std() > 0 and b_s.std() > 0:
            r = float(np.corrcoef(a_s, b_s)[0, 1])
        else:
            r = np.nan
        rmse = float(np.sqrt(np.mean((a - b)**2)))
        bc = float(np.sum(np.sqrt(a * b)))
        return dict(label=label, pearson_r=r, rmse=rmse,
                    overlap_score=max(0.0, 1.0 - rmse), bhattacharyya=bc)
    m_raw = _metrics(est,         ref, 'raw     (unaligned)')
    m_aligned = _metrics(est_aligned, ref, 'aligned (peak-shift corrected)')
    fwhm_est = _fwhm_ns(est, tcspc_res)
    fwhm_ref = _fwhm_ns(ref, tcspc_res)
    peak_est_ns = peak_est_bin * tcspc_res * 1e9
    peak_ref_ns = peak_ref_bin * tcspc_res * 1e9
    metrics = dict(
        fwhm_estimated_ns = fwhm_est,
        fwhm_xlsx_ns = fwhm_ref,
        peak_estimated_ns = peak_est_ns,
        peak_xlsx_ns = peak_ref_ns,
        peak_shift_ns = shift_bins * tcspc_res * 1e9,
        peak_shift_bins = shift_bins,
        raw = m_raw,
        aligned = m_aligned,
    )
    print(f"\n  IRF Comparison  ({strategy.split('peak_bin')[0].strip()}  vs  xlsx)")
    print(f"  {'Metric':<28} {'Estimated':>12} {'xlsx':>12}")
    print(f"  {'─'*54}")
    print(f"  {'FWHM (ns)':<28} {fwhm_est:>12.4f} {fwhm_ref:>12.4f}")
    print(f"  {'Peak position (ns)':<28} {peak_est_ns:>12.4f} {peak_ref_ns:>12.4f}")
    print(f"  {'Peak shift (est − xlsx)':<28} "
          f"{shift_bins * tcspc_res * 1e9:>+11.4f} ns  ({shift_bins:+d} bins)")
    print(f"  {'─'*54}")
    for m in (m_raw, m_aligned):
        print(f"  [{m['label']}]")
        print(f"    {'Pearson r':<26} {m['pearson_r']:>12.4f}")
        print(f"    {'RMSE (normalised)':<26} {m['rmse']:>12.6f}")
        print(f"    {'Overlap score (1−RMSE)':<26} {m['overlap_score']:>12.4f}")
        print(f"    {'Bhattacharyya coeff.':<26} {m['bhattacharyya']:>12.4f}")
    bc_a = m_aligned['bhattacharyya']
    if bc_a >= 0.99:
        print(f"\n  Excellent shape match after alignment (BC={bc_a:.4f})")
        print(f"    → Use --irf-fwhm with adjusted peak; shape is correct.")
    elif bc_a >= 0.90:
        print(f"\n  ~ Acceptable shape match after alignment (BC={bc_a:.4f})")
        print(f"    → Shape is reasonable but consider --xlsx for fitting.")
    else:
        print(f"\n  Poor shape match even after alignment (BC={bc_a:.4f})")
        print(f"    → FWHM or IRF model is wrong. Use --xlsx for fitting.")
    if abs(shift_bins) >= 2:
        print(f"  Peak misaligned by {shift_bins:+d} bins ({shift_bins*tcspc_res*1e12:+.0f} ps) "
              f"- IRF peak bin estimate may be off.")
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 10,
                          'axes.spines.top': False, 'axes.spines.right': False,
                          'text.color': 'black', 'axes.labelcolor': 'black',
                          'xtick.color': 'black', 'ytick.color': 'black',
                          'axes.titlecolor': 'black'})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('IRF Comparison - Estimated vs FLIM microscope xlsx',
                 fontsize=11, fontweight='bold')
    support = (est > 1e-8) | (ref > 1e-8)
    idx_sup = np.where(support)[0]
    if len(idx_sup):
        x_lo = max(0,        idx_sup[0]  - 10) * tcspc_res * 1e9
        x_hi = min(n_bins-1, idx_sup[-1] + 10) * tcspc_res * 1e9
    else:
        x_lo, x_hi = t_ns[0], t_ns[-1]
    row_labels = ['Unaligned', 'Peak-aligned']
    for row, (e_plot, m) in enumerate([(est, m_raw), (est_aligned, m_aligned)]):
        diff = e_plot - ref
        axes[row, 0].plot(t_ns, ref,    'b-',  lw=2,   label='xlsx IRF')
        axes[row, 0].plot(t_ns, e_plot, 'r--', lw=1.8, label='estimated')
        axes[row, 0].set_xlim(x_lo, x_hi)
        axes[row, 0].set_ylabel('Normalised amplitude')
        axes[row, 0].set_title(f"{row_labels[row]} - linear")
        axes[row, 0].legend(fontsize=8)
        if row == 1:
            axes[row, 0].set_xlabel('Time (ns)')
        axes[row, 1].semilogy(t_ns, np.clip(ref,    1e-8, None), 'b-',  lw=2)
        axes[row, 1].semilogy(t_ns, np.clip(e_plot, 1e-8, None), 'r--', lw=1.8)
        axes[row, 1].set_xlim(x_lo, x_hi)
        axes[row, 1].set_title(f"{row_labels[row]} - log")
        if row == 1:
            axes[row, 1].set_xlabel('Time (ns)')
        axes[row, 2].fill_between(t_ns, diff, where=diff >= 0,
                                   alpha=0.6, color='#e63946', label='est > xlsx')
        axes[row, 2].fill_between(t_ns, diff, where=diff < 0,
                                   alpha=0.6, color='#457b9d', label='est < xlsx')
        axes[row, 2].axhline(0, color='k', lw=0.8, ls='--')
        axes[row, 2].set_xlim(x_lo, x_hi)
        axes[row, 2].set_ylabel('Δ (estimated − xlsx)')
        axes[row, 2].set_title(f"Difference  RMSE={m['rmse']:.5f}")
        axes[row, 2].legend(fontsize=8)
        if row == 1:
            axes[row, 2].set_xlabel('Time (ns)')
        txt = (f"Pearson r = {m['pearson_r']:.4f}\n"
               f"BC        = {m['bhattacharyya']:.4f}\n"
               f"FWHM est  = {fwhm_est:.4f} ns\n"
               f"FWHM xlsx = {fwhm_ref:.4f} ns")
        if row == 0:
            txt += f"\nΔpeak = {shift_bins*tcspc_res*1e12:+.0f} ps ({shift_bins:+d} bins)"
        axes[row, 2].text(0.97, 0.97, txt, transform=axes[row, 2].transAxes,
                          va='top', ha='right', fontsize=8, family='monospace',
                          bbox=dict(boxstyle='round,pad=0.3', fc='#f7f7f7', alpha=0.9))
    plt.tight_layout()
    out = f"{out_prefix}_irf_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return metrics