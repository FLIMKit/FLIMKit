import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, '.')

GRIDS = (200, 800, 1600, 3200)
CHI2_SIZES = (20_000, 262_144)
FIELD_SIDES = (256, 512, 1024)
BLOCK_BUDGETS_MB = (32, 256, 4096)
SCALING_GRID = 1600
HEAVY_SIDE = 128
N_BINS = 459
TCSPC_RES = 97e-12
TRUE_TAU_NS = 2.4
PHOTONS_PER_PIXEL = 300


def describe_machine():
    found = {
        'os': f'{platform.system()} {platform.release()}',
        'machine': platform.machine(),
        'python': platform.python_version(),
        'numpy': np.__version__,
    }
    try:
        import psutil
        found['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
        found['cpu_cores'] = psutil.cpu_count(logical=False)
    except Exception:
        found['ram_gb'] = None
        found['cpu_cores'] = None
    try:
        import torch
        found['torch'] = torch.__version__
        found['torch_cuda'] = bool(torch.cuda.is_available())
        if found['torch_cuda']:
            found['gpu_name'] = torch.cuda.get_device_name(0)
            found['gpu_memory_gb'] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            found['hip'] = getattr(torch.version, 'hip', None)
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            found['gpu_name'] = 'Apple Metal (MPS)'
    except Exception:
        found['torch'] = None
    try:
        import mlx.core as mx
        found['mlx'] = True
        found['mlx_device'] = str(mx.default_device())
    except Exception:
        found['mlx'] = False
    return found


@contextlib.contextmanager
def quiet():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


def pick_backend():
    from flimkit.FLIM.fitters import _init_gpu_backend
    backend = _init_gpu_backend()
    return backend, (type(backend).__name__ if backend is not None else None)


def torch_backend_if_any():
    try:
        import torch
        from flimkit.GPU.torch_backend import TorchBackend
    except Exception:
        return None, None
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = 'mps'
    else:
        return None, None
    try:
        return TorchBackend(device=device), f'TorchBackend({device})'
    except Exception:
        return None, None


def synthetic_field(side, n_bins, seed=3):
    rng = np.random.default_rng(seed)
    axis = np.arange(n_bins) * TCSPC_RES
    peak = 20
    irf = np.exp(-0.5 * ((np.arange(n_bins) - peak) / 2.0) ** 2)
    irf = irf / irf.sum()
    pure = np.exp(-axis / (TRUE_TAU_NS * 1e-9))
    shape = np.convolve(pure, irf)[:n_bins]
    shape = shape / shape.sum()
    stack = np.empty((side, side, n_bins), dtype=np.uint16)
    step = max(1, 4_000_000 // (side * n_bins))
    for row in range(0, side, step):
        rows = min(step, side - row)
        drawn = rng.poisson(shape[None, None, :] * PHOTONS_PER_PIXEL,
                            size=(rows, side, n_bins))
        stack[row:row + rows] = np.minimum(drawn, 65535).astype(np.uint16)
    return stack, irf


def prepare_fit(stack, irf, n_bins):
    from flimkit.FLIM.fitters import fit_summed
    from flimkit.configs import Tau_min, Tau_max
    decay = stack.reshape(-1, n_bins).sum(axis=0).astype(float)
    with quiet():
        popt, _ = fit_summed(decay, TCSPC_RES, n_bins, irf, False, True, False, 1,
                             Tau_min, Tau_max, cost_function='poisson')
    return popt, Tau_min, Tau_max


def two_exp_popt(popt):
    tau, amp, shift, bg = popt[0], popt[1], popt[2], popt[3]
    return np.array([tau * 0.6, tau * 1.6, amp * 0.5, amp * 0.5, shift, bg])


def dist_popt(popt):
    tau, amp, shift = popt[0], popt[1], popt[2]
    return np.array([tau, tau * 0.125, amp, shift])


def load_fitters(points):
    import importlib
    os.environ['FLIMKIT_TAU_GRID_POINTS'] = str(points)
    from flimkit.FLIM import fitters
    return importlib.reload(fitters)


def summarise(maps, seconds, note=''):
    tau = np.asarray(maps['tau_mean_amp'], dtype=float)
    good = np.isfinite(tau)
    entry = {
        'seconds': round(seconds, 3),
        'pixels': int(good.sum()),
        'median_tau_ns': round(float(np.median(tau[good])), 5) if good.any() else None,
        'levels': int(len(np.unique(np.round(tau[good], 9)))) if good.any() else 0,
    }
    if entry['median_tau_ns'] is not None:
        entry['error_vs_truth_ns'] = round(entry['median_tau_ns'] - TRUE_TAU_NS, 5)
    if note:
        entry['note'] = note
    return entry, tau


def run_per_pixel(fitters, stack, irf, popt, bounds, n_bins, n_exp, free_tau,
                  use_gpu, backend):
    lo, hi = bounds
    with quiet() as log:
        started = time.time()
        maps = fitters.fit_per_pixel(
            stack, TCSPC_RES, n_bins, irf, has_tail=False, fit_bg=True,
            fit_sigma=False, global_popt=popt, n_exp=n_exp, min_photons=50,
            free_tau=free_tau, tau_min_ns=lo, tau_max_ns=hi,
            use_gpu=use_gpu, gpu_backend=backend if use_gpu else None)
        taken = time.time() - started
    return summarise(maps, taken, fallback_note(log, use_gpu))


def run_dist(fitters, stack, irf, popt, bounds, n_bins, use_gpu, backend):
    lo, hi = bounds
    with quiet() as log:
        started = time.time()
        maps = fitters.fit_per_pixel_dist(
            stack, TCSPC_RES, n_bins, irf, popt, 1, 'gaussian',
            fit_bg=True, fit_sigma=False, min_photons=50,
            tau_min_ns=lo, tau_max_ns=hi,
            use_gpu=use_gpu, gpu_backend=backend if use_gpu else None)
        taken = time.time() - started
    return summarise(maps, taken, fallback_note(log, use_gpu))


def fallback_note(log, use_gpu):
    if not use_gpu:
        return ''
    text = log.getvalue()
    for phrase in ('exceeds GPU limit', 'using the CPU path', 'uses the CPU path'):
        if phrase in text:
            return 'fell back to the CPU'
    return ''


def chi2_variants(n_pixels, n_bins, seed=5):
    from flimkit.FLIM.fit_tools import chi2_terms
    rng = np.random.default_rng(seed)
    decay = rng.poisson(30, (n_pixels, n_bins)).astype(np.float32)
    basis = np.abs(rng.normal(1.0, 0.2, (n_pixels, n_bins)))
    model = np.abs(rng.normal(20, 5, n_pixels))[:, None] * basis + 3.0

    def as_float64():
        return chi2_terms(decay, model, axis=1)[0]

    def as_float32():
        d = np.asarray(decay, dtype=np.float32)
        m = np.asarray(model, dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            return np.sum((d - m) ** 2 / np.maximum(m, 1.0), axis=1)

    runners = {'numpy_float64': as_float64, 'numpy_float32': as_float32}

    try:
        import mlx.core as mx

        def as_mlx():
            d = mx.array(np.asarray(decay, dtype=np.float32))
            m = mx.array(np.asarray(model, dtype=np.float32))
            out = ((d - m) ** 2 / mx.maximum(m, 1.0)).sum(axis=1)
            mx.eval(out)
            return np.array(out)

        runners['mlx_float32'] = as_mlx
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = None
        if device is not None:
            def as_torch():
                d = torch.as_tensor(np.asarray(decay, dtype=np.float32), device=device)
                m = torch.as_tensor(np.asarray(model, dtype=np.float32), device=device)
                out = ((d - m) ** 2 / torch.clamp(m, min=1.0)).sum(dim=1)
                if device == 'cuda':
                    torch.cuda.synchronize()
                return out.cpu().numpy()

            runners[f'torch_float32_{device}'] = as_torch
    except Exception:
        pass

    found = {}
    reference = None
    for name, runner in runners.items():
        best = None
        out = None
        for _ in range(3):
            started = time.time()
            out = runner()
            taken = time.time() - started
            best = taken if best is None else min(best, taken)
        if reference is None:
            reference = np.asarray(out, dtype=np.float64)
            worst = 0.0
        else:
            got = np.asarray(out, dtype=np.float64)
            rel = np.abs(got - reference) / np.maximum(np.abs(reference), 1e-30)
            worst = float(rel.max())
        found[name] = {'ms': round(best * 1000, 1),
                       'worst_relative_difference': f'{worst:.2e}'}
    return found


def peak_memory_gb():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(usage / (1e9 if sys.platform == 'darwin' else 1e6), 2)
    except Exception:
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / 1e9, 2)
        except Exception:
            return None


def banner(text):
    print()
    print(text)
    print('-' * len(text))


def seconds_or_dash(value):
    return f'{value:.2f}s' if value is not None and value == value else '-'


def speedup(cpu_s, gpu_s):
    if not gpu_s or cpu_s is None or cpu_s != cpu_s:
        return 'n/a'
    return f'{cpu_s / gpu_s:.2f}x'


def worst_gap(left, right):
    if left is None or right is None:
        return None
    both = np.isfinite(left) & np.isfinite(right)
    if not both.any():
        return None
    return float(np.abs(left[both] - right[both]).max())


def kernel_table(title, rows):
    banner(title)
    print(f'  {"kernel":34} {"GPU":>9} {"CPU":>9} {"speedup":>8} '
          f'{"median tau":>11} {"GPU vs CPU":>12}')
    for name, row in rows.items():
        gpu_s = row.get('gpu', {}).get('seconds')
        cpu_s = row.get('cpu', {}).get('seconds')
        shown = row.get('cpu') or row.get('gpu') or {}
        median = shown.get('median_tau_ns')
        gap = row.get('worst_gpu_cpu_gap_ns')
        print(f'  {name:34} {seconds_or_dash(gpu_s):>9} {seconds_or_dash(cpu_s):>9} '
              f'{speedup(cpu_s, gpu_s):>8} '
              f'{(f"{median:.4f}ns" if median is not None else "-"):>11} '
              f'{(f"{gap:.2e}ns" if gap is not None else "-"):>12}')
        note = row.get('gpu', {}).get('note')
        if note:
            print(f'  {"":34} GPU run {note}')


def parse_sides(text, default):
    if not text:
        return default
    return tuple(int(part) for part in text.replace(',', ' ').split())


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark every FLIMKit per-pixel GPU kernel on this machine')
    parser.add_argument('--side', type=int, default=512,
                        help='field for the grid sweep, side x side pixels (default 512)')
    parser.add_argument('--sides', default='',
                        help='fields for the scaling sweep, comma separated '
                             f'(default {",".join(str(s) for s in FIELD_SIDES)})')
    parser.add_argument('--bins', type=int, default=N_BINS)
    parser.add_argument('--quick', action='store_true',
                        help='fewer grid sizes, smaller fields, no scaling sweep')
    parser.add_argument('--json', default='',
                        help='also write the results to this file')
    parser.add_argument('--no-cpu', action='store_true',
                        help='skip the CPU timings, which dominate the runtime')
    parser.add_argument('--skip', default='',
                        help='sections to leave out: grid, kernels, scaling, blocks, chi2')
    args = parser.parse_args()

    skip = {part.strip() for part in args.skip.replace(',', ' ').split()}
    machine = describe_machine()
    side = 256 if args.quick else args.side
    grids = (200, 1600) if args.quick else GRIDS
    sizes = (20_000,) if args.quick else CHI2_SIZES
    sides = parse_sides(args.sides, FIELD_SIDES)
    heavy = min(HEAVY_SIDE, side)
    ram = machine.get('ram_gb') or 8.0
    if not args.quick and ram < 12:
        sizes = (20_000, 65_536)
        print(f'note: {ram} GB of memory, so the large chi-squared case is '
              'reduced to 65,536 pixels')
    if args.quick:
        skip.add('scaling')
    else:
        room = ram * 0.4e9
        kept = tuple(s for s in sides if s * s * args.bins * 4 * 3 <= room)
        if kept != sides:
            dropped = [s for s in sides if s not in kept]
            print(f'note: {ram} GB of memory, so the scaling sweep drops '
                  f'{", ".join(str(d) + " square" for d in dropped)}')
            sides = kept or (min(sides),)
        if side > max(sides, default=side) and side * side * args.bins * 4 * 3 > room:
            side = max(sides)
            print(f'note: grid sweep field reduced to {side} square to fit in memory')

    banner('Machine')
    for key, value in machine.items():
        print(f'  {key:16s} {value}')

    try:
        backend, backend_name = pick_backend()
    except Exception as problem:
        print(f'\ncould not reach a GPU backend: {problem}')
        backend, backend_name = None, None
    print(f'  {"flimkit_backend":16s} {backend_name or "none, CPU only"}')
    torch_backend, torch_name = torch_backend_if_any()
    if torch_name and torch_name != backend_name:
        print(f'  {"also_available":16s} {torch_name}, '
              'the path CUDA and ROCm take')

    banner(f'Building a {side}x{side}x{args.bins} field')
    stack, irf = synthetic_field(side, args.bins)
    print(f'  {stack.nbytes / 1e9:.2f} GB as {stack.dtype}, '
          f'{PHOTONS_PER_PIXEL} photons per pixel, true tau {TRUE_TAU_NS} ns')
    print('  the lifetime is known, so the last column says how close the fit gets')
    popt, lo, hi = prepare_fit(stack, irf, args.bins)
    popt2 = two_exp_popt(popt)
    print(f'  summed fit gives {popt[0] * 1e9:.4f} ns, grid spans {lo} to {hi} ns')

    results = {'machine': machine, 'backend': backend_name,
               'torch_backend': torch_name,
               'field': {'side': side, 'bins': args.bins,
                         'gigabytes': round(stack.nbytes / 1e9, 3)}}

    fits = {}
    if 'grid' not in skip:
        banner('One exponential, free tau, grid scan (batch_grid_scan_1exp)')
        print(f'  {"grid":>6} {"step at 2ns":>12} {"GPU":>8} {"CPU":>8} {"speedup":>8} '
              f'{"levels":>7} {"median tau":>11} {"vs truth":>10}')
        for points in grids:
            fitters = load_fitters(points)
            row = {}
            if backend is not None:
                row['gpu'], _ = run_per_pixel(fitters, stack, irf, popt, (lo, hi),
                                              args.bins, 1, True, True, backend)
            if args.no_cpu and backend is not None:
                row['cpu'] = dict(row['gpu'], seconds=float('nan'))
            else:
                row['cpu'], _ = run_per_pixel(fitters, stack, irf, popt, (lo, hi),
                                              args.bins, 1, True, False, None)
            fits[points] = row
            step = 2.0 * ((hi / lo) ** (1.0 / (points - 1)) - 1.0)
            gpu_s = row.get('gpu', {}).get('seconds')
            cpu_s = row['cpu']['seconds']
            median = row['cpu']['median_tau_ns']
            error = row['cpu'].get('error_vs_truth_ns')
            print(f'  {points:>6} {step:>11.4f}ns {seconds_or_dash(gpu_s):>8} '
                  f'{seconds_or_dash(cpu_s):>8} {speedup(cpu_s, gpu_s):>8} '
                  f'{row["cpu"]["levels"]:>7} {median:>10.4f}ns {error:>+9.4f}ns')
        print(f'  fitted {fits[grids[0]]["cpu"]["pixels"]:,} pixels, '
              f'peak memory {peak_memory_gb()} GB')
    results['per_pixel'] = {str(k): v for k, v in fits.items()}

    kernels = {}
    if 'kernels' not in skip:
        fitters = load_fitters(SCALING_GRID)
        small = stack[:heavy, :heavy]
        cases = [
            ('fixed tau, 2 components', 'batch_fixed_tau', stack, popt2, 2, False),
            ('free tau, 2 components', 'batch_free_tau_fit', small, popt2, 2, True),
        ]
        for label, kernel, field, params, n_exp, free_tau in cases:
            row = {'kernel': kernel,
                   'side': int(field.shape[0]),
                   'gigabytes_float32': round(field.size * 4 / 1e9, 3)}
            gpu_tau = cpu_tau = None
            if backend is not None:
                row['gpu'], gpu_tau = run_per_pixel(
                    fitters, field, irf, params, (lo, hi), args.bins, n_exp,
                    free_tau, True, backend)
            if not args.no_cpu or backend is None:
                row['cpu'], cpu_tau = run_per_pixel(
                    fitters, field, irf, params, (lo, hi), args.bins, n_exp,
                    free_tau, False, None)
            row['worst_gpu_cpu_gap_ns'] = worst_gap(gpu_tau, cpu_tau)
            kernels[f'{label} ({field.shape[0]} sq)'] = row
        row = {'kernel': 'batch_dist_scan_unimodal',
               'side': heavy,
               'gigabytes_float32': round(small.size * 4 / 1e9, 3)}
        gpu_tau = cpu_tau = None
        if backend is not None:
            row['gpu'], gpu_tau = run_dist(fitters, small, irf, dist_popt(popt),
                                           (lo, hi), args.bins, True, backend)
        if not args.no_cpu or backend is None:
            row['cpu'], cpu_tau = run_dist(fitters, small, irf, dist_popt(popt),
                                           (lo, hi), args.bins, False, None)
        row['worst_gpu_cpu_gap_ns'] = worst_gap(gpu_tau, cpu_tau)
        kernels[f'gaussian distribution ({heavy} sq)'] = row
        kernel_table('Every other per-pixel kernel', kernels)
        print(f'  the field is one exponential at {TRUE_TAU_NS} ns, so a two-component '
              f'model has nothing to find and its tau only has to match between GPU and CPU')
        print(f'  the two heavy kernels run on {heavy} square so the script finishes; '
              f'neither is blocked yet')
    results['kernels'] = kernels

    scaling = {}
    if 'scaling' not in skip and backend is not None:
        banner('Field size, one exponential, free tau, 1600 points')
        print(f'  {"field":>11} {"float32 set":>12} {"GPU":>9} {"CPU":>9} '
              f'{"speedup":>8}')
        fitters = load_fitters(SCALING_GRID)
        for wide in sides:
            if wide == side:
                field = stack
            else:
                field, _ = synthetic_field(wide, args.bins)
            working = wide * wide * args.bins * 4
            row = {'gigabytes_float32': round(working / 1e9, 3)}
            row['gpu'], _ = run_per_pixel(fitters, field, irf, popt, (lo, hi),
                                          args.bins, 1, True, True, backend)
            if not args.no_cpu:
                row['cpu'], _ = run_per_pixel(fitters, field, irf, popt, (lo, hi),
                                              args.bins, 1, True, False, None)
            scaling[wide] = row
            gpu_s = row['gpu']['seconds']
            cpu_s = row.get('cpu', {}).get('seconds')
            print(f'  {f"{wide}x{wide}":>11} {working / 1e9:>11.2f}GB '
                  f'{seconds_or_dash(gpu_s):>9} {seconds_or_dash(cpu_s):>9} '
                  f'{speedup(cpu_s, gpu_s):>8}')
            if field is not stack:
                del field
        print('  anything past 1.00 GB used to be refused and sent to the CPU')
    results['scaling'] = {str(k): v for k, v in scaling.items()}

    blocks = {}
    if 'blocks' not in skip and backend is not None:
        banner('Block budget, same fit, 1600 points')
        print(f'  {"budget":>9} {"blocks":>7} {"seconds":>9} '
              f'{"agrees with the largest budget":>31}')
        fitters = load_fitters(SCALING_GRID)
        run_per_pixel(fitters, stack[:64, :64], irf, popt, (lo, hi), args.bins,
                      1, True, True, backend)
        previous = os.environ.get('FLIMKIT_GPU_BLOCK_BYTES')
        reference = None
        pixels = side * side
        try:
            for megabytes in sorted(BLOCK_BUDGETS_MB, reverse=True):
                os.environ['FLIMKIT_GPU_BLOCK_BYTES'] = str(megabytes * 1024 * 1024)
                entry, tau = run_per_pixel(fitters, stack, irf, popt, (lo, hi),
                                           args.bins, 1, True, True, backend)
                per_pixel = 4 * (3 * args.bins + SCALING_GRID)
                per_block = max(1, (megabytes * 1024 * 1024) // per_pixel)
                entry['blocks'] = int(np.ceil(pixels / per_block))
                if reference is None:
                    reference = tau
                    entry['identical_to_largest_budget'] = True
                else:
                    both = np.isfinite(tau) & np.isfinite(reference)
                    entry['identical_to_largest_budget'] = bool(
                        np.array_equal(tau[both], reference[both]))
                blocks[megabytes] = entry
                print(f'  {f"{megabytes} MB":>9} {entry["blocks"]:>7} '
                      f'{entry["seconds"]:>8.2f}s '
                      f'{str(entry["identical_to_largest_budget"]):>31}')
        finally:
            if previous is None:
                os.environ.pop('FLIMKIT_GPU_BLOCK_BYTES', None)
            else:
                os.environ['FLIMKIT_GPU_BLOCK_BYTES'] = previous
        print('  a smaller budget should only cost time, never change an answer')
    results['blocks'] = {str(k): v for k, v in blocks.items()}

    torch_fits = {}
    if torch_backend is not None and torch_name != backend_name and 'grid' not in skip:
        banner(f'Same one-exponential fit through {torch_name}')
        print(f'  {"grid":>6} {"seconds":>9} {"levels":>7} {"median tau":>11} '
              f'{"vs truth":>10}')
        for points in grids:
            fitters = load_fitters(points)
            row, _ = run_per_pixel(fitters, stack, irf, popt, (lo, hi), args.bins,
                                   1, True, True, torch_backend)
            torch_fits[points] = row
            print(f'  {points:>6} {row["seconds"]:>8.2f}s {row["levels"]:>7} '
                  f'{row["median_tau_ns"]:>10.4f}ns '
                  f'{row["error_vs_truth_ns"]:>+9.4f}ns')
    results['per_pixel_torch'] = {str(k): v for k, v in torch_fits.items()}

    chi2 = {}
    if 'chi2' not in skip:
        banner('Chi-squared kernel, milliseconds')
        for n_pixels in sizes:
            chi2[n_pixels] = chi2_variants(n_pixels, args.bins)
            print(f'  {n_pixels:,} pixels x {args.bins} bins')
            for name, entry in chi2[n_pixels].items():
                print(f'    {name:22s} {entry["ms"]:>8.1f} ms   '
                      f'differs from float64 by {entry["worst_relative_difference"]}')
    results['chi2'] = {str(k): v for k, v in chi2.items()}
    results['peak_memory_gb'] = peak_memory_gb()

    banner('Paste this back')
    print(json.dumps(results, indent=None, separators=(',', ':')))
    if args.json:
        with open(args.json, 'w') as handle:
            json.dump(results, handle, indent=2)
        print(f'\nalso written to {args.json}')


if __name__ == '__main__':
    main()
