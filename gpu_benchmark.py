import argparse
import json
import platform
import sys
import time

import numpy as np

sys.path.insert(0, '.')

GRIDS = (200, 800, 1600, 3200)
CHI2_SIZES = (20_000, 262_144)
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


def pick_backend():
    from flimkit.FLIM.fitters import _init_gpu_backend
    backend = _init_gpu_backend()
    return backend, (type(backend).__name__ if backend is not None else None)


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
    popt, _ = fit_summed(decay, TCSPC_RES, n_bins, irf, False, True, False, 1,
                         Tau_min, Tau_max, cost_function='poisson')
    return popt, Tau_min, Tau_max


def time_per_pixel(stack, irf, popt, bounds, n_bins, points, use_gpu, backend,
                   repeats=1):
    import importlib
    import os
    os.environ['FLIMKIT_TAU_GRID_POINTS'] = str(points)
    from flimkit.FLIM import fitters
    importlib.reload(fitters)
    lo, hi = bounds
    best = None
    maps = None
    for _ in range(repeats):
        started = time.time()
        maps = fitters.fit_per_pixel(
            stack, TCSPC_RES, n_bins, irf, has_tail=False, fit_bg=True,
            fit_sigma=False, global_popt=popt, n_exp=1, min_photons=50,
            free_tau=True, tau_min_ns=lo, tau_max_ns=hi,
            use_gpu=use_gpu, gpu_backend=backend if use_gpu else None)
        taken = time.time() - started
        best = taken if best is None else min(best, taken)
    tau = np.asarray(maps['tau_mean_amp'])
    good = np.isfinite(tau)
    return {
        'seconds': round(best, 3),
        'pixels': int(good.sum()),
        'median_tau_ns': round(float(np.median(tau[good])), 5) if good.any() else None,
        'levels': int(len(np.unique(np.round(tau[good], 9)))) if good.any() else 0,
    }


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


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark FLIMKit per-pixel fitting on this machine')
    parser.add_argument('--side', type=int, default=512,
                        help='field is side x side pixels (default 512)')
    parser.add_argument('--bins', type=int, default=N_BINS)
    parser.add_argument('--quick', action='store_true',
                        help='fewer grid sizes and a smaller field')
    parser.add_argument('--json', default='',
                        help='also write the results to this file')
    args = parser.parse_args()

    side = 256 if args.quick else args.side
    grids = (200, 1600) if args.quick else GRIDS
    sizes = (20_000,) if args.quick else CHI2_SIZES

    machine = describe_machine()
    banner('Machine')
    for key, value in machine.items():
        print(f'  {key:16s} {value}')

    try:
        backend, backend_name = pick_backend()
    except Exception as problem:
        print(f'\ncould not reach a GPU backend: {problem}')
        backend, backend_name = None, None
    print(f'  {"flimkit_backend":16s} {backend_name or "none, CPU only"}')

    banner(f'Building a {side}x{side}x{args.bins} field')
    stack, irf = synthetic_field(side, args.bins)
    print(f'  {stack.nbytes / 1e9:.2f} GB as {stack.dtype}, '
          f'{PHOTONS_PER_PIXEL} photons per pixel, true tau {TRUE_TAU_NS} ns')
    popt, lo, hi = prepare_fit(stack, irf, args.bins)
    print(f'  summed fit gives {popt[0] * 1e9:.4f} ns, grid spans {lo} to {hi} ns')

    banner('Per-pixel fit, seconds')
    print(f'  {"grid":>6} {"step at 2ns":>12} {"GPU":>9} {"CPU":>9} {"speedup":>8} '
          f'{"levels":>7} {"median tau":>11}')
    fits = {}
    for points in grids:
        row = {}
        if backend is not None:
            row['gpu'] = time_per_pixel(stack, irf, popt, (lo, hi), args.bins,
                                        points, True, backend)
        row['cpu'] = time_per_pixel(stack, irf, popt, (lo, hi), args.bins,
                                    points, False, None)
        fits[points] = row
        step = 2.0 * ((hi / lo) ** (1.0 / (points - 1)) - 1.0)
        gpu_s = row['gpu']['seconds'] if 'gpu' in row else None
        cpu_s = row['cpu']['seconds']
        speed = f'{cpu_s / gpu_s:.2f}x' if gpu_s else 'n/a'
        print(f'  {points:>6} {step:>11.4f}ns '
              f'{(f"{gpu_s:.2f}" if gpu_s else "-"):>9} {cpu_s:>8.2f}s {speed:>8} '
              f'{row["cpu"]["levels"]:>7} {row["cpu"]["median_tau_ns"]:>10.4f}ns')
    print(f'  fitted {fits[grids[0]]["cpu"]["pixels"]:,} pixels, '
          f'peak memory {peak_memory_gb()} GB')

    banner('Chi-squared kernel, milliseconds')
    chi2 = {}
    for n_pixels in sizes:
        chi2[n_pixels] = chi2_variants(n_pixels, args.bins)
        print(f'  {n_pixels:,} pixels x {args.bins} bins')
        for name, entry in chi2[n_pixels].items():
            print(f'    {name:22s} {entry["ms"]:>8.1f} ms   '
                  f'differs from float64 by {entry["worst_relative_difference"]}')

    results = {'machine': machine, 'backend': backend_name,
               'field': {'side': side, 'bins': args.bins,
                         'gigabytes': round(stack.nbytes / 1e9, 3)},
               'per_pixel': {str(k): v for k, v in fits.items()},
               'chi2': {str(k): v for k, v in chi2.items()},
               'peak_memory_gb': peak_memory_gb()}

    banner('Paste this back')
    print(json.dumps(results, indent=None, separators=(',', ':')))
    if args.json:
        with open(args.json, 'w') as handle:
            json.dump(results, handle, indent=2)
        print(f'\nalso written to {args.json}')


if __name__ == '__main__':
    main()
