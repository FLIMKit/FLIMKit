import sys, time, traceback, platform
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "flimkit_tests")

import psutil
from flimkit.GPU import get_backend
from flimkit.FLIM.fitters import fit_per_pixel
from flimkit.FLIM.irf_tools import gaussian_irf_from_fwhm
from flimkit_tests.mock_data import (
    MOCK_TCSPC_RES, MOCK_IRF_FWHM_BINS, MOCK_IRF_CENTER,
    MOCK_TAU1_NS, MOCK_TAU2_NS, MOCK_AMP1, MOCK_AMP2,
    generate_synthetic_biexp_decay,
)

N_BINS           = 256
TCSPC            = MOCK_TCSPC_RES
PEAK_COUNTS      = 3_000
TAU1_TRUE        = MOCK_TAU1_NS
TAU2_TRUE        = MOCK_TAU2_NS
MIN_PHOTONS      = 50
TIME_BUDGET_S    = 60.0
FREE_TAU_SAMPLE  = 128

# Canvas sizes to step through (square tiles)
SIZES = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]

# Typical real-world acquisition sizes for recommendations
TYPICAL = {
    "Single 512 tile":        512  * 512,
    "Single 1024 tile":       1024 * 1024,
    "2×2 mosaic (1024 total)": 1024 * 1024,
    "4×4 mosaic (2048 total)": 2048 * 2048,
    "Whole-slide (4096)":      4096 * 4096,
}

ram_gb   = psutil.virtual_memory().total / 1e9
cpu_cnt  = psutil.cpu_count(logical=False)
print(f"{'='*65}")
print(f"  FLIMKit Hardware Limits Test")
print(f"{'='*65}")
print(f"  OS       : {platform.system()} {platform.machine()}")
print(f"  RAM      : {ram_gb:.1f} GB")
print(f"  CPU cores: {cpu_cnt} physical")

backend = get_backend()
backend_name = type(backend).__name__ if backend else "None"
print(f"  GPU      : {backend_name}")
print(f"{'='*65}\n")

irf_fwhm_ns = MOCK_IRF_FWHM_BINS * TCSPC * 1e9
irf = gaussian_irf_from_fwhm(N_BINS, TCSPC, irf_fwhm_ns, MOCK_IRF_CENTER)
global_popt = np.array([TAU1_TRUE * 1e-9, TAU2_TRUE * 1e-9,
                         MOCK_AMP1, MOCK_AMP2, 0.0])

COMMON = dict(
    tcspc_res=TCSPC, n_bins=N_BINS, irf_prompt=irf,
    has_tail=False, fit_bg=False, fit_sigma=False,
    global_popt=global_popt, n_exp=2, min_photons=MIN_PHOTONS,
)

def _make_stack(ny, nx):
    """Build a synthetic bi-exp stack; reuse a single pixel tiled for speed."""
    pixel = generate_synthetic_biexp_decay(
        N_BINS, TCSPC, tau1_ns=TAU1_TRUE, tau2_ns=TAU2_TRUE,
        a1=MOCK_AMP1, a2=MOCK_AMP2, bg=5.0,
        peak_counts=PEAK_COUNTS, noise=False,
    ).astype(np.float32)
    return np.broadcast_to(pixel, (ny, nx, N_BINS)).copy()

def _stack_ram_mb(ny, nx):
    return ny * nx * N_BINS * 4 / 1e6

print(f"{'FIXED-τ GPU ramp':^65}")
print(f"{'Size':>12}  {'Pixels':>10}  {'Stack MB':>9}  "
      f"{'Time (s)':>9}  {'px/s':>10}  {'Status'}")
print("-" * 65)

gpu_results = []
prev_pxs = None

for sz in SIZES:
    ram_mb = _stack_ram_mb(sz, sz)
    available_mb = psutil.virtual_memory().available / 1e6
    if ram_mb > available_mb * 0.7:
        print(f"  {sz:>4}×{sz:<4}  {sz*sz:>10,}  {ram_mb:>8.0f}M  "
              f"{'':>9}  {'':>10}  SKIP — would exceed available RAM")
        break

    try:
        stack = _make_stack(sz, sz)
    except MemoryError:
        print(f"  {sz:>4}×{sz:<4}  {sz*sz:>10,}  {ram_mb:>8.0f}M  "
              f"{'':>9}  {'':>10}  OOM — RAM exhausted")
        break

    try:
        t0 = time.perf_counter()
        fit_per_pixel(**COMMON, stack=stack, free_tau=False, gpu_backend=backend)
        elapsed = time.perf_counter() - t0
        pxs = sz * sz / elapsed
        note = ""
        if prev_pxs is not None and pxs < prev_pxs * 0.7:
            note = " ← throughput drop (memory pressure?)"
        print(f"  {sz:>4}×{sz:<4}  {sz*sz:>10,}  {ram_mb:>8.0f}M  "
              f"{elapsed:>9.2f}  {pxs:>10,.0f}  {note}")
        gpu_results.append((sz, pxs, elapsed))
        prev_pxs = pxs
        del stack
        if elapsed > TIME_BUDGET_S:
            print(f"  (stopping ramp — exceeded {TIME_BUDGET_S:.0f}s budget)")
            break
    except Exception as e:
        print(f"  {sz:>4}×{sz:<4}  {sz*sz:>10,}  {ram_mb:>8.0f}M  "
              f"{'':>9}  {'':>10}  FAILED — {e}")
        break

print(f"\n{'FREE-τ CPU throughput sample':^65}")
print(f"  (sampling on {FREE_TAU_SAMPLE}×{FREE_TAU_SAMPLE} = "
      f"{FREE_TAU_SAMPLE**2:,} px)")

ft_stack = _make_stack(FREE_TAU_SAMPLE, FREE_TAU_SAMPLE)
t0 = time.perf_counter()
fit_per_pixel(**COMMON, stack=ft_stack, free_tau=True, use_gpu=False)
t_ft_cpu = time.perf_counter() - t0
pxs_ft_cpu = FREE_TAU_SAMPLE**2 / t_ft_cpu
del ft_stack

ft_stack_gpu = _make_stack(FREE_TAU_SAMPLE, FREE_TAU_SAMPLE)
t0 = time.perf_counter()
fit_per_pixel(**COMMON, stack=ft_stack_gpu, free_tau=True, gpu_backend=backend)
t_ft_gpu = time.perf_counter() - t0
pxs_ft_gpu = FREE_TAU_SAMPLE**2 / t_ft_gpu
del ft_stack_gpu

print(f"  Free-τ CPU (sequential) : {t_ft_cpu:.2f}s  →  {pxs_ft_cpu:,.0f} px/s")
print(f"  Free-τ GPU* (threaded)  : {t_ft_gpu:.2f}s  →  {pxs_ft_gpu:,.0f} px/s")
print(f"  (* GPU* = parallel CPU scipy, not hardware GPU)")

if gpu_results:
    best_sz, best_pxs, _ = max(gpu_results, key=lambda r: r[1])
    largest_sz, _, largest_t = gpu_results[-1]
else:
    best_sz = best_pxs = largest_sz = largest_t = None

print(f"\n{'='*65}")
print(f"  RECOMMENDATIONS")
print(f"{'='*65}")

if best_pxs:
    print(f"\n  Peak fixed-τ GPU throughput: {best_pxs:,.0f} px/s "
          f"(at {best_sz}×{best_sz})")

print(f"\n  Estimated wall-clock times for common acquisitions:")
print(f"  {'Acquisition':35s}  {'Pixels':>10}  {'Fixed-τ GPU':>12}  "
      f"{'Free-τ CPU':>12}  {'Free-τ GPU*':>12}")
print(f"  {'-'*83}")

for name, total_px in TYPICAL.items():
    if best_pxs:
        t_gpu_est = total_px / best_pxs
        gpu_str = (f"{t_gpu_est:.1f}s" if t_gpu_est < 60
                   else f"{t_gpu_est/60:.1f} min")
    else:
        gpu_str = "n/a"
    t_cpu_est = total_px / pxs_ft_cpu
    cpu_str = (f"{t_cpu_est:.1f}s" if t_cpu_est < 60
               else f"{t_cpu_est/60:.1f} min")
    t_gfree_est = total_px / pxs_ft_gpu
    gfree_str = (f"{t_gfree_est:.1f}s" if t_gfree_est < 60
                 else f"{t_gfree_est/60:.1f} min")
    print(f"  {name:35s}  {total_px:>10,}  {gpu_str:>12}  "
          f"{cpu_str:>12}  {gfree_str:>12}")

# RAM headroom check
print(f"\n  RAM available now: "
      f"{psutil.virtual_memory().available/1e9:.1f} GB  "
      f"/ {ram_gb:.1f} GB total")
for sz in [1024, 2048, 4096]:
    mb = _stack_ram_mb(sz, sz)
    fits = "✓ fits" if mb < psutil.virtual_memory().available / 1e6 * 0.7 else "✗ too large"
    print(f"  Stack {sz}×{sz}×{N_BINS} float32: {mb/1000:.1f} GB  → {fits}")

print(f"\n  Upper limits for interactive use (< 10 s):")
if best_pxs:
    max_px_interactive = int(best_pxs * 10)
    side = int(max_px_interactive ** 0.5)
    print(f"    Fixed-τ GPU : ~{side}×{side} canvas ({max_px_interactive:,} px)")
max_px_cpu = int(pxs_ft_cpu * 10)
side_cpu = int(max_px_cpu ** 0.5)
print(f"    Free-τ CPU  : ~{side_cpu}×{side_cpu} canvas ({max_px_cpu:,} px)")

print(f"\n  Note: fixed-τ GPU assumes τ values are fixed from the global fit.")
print(f"  Free-τ is needed when per-pixel lifetime variation is expected (e.g. FRET).")
print(f"{'='*65}")
