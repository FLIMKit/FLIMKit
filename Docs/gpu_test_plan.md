# GPU test plan

Everything that needs a GPU to check. The Mac covers MLX on Metal, so what is
missing is CUDA, ROCm, and the paths that only diverge under a discrete card
with its own memory.

Run the sections in order. A through C are quick and tell you whether the rest
is worth starting. D through G are the ones that can find a real defect. H
through J are optional.

Record the machine once at the top of your notes: GPU name, driver, CUDA or
ROCm version, `torch.__version__`, RAM, and OS.

## Runs so far

Two cards, both through section C.

| | RTX 3070 Ti | RTX A5000 |
| --- | --- | --- |
| VRAM | 8 GB | 25.4 GB |
| Grid scan, 1024 square | 14.9 s against 67.8 s | 8.35 s against 52.1 s |
| Fixed tau, 512 square | not recorded | 1.24 s against 10.2 s |
| Distribution, 128 square | not recorded | 0.39 s against 55.1 s |
| Free tau, 128 square | not recorded | 125.9 s against 70.4 s |
| 32 MB block disagreement | 10 pixels, 0.009954542882180561 ns | identical |
| Peak device memory | not recorded | 8.61 GB |

The A5000 run was on Python 3.14.7, torch 2.13.0+cu126, numpy 2.4.2, 24 cores,
269 GB of memory.

Three things came out of it.

The free tau row is the wrong way round. Asking for the GPU makes that fit 1.8
times slower than not asking, because `_scipy_parallel_free_tau_fit` in
`_base.py` runs a thread pool over scipy while the CPU path runs a process
pool. See G1 and G2.

The 32 MB disagreement reproduced to sixteen digits on a different card, which
rules out device nondeterminism and leaves adjacent grid points tying at a
block boundary.

Smaller blocks were faster, not slower: 1.10 s at 32 MB, 2.10 s at the 256 MB
default, 4.83 s at 4096 MB. That is worth a look on its own.

## Before anything

```bash
git fetch && git checkout bench/all-gpu-kernels
pip install -e '.[torch]'          # or the CUDA wheel index for your card
pip install pytest psutil
```

Two environment rules that bite:

- Run pytest from inside `flimkit_tests/`. The `pytest.ini` there has a
  `[tool:pytest]` header, which is `setup.cfg` syntax, so pytest ignores the
  whole file and the `pythonpath` in it does nothing.
- Any standalone script that calls `fit_per_pixel` or `fit_summed` with
  `workers != 1` needs an `if __name__ == '__main__':` guard. Windows and macOS
  spawn, so without the guard the script re-imports itself and fork-bombs.

Set `PYTHONPATH` to the checkout for anything run from outside it.

## A. Environment sanity

**A1. The backend is found at all.**

```bash
python -c "
from flimkit.GPU import get_backend
b = get_backend()
print(type(b).__name__ if b else 'none')
"
```

Pass: prints `CudaBackend` or `RocmBackend`. If it prints `none`, stop and
work out why, because everything below silently falls back to the CPU and
still passes.

**A2. Selection order is respected.**

```bash
python -c "
from flimkit.GPU import get_backend
for name in ('auto', 'cuda', 'rocm', 'mps', 'mlx'):
    b = get_backend(name)
    print(f'{name:6} {type(b).__name__ if b else None}')
"
```

Pass: `auto` and the card's own name give a backend, the other three give
`None` on a non-Apple machine. `mlx` returning anything other than `None` off
Darwin is a bug.

**A3. The card is actually used.** Watch `nvidia-smi -l 1` or `rocm-smi`
during section C and confirm utilisation moves. A backend that loads but does
no device work is exactly the failure mode section G is about.

## B. The automated suite

```bash
cd flimkit_tests
python -m pytest tests/ -q
```

Pass: the same count as on a CPU-only box, with the GPU-gated tests now
running instead of skipping. Check that specifically:

```bash
python -m pytest tests/test_cpu_gpu_parity.py tests/test_gpu_backends.py -v -rs
```

Pass: no `skipped` lines mentioning "No GPU backend available". If they still
skip, A1 lied.

The parity file compares CPU and GPU maps at 8 per cent on amplitudes and
lifetimes, 12 per cent on tau1. Those tolerances are loose on purpose because
the GPU path grid-scans where the CPU path optimises. A failure there is
meaningful; a pass is weak evidence.

## C. The benchmark script

```bash
python gpu_benchmark.py --sides 256,512,1024 --json bench_<gpu>.json
```

Runtime is roughly 20 to 40 minutes with the CPU reference included. Add
`--no-cpu` to cut it to a few minutes, at the cost of the comparison that
makes the numbers mean anything.

Pass criteria, all from the JSON:

| Field | Expect |
| --- | --- |
| `backend` | the card, not `none` |
| grid sweep `tau_ns` | within 0.02 ns of 2.4 across all grid sizes |
| `speedup` at 1024 square | above 1.0, and rising with field size |
| `pixels_differing` in the block sweep | small, and `worst_gap_ns` at or below one grid step |
| chi-squared section | GPU and CPU agree to 1e-4 relative |

See the table above for what two cards have given at 1024 square. A card
slower than the CPU there is a finding.

`worst_gap_ns` deserves care. At 2 ns with the default 1600-point grid the
step is about 0.011 ns, so a gap at that size is two neighbouring grid points
tying, not a wrong answer. A gap of several steps is a wrong answer.

## D. The four per-pixel kernels

The benchmark exercises all four, but only at the defaults. These vary the
things the defaults hide.

**D1. `batch_grid_scan_1exp` across grid density.**

```bash
for n in 200 800 1600 3200 6400; do
  FLIMKIT_TAU_GRID_POINTS=$n python gpu_benchmark.py --quick --skip kernels,scaling,blocks,chi2 \
    --json grid_$n.json
done
```

Pass: recovered tau converges as the grid tightens and does not move by more
than the grid step between 3200 and 6400. A tau that keeps moving means the
grid bounds, not the density, are binding.

**D2. `batch_fixed_tau`, two and three exponentials.** Covered by the kernels
section of the benchmark. Pass: amplitudes non-negative, fractions summing to
1 within 1e-6, chi-squared finite on every valid pixel.

**D3. `batch_free_tau_fit`.** See G1 before spending time here.

**D4. `batch_dist_scan_unimodal`.** Pass: centre within 0.05 ns of 2.4 and
width strictly greater than the tau floor of 0.145 ns. A width pinned at the
floor with one level means the parameter vector was built wrong, which is the
bug that bit the Mac run.

**D5. Every kernel on an all-background field.**

```bash
python - <<'PY'
import numpy as np
from flimkit.GPU import get_backend
stack = np.zeros((64, 64, 256), dtype=np.uint16)
print('build a zero stack and run each kernel with min_photons=5')
PY
```

Pass: no exception, no NaN in the intensity map, every lifetime map entirely
NaN or masked. Silent NaN propagation into a valid-looking map is the failure
to look for.

## E. Memory and blocking

This is the section the Mac cannot test properly, because unified memory hides
the transfer.

**E1. Block budget sweep.** In the benchmark already. Rerun explicitly at the
extremes:

```bash
for mb in 8 32 256 1024 4096; do
  FLIMKIT_GPU_BLOCK_BYTES=$((mb*1024*1024)) python gpu_benchmark.py --quick \
    --skip grid,kernels,scaling,chi2 --json block_${mb}mb.json
done
```

Pass: the maps agree across every budget to within one grid step, and the 8 MB
case completes rather than thrashing. Timing may vary by several times; that
is fine and worth recording.

**E2. A field larger than the card.** Pick a side that puts the float32
working set past VRAM. On an 8 GB card, 2048 square at 459 bins is about 7.7
GB in one piece.

```bash
python gpu_benchmark.py --sides 2048 --skip grid,kernels,blocks,chi2 --no-cpu \
  --json oversize.json
```

Pass: it completes by blocking, rather than raising a CUDA out-of-memory
error. An OOM here is a real defect in `pixel_blocks`, since the whole point of
the budget is to never hand the device more than it has.

**E3. Memory is released.** Run E2 twice in one process and print
`torch.cuda.max_memory_allocated()` after each. Pass: the second run does not
report meaningfully more than the first. A rising figure means the maps or the
staged blocks are being held.

**E4. Contention with another process.** Start something that holds a few GB
on the card, then run E1's 4096 MB case. Pass: it either completes or fails
with a clear message. A hang is a finding.

## F. Fallback and selection

**F1. `use_gpu=False` really uses the CPU.** Pass: no device utilisation, and
the maps match a GPU run inside the parity tolerances.

**F2. Forcing an absent backend.**

```bash
python -c "
from flimkit.GPU import get_backend
print(get_backend('mlx'))
print(get_backend('nonsense'))
"
```

Pass: `None` for `mlx`, and a `ValueError` naming the valid choices for the
last one.

**F3. Fallback when the device dies mid-fit.** Hard to stage honestly. Skip
unless you have a way to trigger it.

**F4. Interrupt.** Ctrl-C a 1024 square per-pixel fit halfway. Pass: the
process exits, the card is released, and no orphan worker survives. Check with
`nvidia-smi` afterwards.

**F5. Progress callback and cancellation.** Run a per-pixel fit through the
GUI or the bridge with a cancel event and cancel it. Pass: it stops within a
block, not at the end of the whole field.

## G. Paths that are suspect already

These are known or half-known. Confirming them on a second machine is the
point of the exercise.

**G1. `batch_free_tau_fit` does no GPU work.** Both `torch_backend.py:219` and
`mlx_backend.py:328` bind the device module, then hand everything to
`_scipy_parallel_free_tau_fit`, which runs on the CPU under the GIL. On the Mac
the copies were 0.01 s of 74.2 s.

Confirmed on the A5000. It is worse than dead weight: 125.9 s against 70.4 s
for the same field and the same answer, so the GPU route costs 55 seconds.
`_base.py:339` runs a `ThreadPoolExecutor` over scipy `least_squares`, which
holds the GIL, where the CPU path uses a process pool.

Test on any further machine: run a free-tau per-pixel fit with `nvidia-smi -l 1`
open. Utilisation stays near zero and wall time tracks cores rather than the
card. Record both timings, since the gap grows with core count.

**G2. Pile-up correction is discarded on the free-tau path.** Same two
functions compute `dc_flat`, apply Coates to it, and then fit `raw_valid`,
which is taken from `flat` and never sees the correction.

```bash
python - <<'PY'
import numpy as np
print('fit the same bright stack twice, correct_pileup False then True,')
print('n_sync large enough to matter, free_tau=True, n_exp=2')
print('then compare tau maps')
PY
```

Pass, meaning the bug is confirmed: the two tau maps are bit-identical. If
they differ, the correction is reaching the fit after all and the reading of
the code is wrong. This is the one test here whose expected result is a
defect, so record whichever way it goes.

Repeat with `free_tau=False` as the control. There the two runs must differ.

**G3. Tail fits fall back to the CPU by design.** `fitters.py:647` prints a
line saying so. Pass: the message appears and no device work happens. Nothing
to fix, just confirm the message is accurate on this machine.

**G4. Fit window plus time-varying background, one exponential.**
`fitters.py:651` routes that combination to the CPU. Pass: the message appears
and the result matches a plain CPU run.

**G5. The silent stretch after the last printed line.** On the Mac a
single-field QuPath fit looked hung because the per-pixel loop prints nothing
once it starts. Time it here and note whether the same gap appears. If it does,
that is an argument for a progress line rather than a bug.

## H. Precision

Undecided, and the reason to gather numbers rather than opinions.

Running chi-squared and the model in float32 measured 7.9 times faster on
CUDA. It changes fitted numbers, so it would need a `fitter_version` bump.

Test: fit the synthetic validation set at float64 and float32 and tabulate the
tau difference per file. The set is at `~/Downloads/flimkit_synth_validation/`
with known truth, 10 mono, 10 bi, 10 tri.

Pass, in the sense of being worth doing: the float32 tau sits within 0.01 ns of
float64 across all 30, and chi-squared agrees to three significant figures. A
larger spread on the tri-exponential files would settle it the other way.

## I. Stress

**I1. Long run.** A 2048 square fit, or the multi-dimensional stitch set at
`~/Downloads/20260326_oveinght.sptw`. Pass: completes without thermal
throttling changing the answer, which it should not, and without a memory
climb.

**I2. Two fits at once.** Two processes on one card. Pass: both complete and
agree with a serial run. This is worth knowing because the bridge can be asked
for a second fit while one is running.

**I3. Repeated small fits.** A hundred 64 square fits in one process. Pass:
per-fit time flat, no allocation growth. Catches per-call leaks that a single
large fit hides.

## J. Real data end to end

**J1. A FALCON tile through `tile_fit`.** Pass: the lifetime map matches a CPU
run inside the parity tolerances, and the intensity map is identical, since
intensity is a sum and should not vary at all.

**J2. Through the QuPath bridge.** Start `flimkit-bridge`, fit a region, and
confirm the maps arrive. Pass: the lifetime image opens in QuPath with the ns
scaling intact.

**J3. Against SPCImage or LAS X, if the machine has either.** Not a pass or
fail, just a number for the validation table.

## Reporting

Keep the JSON files. Send back `bench_<gpu>.json` plus the pytest summary, and
say which of G1, G2 and G5 reproduced. Those three are the ones that change
what gets fixed.
