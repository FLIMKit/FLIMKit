# FLIMKit Documentation

> **v0.9.9** — Python toolkit for Fluorescence Lifetime Imaging Microscopy

> **Warning:** Active development. Cross-validate results with other software before drawing conclusions.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements & Installation](#requirements--installation)
3. [Quick Start](#quick-start)
4. [Workflows](#workflows)
   - [Desktop GUI](#desktop-gui)
   - [Guided Terminal UI](#guided-terminal-ui-mainpy)
   - [Machine IRF Setup](#machine-irf-setup-required)
   - [FLIM Reconvolution Fitting (CLI)](#flim-reconvolution-fitting-cli)
   - [Phasor Analysis (CLI)](#phasor-analysis-cli)
   - [Python API](#python-api)
5. [Configuration Reference](#configuration-reference)
6. [Module Reference](#module-reference)
7. [Project Structure](#project-structure)
8. [Compiled App](#compiled-app-macos--windows--linux)
9. [Testing](#testing)
10. [Outputs & File Formats](#outputs--file-formats)
11. [Troubleshooting](#troubleshooting)
12. [Contact](#contact)

---

## Overview

FLIMKit handles FLIM data from Leica SP8/FALCON systems (or any PTU-based setup). It's designed as a replacement for Leica LAS X FLIM analysis, with two main workflows:

| Workflow | Description |
|---|---|
| **Reconvolution fitting** | Mono/bi/tri-exponential lifetime fitting with full IRF deconvolution, per-pixel and summed modes, multi-tile ROI stitching, and batch processing |
| **Phasor analysis** | Calibrated phasor plots with interactive elliptical cursors, automatic peak detection, two-component decomposition, and session save/load |

Both are accessible through a desktop GUI, guided terminal UI, CLI scripts, or the Python API.

---

## Requirements & Installation

### System Requirements

- Python ≥ 3.12
- macOS, Linux, or Windows

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array computation |
| `scipy` | Optimisers (Levenberg–Marquardt, Differential Evolution), signal processing |
| `matplotlib` | Plotting (decay curves, lifetime maps, phasor plots) |
| `xarray` | Labelled N-D arrays for FLIM signals |
| `phasorpy` (0.10) | Phasor computation, calibration, cursor masking, spatial filtering, lifetime conversion |
| `PyWavelets` | Wavelet-based phasor denoising |
| `ptufile` | Low-level PTU file reading |
| `inquirer` | Interactive terminal prompts |
| `ipywidgets` + `ipympl` | Jupyter notebook interactive support |
| `opencv-python` | Cell masking and image processing |
| `openpyxl` | Excel XLSX parsing for LAS X IRF extraction |
| `pandas` | Excel/XLSX IRF file parsing |
| `tifffile` | TIFF image I/O |
| `tqdm` | Progress bars |

### Installation

```bash
git clone https://github.com/alex1075/FLIMKit.git
cd FLIMKit
python install.py
```

`install.py` installs core requirements, then auto-detects and installs the right GPU backend (MLX on Apple Silicon, CUDA on NVIDIA, ROCm on AMD, CPU-only fallback). No flags needed for a standard install.

```bash
python install.py --dev      # also installs PyInstaller and test requirements
python install.py --dry-run  # preview commands without executing
```

### Validate Installation

```bash
python validate_installation.py
```

Runs 10 checks: dependencies, module imports, XLIF parsing, stitching, fitting, phasor pipeline, per-tile fit pipeline, and GPU backend dispatch. All should pass.

### Hardware Limits

```bash
python hardware_limits.py
```

Stress-tests the machine by ramping canvas sizes (64×64 → 4096×4096) and measuring fixed-tau GPU and free-tau CPU throughput. Reports peak pixels/second, RAM headroom, and estimated wall-clock times for common acquisition sizes. Useful for understanding what canvas sizes are feasible before starting a long batch run.

---

## Quick Start

Download the compiled app from the Releases tab if you don't want to deal with Python.

Double-click to run. macOS will complain it's unsigned. That's expected, a dev certificate costs money and this is free. Right-click → Open to bypass Gatekeeper on first launch.

From source:

```bash
python main.py               # desktop GUI
python main.py --cli         # guided terminal UI

python fit_cli.py --ptu data.ptu --machine-irf machine_irf_default.npy --nexp 2
python phasor_cli.py --ptu data.ptu --irf irf.xlsx
```

---

## Workflows

### Desktop GUI

```bash
python main.py
```

Five tabs. The right panel shows an FOV preview (intensity image + summed decay) for all fitting tabs, switching to the interactive phasor view when the Phasor tab is active.

#### Single PTU analysis

Open the GUI and select **Single FOV Fit**.

Go to **Fit settings** and fill in:

- **PTU path**: your `.ptu` file. To get this from Leica LAS X, open the lif/lof, go to the FLIM window, and export raw data.
- **IRF method**: Machine IRF is recommended if you've built one. IRF XLSX works if you have a LAS X export for that specific PTU (right-click the summed/tail decay in the FLIM window → Export to Excel). Scatter PTU if you measured one directly. If none of those are available, use "Estimate from decay" and set FWHM to roughly 0.3–0.5 ns.
- **Number of exponentials**: 1, 2, or 3. Beyond 3 the math gets shaky and the biology harder to interpret, so that's the cap. 
- **Lifetime bounds**: 0.145–45 ns by default. Adjust if you're working with unusually short or long lifetimes.
- **Fitting mode**: Full runs both summed and per-pixel fitting and is needed to generate the FLIM image in the UI. If you just want global lifetime values for the whole FOV, use FAST. Per-pixel fitting is slow, especially with more exponentials.
- **Output prefix**: defaults to the PTU filename in the PTU directory. Change it to keep outputs organised.

**Masking and thresholding:**

- Cell mask uses Otsu thresholding to isolate cell regions from background. Whether it works well depends on how much contrast your sample has (success is hit-or-miss, try if you like gambling).
- Intensity threshold cuts out low-signal pixels from per-pixel fitting. Speeds things up and cleans up the map, but don't set it too high or you'll lose dim-but-real regions.

Expert fit settings (optimiser type, cost function, DE parameters) are under the **Expert fit settings** tab and are shared between single FOV and tile stitching pipelines.

#### Tile stitching and fitting

Open **Tile Stitch / Fit**.

You need the XLIF metadata file and the directory with the PTU tiles. The XLIF is in the Metadata folder of your LAS X project and contains stage coordinates and tile layout. If the directory has tiles from multiple scans mixed together, it should sort them out as long as the naming is consistent.

**Pipeline modes:**

- **Stitch only** builds a stitched intensity image and FLIM histogram cube, skips fitting. Exports the FLIM cube as an `.npy` file. Good for a quick visual check or if you want to hand the data off to something else.
- **Stitch then fit full ROI** stitches everything into a single mosaic and fits it. Not recommended unless you have a capable machine and a small tile count, fitting a full mosaic requires a lot of RAM (tens of GBs) and is slow.
- **Per-tile fit** the recommended option for most cases. Builds a global decay from all tiles, runs an initial fit to get the lifetime components, then fits each tile separately using those fixed lifetimes. Results are stitched back together at the end. Same quality as fitting the mosaic directly, but far more memory-efficient.

Fitting parameters are the same as for single FOV. For tile work, the machine IRF is strongly recommended, per-tile XLSX IRFs from LAS X can vary across tiles and cause inconsistencies.

**Per-pixel exports** (when enabled):

- Intensity image (16-bit TIFF) total photon count per pixel
- α-weighted lifetime images (32-bit TIFF) amplitude or intensity-weighted τ maps with configurable display range
- Individual component τ maps (32-bit TIFF) spatial distribution per lifetime component

**Tile registration:**

The stage drifts during acquisition, so tiles need registration before stitching. Phase correlation is used in three passes:

1. Column Y drift correction systematic Y drift across tile columns
2. Row Y residual correction remaining per-row Y misalignment
3. Row X backlash correction X misalignment from stage direction changes

You can set a maximum expected drift to help the algorithm. After registration, tiles are assembled using nearest-centre ownership: each output pixel is assigned to the tile whose centre is closest. Simple and fast, though sometimes not perfect at tile boundaries. Do check the stitched intensity image for any glaring misalignments.

#### Batch ROI fitting

Same settings as tile stitching, but no ROI analysis afterwards. Point it at a folder of XLIFs and it processes each one in sequence, outputting a CSV summary plus a folder per ROI with intensity images, decay arrays, and any lifetime exports you've selected.

#### ROI analysis

Available for single FOVs and single stitched ROIs. Place ROIs on the displayed FLIM image (rectangle, ellipse, polygon, or freehand) and get mean lifetime statistics per region.

ROIs are saved as part of the session. You can export them as GeoJSON for QuPath, the geometry and statistics travel with them. You can also import GeoJSON ROIs back in (from QuPath or anything else that writes GeoJSON), which makes it easy to define ROIs in one tool and analyse lifetimes in another. CSV export is available for spreadsheet-friendly output.

Note: GeoJSON import currently only preserves the outer boundary of shapes, donut-shaped ROIs with holes will lose the hole geometry on import.

##### Per-ROI decay fitting

The **⚗ Fit ROI Decay** button fits the summed decay from the selected region(s) independently, rather than using the whole-FOV fit.

**Selecting regions:**
- Single region: click it in the list, then click **⚗ Fit ROI Decay**.
- Multiple regions: Shift-click or Cmd-click to select several; they are combined into one union mask and treated as a single merged region for the fit.

**Fit Options dialog:** Before the fit runs, a small dialog appears pre-filled with the current global fit parameters. You can change any of the following without touching the main form:

| Option | Description |
|---|---|
| Components (n_exp) | 1-, 2-, or 3-exponential model |
| τ_min / τ_max (ns) | Lifetime search bounds |
| Cost function | Poisson deviance or Pearson χ² |

Click **Run Fit** to proceed or **Cancel** to abort.

**Results window:** Opens automatically after fitting. Shows:
- Decay data (log-scale), IRF overlay, and fitted model curve
- Weighted residuals panel with χ²_r annotation
- Summary table: τ₁…τₙ, amplitudes A₁…Aₙ, amplitude-weighted τ_mean, χ²_r (tail)

**Reopening results without refitting:** Select the same region(s) and click **View Fit**. The last result for that selection is cached in memory and the plot window reopens instantly. The cache is lost when the app is closed.

**Session persistence:** After fitting, the numeric stats (τ_mean_fit, τ₁…τₙ, A₁…Aₙ, χ²_r) are written back into each region's statistics and saved to the `.roi_session.npz` automatically. The plot data (decay array, model curve) is not stored, so after reloading a session you need to refit to regenerate the plot — but the τ values are already there.

**CSV export** ("Export as CSV" button) includes all fit result columns: `Tau_mean_fit_ns`, `Chi2_r_fit`, and dynamic `Tau1_fit_ns / Amp1_fit`, `Tau2_fit_ns / Amp2_fit`, … columns sized to the maximum number of exponential components across all fitted regions. Regions that haven't been fitted yet show `N/A`.

#### Phasor analysis

Load a PTU file plus an IRF calibration (XLSX, machine IRF, etc). The app computes the phasor histogram and shows it alongside the intensity image. Click on the phasor plot to place elliptical cursors, the corresponding pixels in the intensity image highlight immediately.

Per-cursor stats (phase lifetime τ_φ, pixel count, 5th–95th percentile range) print to the progress log. With two or more cursors, a two-component decomposition line is drawn between them and the component lifetimes and mean fractions are reported.

Sessions save to `.npz` allowing you to come back later and pick up where you left off.

**Phasor panel controls:**

- **Clear all / Undo**: remove everything or step back one cursor at a time
- **Save session**: writes phasor arrays + cursor state to `.npz`
- **Radius / Minor:major sliders**: resize the cursor in real time; stats update immediately

**Spatial filtering:**

A filter row sits above the cursor controls. Select a method, set parameters, and click **Apply**. Reset restores the original unfiltered data.

| Method | Parameter | Description |
|---|---|---|
| `gaussian` | σ (0.5–10 px) | Gaussian smoothing via phasorpy's NaN-aware implementation |
| `median` | size (3–15 px, odd) | Median filter; removes outlier pixels while preserving edges |
| `wavelet` | none | Wavelet soft-thresholding (Daubechies db4, MAD noise estimator) |

Filtering is applied in phasor space (G and S coordinates) after calibration. The phasor plot and cursor stats update immediately after applying.

---

### Guided Terminal UI (`main.py`)

```bash
python main.py --cli
```

| Option | Description |
|---|---|
| FLIM FIT a single FOV | Loads a PTU, builds an IRF, runs summed and/or per-pixel fitting |
| Phasor analysis | Opens the interactive phasor cursor tool |
| Reconstruct a FOV and FLIM FIT | Stitches multi-tile PTU data from XLIF metadata then fits the mosaic |
| Just stitch multiple tiles together | Tile stitching only, intensity images and FLIM histogram cubes |
| About | Version info and roadmap |

---

### Machine IRF Setup (Required)

The machine IRF is a calibrated instrument response built from your specific microscope configuration. Build it once per system/session setup and reuse it. This is the most reliable IRF method and the one I'd recommend over any of the XLSX-based alternatives. It stores the full shape instead of recreating one from a few datapoints. It isnt perfect and an actual recorded IRF will always be ideal, but it's a big step up from trying to estimate the IRF or relying on the sometimes spotty XLSX IRF exports.

You need matched `.ptu` + `.xlsx` pairs. The more the better, but 10–20 is generally sufficient.

| Goal | Minimum pairs |
|---|---|
| Peak-placement only | 4–6 |
| Stable IRF shape + placement | 10–12 |
| Robust production use | 15–20 |

Don't go below 10 unless your data is very homogeneous.

#### GUI method

1. Open the GUI, go to **Machine IRF Builder**
2. Select your pairs folder
3. Keep anchor as `peak` and reducer as `median` unless you have a reason to change them
4. Build and save as `machine_irf_default`

**Save locations:**

| Context | Location |
|---|---|
| Running from source | `flimkit/machine_irf/` |
| Compiled app (macOS/Linux) | `~/.flimkit/machine_irf/` |
| Compiled app (Windows) | `C:\Users\<name>\.flimkit\machine_irf\` |

After saving, restart the app so it picks up the new default.

#### Python API

```python
from flimkit.FLIM.irf_tools import build_machine_irf_from_folder

build_machine_irf_from_folder(
    folder="/path/to/pairs",
    align_anchor="peak",
    reducer="median",
    save=True,
    output_name="machine_irf_default",
)
```

---

### FLIM Reconvolution Fitting (CLI)

```bash
python fit_cli.py [OPTIONS]
```

#### Required

| Argument | Description |
|---|---|
| `--ptu PATH` | Path to the PTU file |

#### IRF Arguments

| Argument | Description |
|---|---|
| `--machine-irf PATH` | Pre-built machine IRF `.npy` file (recommended) |
| `--irf PATH` | Scatter PTU for a directly measured IRF |
| `--irf-xlsx PATH` | LAS X Excel export for analytical IRF fitting |
| `--xlsx PATH` | LAS X export XLSX for comparison |
| `--no-xlsx-irf` | Use XLSX for comparison only; don't use its IRF |
| `--estimate-irf {raw,parametric,none}` | Estimate IRF from decay rising edge (default: `none`) |
| `--irf-fwhm FLOAT` | IRF FWHM in ns |
| `--irf-bins INT` | Number of bins for the IRF (default: 21) |
| `--irf-fit-width FLOAT` | Region around time zero for IRF fitting in ns (default: 1.5) |

**IRF priority order (highest to lowest):**
1. `--machine-irf`
2. `--irf` (scatter PTU)
3. `--irf-xlsx`
4. `--xlsx` IRF columns (unless `--no-xlsx-irf`)
5. `--estimate-irf raw` / `parametric`
6. Gaussian fallback from FWHM

#### Fitting Arguments

| Argument | Description |
|---|---|
| `--nexp {1,2,3}` | Number of exponential components (default: 3) |
| `--tau-min FLOAT` | Minimum lifetime bound in ns (default: 0.145) |
| `--tau-max FLOAT` | Maximum lifetime bound in ns (default: 45.0) |
| `--mode {summed,perPixel,both}` | Fitting mode (default: `both`) |
| `--binning INT` | Spatial binning for per-pixel fitting (default: 1) |
| `--min-photons INT` | Minimum photons per pixel (default: 10) |
| `--optimizer {lm_multistart,de}` | Optimiser for summed fit (default: `de`) |
| `--restarts INT` | LM multi-start restarts (default: 8) |
| `--de-population INT` | DE population size (default: 30) |
| `--de-maxiter INT` | DE maximum iterations (default: 5000) |
| `--workers INT` | CPU cores for DE (-1 = all; auto-limited to 1 in compiled app) |
| `--no-polish` | Skip LM polish step after DE |
| `--cost-function {poisson,chi2}` | Cost function (default: `poisson`) |
| `--intensity-threshold INT` | Minimum photons per pixel mask |
| `--tau-display-min FLOAT` | Min lifetime for exported tau images (ns) |
| `--tau-display-max FLOAT` | Max lifetime for exported tau images (ns) |

#### Output Arguments

| Argument | Description |
|---|---|
| `--out NAME` | Output file prefix (default: `flim_out`, anchored to PTU directory) |
| `--no-plots` | Suppress plot generation |
| `--channel INT` | Detection channel (default: auto-detect) |

---

### Phasor Analysis (CLI)

```bash
python phasor_cli.py [OPTIONS]
```

| Argument | Description |
|---|---|
| `--ptu PATH` | Path to a `.ptu` file |
| `--irf PATH` | IRF calibration Excel file (XLSX) |
| `--machine-irf PATH` | Machine IRF `.npy` file |
| `--session PATH` | Resume a saved `.npz` session |

With no arguments, the CLI drops into a guided `inquirer` flow.

---

### Python API

#### Phasor Analysis

```python
from flimkit.phasor_launcher import launch_phasor, save_session, load_session

# Interactive prompts
state = launch_phasor()

# Pass paths directly
state = launch_phasor('data.ptu', irf_path='irf.xlsx')

# With spatial phasor filtering
state = launch_phasor('data.ptu', irf_path='irf.xlsx',
                      phasor_filter='gaussian', filter_kwargs={'sigma': 1.5})
state = launch_phasor('data.ptu', irf_path='irf.xlsx',
                      phasor_filter='median',   filter_kwargs={'size': 5})
state = launch_phasor('data.ptu', irf_path='irf.xlsx',
                      phasor_filter='wavelet')

# Resume a saved session
state = launch_phasor(session_path='session.npz')

# Save/load programmatically
save_session('session.npz',
             real_cal=state['real_cal'], imag_cal=state['imag_cal'],
             mean=state['mean'], frequency=state['frequency'],
             cursors=state['cursors'], params=state['params'])

sess = load_session('session.npz')
```

#### PTU File Reading

```python
from flimkit.PTU.reader import PTUFile

ptu = PTUFile('data.ptu', verbose=True)
decay = ptu.summed_decay(channel=None)           # auto-detect channel
stack = ptu.pixel_stack(channel=None, binning=1) # (Y, X, H)
print(ptu.n_bins, ptu.tcspc_res, ptu.time_ns)
```

#### Signal Extraction (xarray)

```python
from flimkit.PTU.tools import signal_from_PTUFile
import numpy as np

signal = signal_from_PTUFile('data.ptu', dtype=np.uint32, binning=4)
# signal.attrs['frequency'] — modulation frequency in MHz
```

#### Phasor Computation

```python
from flimkit.phasor.signal import (
    return_phasor_from_PTUFile,
    get_phasor_irf,
    calibrate_signal_with_irf,
    calibrate_signal_with_machine_irf,
)

mean, real, imag = return_phasor_from_PTUFile('data.ptu')

# Calibrate with XLSX IRF
irf_time_ns, irf_counts = get_phasor_irf('irf.xlsx')
real_cal, imag_cal = calibrate_signal_with_irf(
    signal, real, imag, irf_time_ns, irf_counts, frequency)

# Calibrate with machine IRF
real_cal, imag_cal = calibrate_signal_with_machine_irf(
    signal, real, imag, 'machine_irf_default.npy', frequency)
```

#### Tile Stitching

```python
from flimkit.PTU.stitch import stitch_flim_tiles, load_flim_for_fitting
from pathlib import Path

result = stitch_flim_tiles(
    xlif_path=Path('metadata/R 2.xlif'),
    ptu_dir=Path('PTU_tiles/'),
    output_dir=Path('stitched/R_2/'),
    ptu_basename='R 2',
    rotate_tiles=True,
)

stack, tcspc_res, n_bins = load_flim_for_fitting(
    Path('stitched/R_2/'), load_to_memory=True)
decay = stack.sum(axis=(0, 1))
```

#### Intensity Images & Cell Masking

```python
from flimkit.image.tools import (
    make_intensity_image, make_cell_mask,
    apply_intensity_threshold, pick_intensity_threshold,
)

intensity = make_intensity_image('data.ptu', rotate_90_cw=True)
mask      = make_cell_mask(intensity, save_mask=True, path='output/')
int_mask  = apply_intensity_threshold(intensity, threshold=50)
threshold = pick_intensity_threshold(intensity)  # interactive slider
```

---

## Configuration Reference

All defaults live in `flimkit/configs.py` and can be overridden via CLI args or the GUI.

### Fitting Defaults

| Parameter | Default | Description |
|---|---|---|
| `Tau_min` | 0.145 ns | Lower lifetime bound |
| `Tau_max` | 45.0 ns | Upper lifetime bound |
| `n_exp` | 3 | Number of exponential components |
| `D_mode` | `'both'` | Fitting mode: `'summed'`, `'perPixel'`, or `'both'` |
| `binning_factor` | 1 | Spatial binning for per-pixel fitting |
| `Optimizer` | `'de'` | `'de'` (Differential Evolution) or `'lm_multistart'` |
| `MIN_PHOTONS_PERPIX` | 10 | Minimum photons for per-pixel fitting |
| `OUT_NAME` | `'flim_out'` | Default output prefix |

### Phasor Filtering Defaults

| Parameter | Default | Description |
|---|---|---|
| `PHASOR_FILTER` | `None` | Filter method: `'gaussian'`, `'median'`, `'wavelet'`, or `None` |
| `PHASOR_FILTER_SIGMA` | 1.0 | Gaussian σ in pixels |
| `PHASOR_FILTER_SIZE` | 3 | Median filter kernel size (pixels) |
| `PHASOR_FILTER_WAVELET` | `'db4'` | Wavelet family for wavelet denoising |
| `PHASOR_FILTER_LEVEL` | 1 | Wavelet decomposition level |

### Optimiser Settings

| Parameter | Default | Description |
|---|---|---|
| `lm_restarts` | 8 | Levenberg–Marquardt multi-start restarts |
| `de_population` | 30 | DE population size |
| `de_maxiter` | 5000 | DE maximum iterations |
| `n_workers` | -1 (source) / 1 (compiled) | CPU cores for DE; capped at 1 in the compiled app to avoid multiprocessing issues |

### Display Range Settings

Pixel values outside the range are clamped to the boundary, not zeroed.

| Parameter | Default | Description |
|---|---|---|
| `TAU_DISPLAY_MIN` | `None` | Min lifetime (ns) for tau images |
| `TAU_DISPLAY_MAX` | `None` | Max lifetime (ns) for tau images |
| `INTENSITY_DISPLAY_MIN` | `None` | Min photon count for intensity images |
| `INTENSITY_DISPLAY_MAX` | `None` | Max photon count for intensity images |

### Machine IRF Settings

| Parameter | Default | Description |
|---|---|---|
| `MACHINE_IRF_DIR` | `flimkit/machine_irf` (source) / `~/.flimkit/machine_irf` (compiled) | Storage directory |
| `MACHINE_IRF_DEFAULT_PATH` | User copy if present, else bundled default | Resolved at startup |
| `MACHINE_IRF_ALIGN_ANCHOR` | `'peak'` | Alignment landmark during IRF construction |
| `MACHINE_IRF_REDUCER` | `'median'` | Aggregation method across paired IRFs |
| `MACHINE_IRF_FIT_STRATEGY` | `'fixed'` | Runtime fitting strategy |
| `MACHINE_IRF_FIT_BG` | `True` | Fit background offset |
| `MACHINE_IRF_FIT_SIGMA` | `False` | Fit Gaussian broadening |
| `MACHINE_IRF_FIT_TAIL` | `False` | Fit exponential tail |

### Cost Functions

| Function | Description |
|---|---|
| `poisson` | Poisson deviance (C-statistic). Recommended |
| `chi2` | Neyman chi-squared (legacy)  |

---

## Module Reference

### `flimkit.PTU` — PTU File I/O

#### `reader.py`
- **`PTUFile(path, verbose=False)`** — parse a PicoQuant PTU file. Extracts TCSPC metadata and T3 photon records.
  - `.summed_decay(channel=None)` — summed decay histogram
  - `.pixel_stack(channel=None, binning=1)` — (Y, X, H) histogram stack
  - `.raw_pixel_stack(channel=None, binning=1)` — overflow-corrected pixel stack using nsync timing
  - `.n_bins`, `.tcspc_res`, `.time_ns` — TCSPC metadata

#### `decode.py`
- Low-level T3 record decoding (PicoHarp, HydraHarp formats)
- `create_time_axis()` — build time axis from PTU metadata

#### `tools.py`
- **`signal_from_PTUFile(path, dtype, binning)`** — load PTU and return an `xarray.DataArray` with labelled dimensions (`Y`, `X`, `H`) and `frequency` attribute

#### `stitch.py`
- **`stitch_flim_tiles(xlif_path, ptu_dir, output_dir, ...)`** — stitch multi-tile PTU data into a mosaic using XLIF metadata. Three-pass phase-correlation registration (Preibisch et al. 2009): column Y drift, row Y residuals, row X backlash. Nearest-centre ownership for canvas assembly.
- **`fit_flim_tiles(...)`** — full fitting pipeline on a stitched mosaic (two-pass: pooled DE fit → per-pixel NNLS)
- **`load_flim_for_fitting(output_dir, load_to_memory)`** — load previously stitched data

---

### `flimkit.FLIM` — Reconvolution Fitting

#### `fitters.py`
- **`fit_summed(decay, tcspc_res, n_bins, irf_prompt, ...)`** — fit a summed FLIM decay via reconvolution. Pass 1: Differential Evolution global search → Levenberg–Marquardt polish. Returns `(best_params, summary_dict)`.
- **`fit_per_pixel(stack, tcspc_res, n_bins, irf_prompt, global_popt, n_exp, ...)`** — per-pixel fitting with τ values fixed from the global fit. Uses NNLS — fast, convex, unique solution. Pass 2.

**Two-pass model:**

```
y(t) = [IRF(t + Shift_IRF) + Bkgr_IRF] ⊗ [Σ αᵢ·exp(−t/τᵢ) + Bkgr]
```

Pass 1 (summed): DE → LM polish → fixes τ₁…τₙ  
Pass 2 (per-pixel): NNLS fits α₁…αₙ and background with fixed τ values

Primary per-pixel output: `tau_mean_amp` = Σ(fracᵢ × τᵢ) — amplitude-weighted mean lifetime

#### `assemble.py`
- **`assemble_tile_maps(tile_results, canvas_h, canvas_w, n_exp)`** — assemble per-tile results into a single canvas
- **`derive_global_tau(canvas, n_exp)`** — ROI-level lifetime statistics from the assembled canvas
- **`save_assembled_maps(canvas, global_summary, output_dir, roi_name, n_exp, ...)`** — save canvas as TIFFs and NPY

#### `irf_tools.py`
- **`build_machine_irf_from_folder(folder, align_anchor, reducer, ...)`** — build machine IRF from paired PTU/XLSX files. Aligns to decay peak, aggregates by median, saves as `.npy` + `_meta.json`.
- **`irf_from_xlsx_analytical(xlsx, ...)`** — fit the Leica analytical IRF model (Gaussian + exponential tail)
- **`gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, peak_bin)`** — generate Gaussian IRF from FWHM

---

### `flimkit.phasor` — Phasor Analysis

#### `signal.py`
- **`return_phasor_from_PTUFile(ptu_file)`** — compute phasor coordinates from a PTU file
- **`get_phasor_irf(irf_xlsx)`** — read IRF from LAS X Excel export
- **`calibrate_signal_with_irf(signal, real, imag, irf_time_ns, irf_counts, frequency)`** — phase/modulation correction via IRF phasor
- **`calibrate_signal_with_machine_irf(signal, real, imag, machine_irf_npy, frequency)`** — calibrate using a machine IRF `.npy`. Reads companion `_meta.json` for time resolution; interpolates onto the signal time axis.

#### `filters.py`
- **`phasor_filter(real, imag, method, *, mean=None, sigma=1.0, size=3, wavelet='db4', level=1, threshold_mode='soft')`** — apply a spatial filter to calibrated phasor G/S arrays. When `mean` is supplied, the phasorpy 0.10 NaN-aware C implementation is used for Gaussian and median; otherwise falls back to scipy. Wavelet denoising uses PyWavelets with a MAD noise estimator. Returns `(real_f, imag_f)`.

#### `interactive.py`
- **`phasor_cursor_tool(real_cal, imag_cal, mean, frequency, ...)`** — interactive phasor cursor widget. Works in Jupyter (ipywidgets) and standalone scripts (matplotlib.widgets). Click-to-place elliptic cursors, adjustable radius/angle, per-cursor τ_φ maps, two-component decomposition, Undo/Peaks/Export/Save.

#### `peaks.py`
- **`find_phasor_peaks(real_cal, imag_cal, mean, frequency, ...)`** — automatic peak detection on 2-D phasor histograms via Gaussian smoothing and local maxima detection

---

### `flimkit.UI` — Desktop GUI

#### `gui.py`
- **`launch_gui()`** — entry point for the Tkinter GUI
- **`FLIMKitApp`** — main application class. Tabs: Single FOV Fit, Tile Stitch/Fit, Batch ROI Fit, Machine IRF Builder, Phasor Analysis
- **`FOVPreviewPanel`** — right-panel widget showing intensity image and summed decay. Switches to `PhasorViewPanel` when the Phasor tab is active. Caches the last fitted IRF prompt (`_irf_prompt`) so per-ROI fits can reuse it.

#### `roi_tools.py`
- **`RoiManager`** — stores region geometry and per-region statistics. Serialises to/from JSON for `.roi_session.npz` persistence.
  - `.add_region(name, tool, coords)` — register a new region, returns its integer ID
  - `.compute_region_mask(region_id, image_shape)` — boolean (H×W) mask for a region
  - `.to_json()` / `.from_json(json_str)` — serialise/deserialise for session files
- **`RoiAnalysisPanel`** — tab panel for region drawing, statistics display, and per-ROI fitting.
  - Drawing modes: Select, Rectangle, Ellipse, Polygon, Freehand
  - Per-region stats: τ_mean, τ_median, τ_stdev, photon count (all from the loaded lifetime/intensity maps)
  - **`_fit_roi_decay()`** — shows the Fit Options dialog, builds a union mask for all selected regions, extracts the summed decay, and runs `fit_summed` in a background thread. On completion writes τ stats back to each merged region and updates the session file.
  - **`_show_roi_fit_result(result)`** / **`_view_last_fit_result()`** — open (or reopen from cache) the dark-themed fit result popup with decay plot, residuals, and summary table.
  - Export: CSV (including fit columns), GeoJSON (single or all regions), GeoJSON import
- **`_ask_roi_fit_options(params)`** — modal dialog for overriding n_exp, τ bounds, and cost function before a per-ROI fit. Returns an updated params dict or None if cancelled.

#### `phasor_panel.py`
- **`PhasorViewPanel(parent, max_cursors=6)`** — embedded Tkinter widget with `FigureCanvasTkAgg`. Top axes: FOV intensity image (colourised once cursors are placed); bottom axes: phasor histogram. Controls: Clear, Undo, Save session, Radius slider, Minor/major slider, spatial filter row (method selector, σ/size spinboxes, Apply, Reset).
  - `.set_data(real_cal, imag_cal, mean, frequency, display_image, min_photons)` — load phasor data; call on main thread
  - `.load_session(session, min_photons)` — restore a saved `.npz` session
  - `.get_session_dict()` — export current state for saving

---

### `flimkit.image` — Image Utilities

#### `tools.py`
- **`make_intensity_image(ptu_path, rotate_90_cw, save_image)`** — 2-D intensity image from PTU
- **`make_cell_mask(intensity_image, ...)`** — binary cell mask via Otsu thresholding + morphological cleanup
- **`apply_intensity_threshold(intensity_image, threshold)`** — boolean mask for photon-count gating
- **`pick_intensity_threshold(intensity_image)`** — interactive slider for visual threshold selection

---

### `flimkit.utils` — Shared Utilities

#### `plotting.py`
- **`plot_summed(...)`** — main summed-fit figure: log-scale decay + model overlay, weighted residuals, parameter table
- **`plot_pixel_maps(...)`** — per-pixel lifetime and amplitude maps
- **`plot_lifetime_histogram(...)`** — lifetime distribution histogram

#### `enhanced_outputs.py`
- **`save_fit_summary_txt(...)`** — human-readable fit results text file
- **`save_weighted_tau_images(...)`** — intensity-weighted and amplitude-weighted τ TIFFs with optional display range clipping

#### `lifetime_image.py`
- **`make_lifetime_image(canvas, output_dir, roi_name, tau_min_ns, tau_max_ns, ...)`** — colourised lifetime image with NaN-aware smoothing and gamma correction

#### `xlsx_tools.py`
- **`load_xlsx(path, debug=False)`** — parse a LAS X FLIM export XLSX. Auto-detects column layout; returns `decay_t/c`, `irf_t/c`, `fit_t/c`, `res_t/c`.

#### `xml_utils.py`
- **`parse_xlif_tile_positions(xlif_path, ptu_basename)`** — tile positions from XLIF (microns)
- **`get_pixel_size_from_xlif(xlif_path)`** — pixel size (m) and pixel count
- **`compute_tile_pixel_positions(tiles, pixel_size_m, tile_size)`** — convert physical positions to pixel coordinates and compute canvas size

---

## Project Structure

```
├── main.py                        # Guided terminal UI
├── fit_cli.py                     # FLIM fitting CLI
├── phasor_cli.py                  # Phasor analysis CLI
├── build_and_sign.py              # PyInstaller build + codesign
├── validate_installation.py       # Installation sanity check (10 checks)
├── hardware_limits.py             # Hardware stress test — throughput & RAM headroom
├── requirements.txt
│
├── flimkit/
│   ├── configs.py                 # Default fitting parameters
│   ├── interactive.py             # Guided fitting launcher
│   ├── phasor_launcher.py         # Guided phasor launcher
│   ├── machine_irf/               # Machine IRF files — generated per system
│   │
│   ├── UI/
│   │   ├── gui.py                 # Tkinter desktop GUI
│   │   ├── roi_tools.py           # ROI drawing panel, RoiManager, per-ROI decay fitting
│   │   └── phasor_panel.py        # Embedded phasor view panel
│   │
│   ├── PTU/
│   │   ├── reader.py              # PTUFile — T3 record decoding
│   │   ├── decode.py              # Low-level histogram extraction
│   │   ├── tools.py               # signal_from_PTUFile (xarray)
│   │   └── stitch.py              # Multi-tile stitching + registration
│   │
│   ├── FLIM/
│   │   ├── models.py              # Decay models + DE cost functions
│   │   ├── fitters.py             # fit_summed / fit_per_pixel (NNLS)
│   │   ├── fit_tools.py           # IRF alignment, bin utilities
│   │   ├── assemble.py            # Tile map assembly + global tau stats
│   │   └── irf_tools.py           # IRF estimation + machine IRF builder
│   │
│   ├── phasor/
│   │   ├── signal.py              # Phasor computation & calibration
│   │   ├── interactive.py         # Interactive cursor tool
│   │   ├── peaks.py               # Automatic peak detection
│   │   └── filters.py             # Spatial phasor filtering (gaussian/median/wavelet)
│   │
│   ├── image/
│   │   └── tools.py               # Intensity images, cell masking
│   │
│   └── utils/
│       ├── plotting.py            # Decay + pixel map plots
│       ├── enhanced_outputs.py    # TIFF exports, summary text
│       ├── lifetime_image.py      # Colourised lifetime images
│       ├── xlsx_tools.py          # LAS X Excel parsing
│       ├── xml_utils.py           # XLIF tile-position parsing
│       ├── misc.py                # Logging helpers
│       └── fancy.py               # Terminal banners
│
└── flimkit_tests/
    ├── run_tests.py
    ├── mock_data.py
    ├── conftest.py
    ├── test_complete_pipeline.py
│       ├── test_roi_decay_fit.py
        └── tests/
        ├── test_decode.py
        ├── test_integration.py
        └── test_xml_utils.py
```

---

## Compiled App (macOS / Windows / Linux)

FLIMKit can be packaged as a standalone executable — no Python needed on the target machine.

### Build

```bash
python install.py --dev   # installs PyInstaller (and test requirements)
python build_and_sign.py
```

Output: `dist/FLIMKit.app` (macOS) or `dist/FLIMKit` / `dist/FLIMKit.exe` (Linux/Windows).

### GPU acceleration in the compiled app

The compiled app bundles whatever GPU libraries are present on the **build machine**. The GPU backend is not fetched or probed at runtime from an external install, it must be baked in at build time.

| Build machine | GPU bundled |
|---|---|
| Apple Silicon (M-series) Mac with `mlx` installed | MLX + PyTorch MPS |
| Intel Mac | CPU only (no Metal GPU) |
| Linux/Windows with CUDA PyTorch | CUDA |
| Linux/Windows with ROCm PyTorch | ROCm |

**If you need GPU acceleration in the compiled app, build it yourself on the machine (or OS/hardware type) where it will run.** A pre-built binary downloaded from Releases will only have GPU support if it was built on matching hardware.

`build_and_sign.py` detects and bundles GPU backends automatically, run it on the target hardware after running `python install.py` to install the right backend.

### macOS notes

The app is self-signed (ad-hoc). If you built it locally, Gatekeeper won't prompt because there's no quarantine flag. For distribution to other machines, you need a paid Apple Developer ID and notarization via `xcrun notarytool`.

Uses `--onedir` for a proper `.app` bundle and avoids the two-dock-icon issue you get with `--onefile`'s two-stage launcher.

### Output file location

All output files are saved to the same directory as the input PTU file. The working directory inside the bundle is read-only.

### Machine IRF

Machine IRFs are stored in `~/.flimkit/machine_irf/` (created automatically). The app ships with a bundled default until you build your own. After saving a new machine IRF, restart the app.

---

## Testing

```bash
python install.py --dev   # installs test requirements (and PyInstaller)

cd flimkit_tests
python run_tests.py              # all tests
python run_tests.py -c           # with coverage report
python run_tests.py integration  # integration tests only

# Individual modules
pytest tests/test_xml_utils.py -v
pytest tests/test_decode.py -v
pytest tests/test_integration.py -v
```

| Area | What's tested |
|---|---|
| XML/XLIF parsing | Tile positions, metadata extraction |
| PTU decoding | Histogram extraction, time axis |
| Tile stitching | Canvas computation, overlap handling |
| Integration | Complete workflows, error handling |
| Per-tile fit pipeline | Assembly, global tau, output files |
| Per-ROI decay fitting | ROI mask extraction, fit pipeline, IRF fallback, stat writeback, edge cases |

---

## Outputs & File Formats

| Format | Description |
|---|---|
| PNG | Intensity and lifetime map images for quick visualisation |
| OME-TIFF | Lossless, metadata-preserving export, opens correctly in Fiji/ImageJ |
| GeoJSON | ROI geometries and statistics, imports directly into QuPath |
| CSV | Fit summaries and per-ROI statistics |
| NPZ | Session files (fitting results, phasor arrays, cursor state) for session restoration |
| NPY | Raw FLIM histogram cubes and assembled lifetime maps |
| TXT | Human-readable fit summaries |

---

## Troubleshooting

**Gatekeeper blocks the compiled app on macOS**  
Right-click → Open on first launch. After that it should run normally.

**Machine IRF not found after saving**  
Restart the app — the default IRF path is resolved at startup and won't update mid-session.

**Per-pixel fitting is very slow**  
That's expected for large FOVs on CPU. Try increasing `--binning` to aggregate pixels before fitting, or switch to summed-only mode if you don't need spatial maps. If you have a supported GPU (Apple Silicon, NVIDIA, AMD) and ran `python install.py`, GPU acceleration is detected and used automatically, no extra flags needed. Note that `--free-tau-perpixel` with n_exp ≥ 2 uses batched Adam on GPU; it only falls back to CPU when no backend is detected.

**Tile stitching produces visible seams**  
Check that the max drift setting isn't too restrictive. If registration looks fine but seams persist, it's likely a sample contrast issue at tile boundaries rather than a registration failure.

**ROI holes are lost on GeoJSON import**  
Known limitation... Only the outer boundary is imported. Donut-shaped ROIs with holes lose the hole geometry on import.

**Phasor calibration looks off**  
Make sure the IRF file is from the same acquisition session. XLSX-based IRFs from LAS X can vary between sessions, which is why the machine IRF exists.

---

## References

If you use FLIMKit in published work, please also cite the relevant dependencies where appropriate:

**PhasorPy** — phasor computation, calibration, and cursor analysis:
> Gohlke, C. et al. PhasorPy. Zenodo. https://doi.org/10.5281/zenodo.13862586

**Tile stitching** — phase-correlation registration algorithm:
> Preibisch, S., Saalfeld, S. and Tomancak, P. (2009). Globally optimal stitching of tiled 3D microscopic image acquisitions. *Bioinformatics* 25(11), 1463–1465. https://doi.org/10.1093/bioinformatics/btp184

---

## Contact

Alex Hunt — alexander.hunt@ed.ac.uk
