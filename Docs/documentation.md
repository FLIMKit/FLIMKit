# FLIMKit Documentation

> **v0.9.17** - Python toolkit for Fluorescence Lifetime Imaging Microscopy

> **Warning:** Active development. Cross-validate results with other software before drawing conclusions.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Input Formats](#supported-input-formats)
3. [Requirements & Installation](#requirements--installation)
4. [Quick Start](#quick-start)
5. [Workflows](#workflows)
   - [Desktop GUI](#desktop-gui)
   - [Guided Terminal UI](#guided-terminal-ui-mainpy)
   - [Machine IRF Setup](#machine-irf-setup-required)
   - [FLIM Reconvolution Fitting (CLI)](#flim-reconvolution-fitting-cli)
   - [Timelapse and Z-stack Fitting](#timelapse-and-z-stack-fitting)
   - [Synthetic Data Generation (CLI)](#synthetic-data-generation-cli)
   - [Phasor Analysis (CLI)](#phasor-analysis-cli)
   - [Python API](#python-api)
6. [Configuration Reference](#configuration-reference)
7. [Module Reference](#module-reference)
8. [Project Structure](#project-structure)
9. [Compiled App](#compiled-app-macos--windows--linux)
10. [Plugins](#plugins)
11. [Testing](#testing)
12. [Outputs & File Formats](#outputs--file-formats)
13. [Troubleshooting](#troubleshooting)
14. [Contact](#contact)

---

## Overview

FLIMKit handles FLIM data from FLIM microscope systems and common TCSPC / time-tag formats (PicoQuant `.ptu`, Becker & Hickl `.sdt`, ISS time-tag, Photonscore `.photons`). It's designed as a replacement for FLIM microscope software, with two main workflows:

| Workflow | Description |
|---|---|
| **Reconvolution fitting** | Mono/bi/tri-exponential lifetime fitting with full IRF deconvolution, per-pixel and summed modes, multi-tile ROI stitching, and batch processing |
| **Lifetime distribution fitting** | Gaussian and Lorentzian continuous lifetime distributions (α(τ) models), per-ROI and per-pixel maps with GPU acceleration |
| **Phasor analysis** | Calibrated phasor plots with interactive elliptical cursors, automatic peak detection, two-component decomposition, and session save/load |

Both are accessible through a desktop GUI, guided terminal UI, CLI scripts, or the Python API.

Imaging files are reconstructed into a per-pixel decay cube `(Y, X, H)` from their scan / frame / line / pixel markers, and the intensity image is that cube summed over the time axis. Files without imaging markers (point, single-spot, FCS) are fit as a single decay with no image.

---

## Supported Input Formats

FLIMKit auto-detects the file type and routes everything through one loader (`FLIMFile` in `flimkit.formats`), so every workflow behaves the same regardless of instrument.

**Loads as** says what the file can drive: *Fitting + phasor* means it carries a time-resolved decay per pixel; *Phasor only* means the file already stores computed phasor coordinates, so there is no decay to fit; *Intensity only* means no lifetime data at all.

| Format | Extension | Loads as | Reader | Validated against real files |
|---|---|---|---|---|
| PicoQuant PTU | `.ptu` | Fitting + phasor | [`ptufile`](https://github.com/cgohlke/ptufile) | Yes (32 files) |
| Becker & Hickl SDT | `.sdt` | Fitting + phasor | [`sdtfile`](https://github.com/cgohlke/sdtfile) | Yes (bit-identical) |
| Photonscore LINCam | `.photons` | Fitting + phasor | [`photonsfile`](https://github.com/alex1075/photonsfile) | Yes (bit-exact vs SDK) |
| PicoQuant BIN | `.bin` | Fitting + phasor | [`ptufile`](https://github.com/cgohlke/ptufile) | Upstream |
| PicoQuant PHU | `.phu` | Fitting (no image) | [`ptufile`](https://github.com/cgohlke/ptufile) | Upstream |
| SimFCS B&H | `.b&h` | Fitting + phasor | [`lfdfiles`](https://github.com/cgohlke/lfdfiles) | Upstream (no time axis in file) |
| SimFCS BHZ | `.bhz` | Fitting + phasor | [`lfdfiles`](https://github.com/cgohlke/lfdfiles) | Upstream (no time axis in file) |
| ImSpector FLIM TIFF | `.tif`, `.tiff` (sniffed) | Fitting + phasor | [`tifffile`](https://github.com/cgohlke/tifffile) | Upstream |
| ISS Vista TDFLIM | `.iss-tdflim`, `.tdflim` | Fitting + phasor | [`lfdfiles`](https://github.com/cgohlke/lfdfiles) | Upstream |
| FLIM LABS imaging | `.json` (sniffed) | Fitting + phasor | [`phasorpy`](https://github.com/phasorpy/phasorpy) | Upstream |
| ISS time-tag | `.tagtime`, `.tagchannel`, `.tagdecay` | Fitting + phasor | FLIMKit (from ISS spec) | **No** |
| ISS FD-FLIM | `.ifli` | Phasor only | [`lfdfiles`](https://github.com/cgohlke/lfdfiles) | Upstream |
| SimFCS referenced | `.ref`, `.r64` | Phasor only | [`lfdfiles`](https://github.com/cgohlke/lfdfiles) | Upstream (no frequency in file) |
| PhasorPy OME-TIFF | `.ome.tif` (sniffed) | Phasor only | [`tifffile`](https://github.com/cgohlke/tifffile) | Upstream |
| FLIM LABS phasor | `.json` (sniffed) | Phasor only | [`phasorpy`](https://github.com/phasorpy/phasorpy) | Upstream |
| ISS intensity image | `.ifi` | Intensity only | FLIMKit (from ISS spec) | **No** |

"Upstream" means decoding is delegated to a maintained third-party reader that is tested against real files by its own author; FLIMKit has not independently re-validated it. "Sniffed" means the extension is ambiguous (an ordinary TIFF or JSON is not claimed), so the file is identified by its content rather than its name.

Formats whose files carry no time axis (`.b&h`, `.bhz`) or no modulation frequency (`.ref`, `.r64`) will prompt for the missing value, since fits and the universal circle cannot be computed without it.

### Provenance

- **PicoQuant `.ptu` (T3)** - PicoHarp, HydraHarp v1/v2, TimeHarp 260 N/P, MultiHarp / generic. The original FLIMKit decoder is kept as a cross-checked reference in `flim-native-decoders` (`flimkit/formats/PTU/NOTICE.md`).
- **Becker & Hickl `.sdt`** - SPCM histogram / image files (per-pixel decays already binned). FLIMKit's own decoder, written from B&H's SPCM docs and checked bit-for-bit against `sdtfile`, is kept as a reference in `flim-native-decoders` (`flimkit/formats/BH/NOTICE.md`).
- **Photonscore `.photons`** - LINCam D7 container (position-sensitive). `photonsfile` is a pure-Python reader spun out of FLIMKit, with no native dependency; `dt` calibration comes from the `TacChannel` attribute (`flimkit/formats/PS/NOTICE.md`).
- **ISS** - the `.TAGTIME`/`.TAGCHANNEL`/`.TAGDECAY` triplet is read together from any one of the three paths or their shared basename. Format specifications were provided by ISS (`flimkit/formats/ISS/NOTICE.md`).

> **The ISS time-tag and `.ifi` readers are experimental and need testing.** They were written from ISS's format specifications and have **not been validated against real ISS acquisitions** - byte order and the marker conventions are assumptions. Treat their results as unverified and cross-check them. If you have ISS data, trying it and reporting back is very welcome. The `.ifli` and `.tdflim` paths are delegated to `lfdfiles` and inherit that library's own testing.

Not decoded yet: T2-mode PTUs (`ptufile` reads the records, but FLIMKit does not build a decay cube from them), older PicoQuant `.pt3` / `.ht3`, Becker & Hickl raw `.spc` photon streams, and Leica `.lif`.

---

## Requirements & Installation

### System Requirements

- Python ≥ 3.12 (3.14 recommended, official builds use 3.14)
- macOS, Linux, or Windows

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array computation |
| `scipy` | Optimisers (Levenberg-Marquardt, Differential Evolution), signal processing |
| `matplotlib` | Plotting (decay curves, lifetime maps, phasor plots) |
| `xarray` | Labelled N-D arrays for FLIM signals |
| `phasorpy` (0.10) | Phasor computation, calibration, cursor masking, spatial filtering, lifetime conversion |
| `PyWavelets` | Wavelet-based phasor denoising |
| `ptufile` | PicoQuant `.ptu`, `.bin`, `.phu` decoding |
| `sdtfile` | Becker & Hickl `.sdt` decoding |
| `lfdfiles` | SimFCS `.b&h`, `.bhz`, `.ref`, `.r64` and ISS `.ifli`, `.iss-tdflim` decoding |
| `photonsfile` | Photonscore LINCam `.photons` (D7) decoding |
| `inquirer` | Interactive terminal prompts |
| `ipywidgets` + `ipympl` | Jupyter notebook interactive support |
| `cellpose` (≥ 3.0) | Deep-learning cell segmentation (Cellpose-SAM) for cell masking |
| `opencv-python` | Image I/O, resizing, and general image processing |
| `openpyxl` | Excel XLSX parsing for FLIM microscope software IRF extraction |
| `pandas` | Excel/XLSX IRF file parsing |
| `tifffile` | TIFF image I/O |
| `tqdm` | Progress bars |

### Installation

```bash
git clone https://github.com/FLIMKit/FLIMKit.git
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

- **PTU path**: your `.ptu` file. To get this from FLIM microscope software, open the lif/lof, go to the FLIM window, and export raw data.
- **IRF method**: Machine IRF is recommended if you've built one. IRF XLSX works if you have a FLIM microscope software export for that specific PTU (right-click the summed/tail decay in the FLIM window → Export to Excel). Scatter PTU if you measured one directly. If none of those are available, use "Estimate from decay" and set FWHM to roughly 0.3-0.5 ns.
- **Number of exponentials**: 1, 2, or 3. Beyond 3 the math gets shaky and the biology harder to interpret, so that's the cap. 
- **Lifetime bounds**: 0.145-45 ns by default. Adjust if you're working with unusually short or long lifetimes.
- **Fitting mode**: Full runs both summed and per-pixel fitting and is needed to generate the FLIM image in the UI. If you just want global lifetime values for the whole FOV, use FAST. Per-pixel fitting is slow, especially with more exponentials.
- **Output prefix**: defaults to the PTU filename in the PTU directory. Change it to keep outputs organised.

**Masking and thresholding:**

- Cell mask uses the Cellpose-SAM deep-learning segmentation model to isolate cell regions from background. It runs on GPU when available and falls back to CPU otherwise. The intensity image is percentile-normalised and resized to 224×224 before segmentation, then the label map is scaled back to the original resolution.
- Intensity threshold cuts out low-signal pixels from per-pixel fitting. Speeds things up and cleans up the map, but don't set it too high or you'll lose dim-but-real regions.

Expert fit settings (optimiser type, cost function, DE parameters) are under the **Expert fit settings** tab and are shared between single FOV and tile stitching pipelines.

#### Tile stitching and fitting

Open **Tile Stitch / Fit**.

You need the XLIF metadata file and the directory with the PTU tiles. The XLIF is in the Metadata folder of your FLIM microscope software project and contains stage coordinates and tile layout. If the directory has tiles from multiple scans mixed together, it should sort them out as long as the naming is consistent.

**Pipeline modes:**

- **Stitch only** builds a stitched intensity image and FLIM histogram cube, skips fitting. Exports the FLIM cube as an `.npy` file. Good for a quick visual check or if you want to hand the data off to something else.
- **Stitch then fit full ROI** stitches everything into a single mosaic and fits it. Not recommended unless you have a capable machine and a small tile count, fitting a full mosaic requires a lot of RAM (tens of GBs) and is slow.
- **Per-tile fit** the recommended option for most cases. Builds a global decay from all tiles, runs an initial fit to get the lifetime components, then fits each tile separately using those fixed lifetimes. Results are stitched back together at the end. Same quality as fitting the mosaic directly, but far more memory-efficient.
- **Multidimensional series** per-tile fit repeated over a z and/or time axis, writing one stitched plane per `(t, z)`. Works with or without an XLIF. See [Multidimensional Series](#multidimensional-series-stitched-tiles-over-z-and-time).

Fitting parameters are the same as for single FOV. For tile work, the machine IRF is strongly recommended, per-tile XLSX IRFs from FLIM microscope software can vary across tiles and cause inconsistencies.

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
- Summary table: τ₁...τₙ, amplitudes A₁...Aₙ, amplitude-weighted τ_mean, χ²_r (tail)

**Reopening results without refitting:** Select the same region(s) and click **View Fit**. The last result for that selection is cached in memory and the plot window reopens instantly. The cache is lost when the app is closed.

**Session persistence:** After fitting, the numeric stats (τ_mean_fit, τ₁...τₙ, A₁...Aₙ, χ²_r) are written back into each region's statistics and saved to the `.roi_session.npz` automatically. The plot data (decay array, model curve) is not stored, so after reloading a session you need to refit to regenerate the plot - but the τ values are already there.

**CSV export** ("Export as CSV" button) includes all fit result columns: `Tau_mean_fit_ns`, `Chi2_r_fit`, and dynamic `Tau1_fit_ns / Amp1_fit`, `Tau2_fit_ns / Amp2_fit`, ... columns sized to the maximum number of exponential components across all fitted regions. Regions that haven't been fitted yet show `N/A`.

#### Phasor analysis

Load a PTU file plus an IRF calibration (XLSX, machine IRF, etc). The app computes the phasor histogram and shows it alongside the intensity image. Click on the phasor plot to place elliptical cursors, the corresponding pixels in the intensity image highlight immediately.

Per-cursor stats (phase lifetime τ_φ, pixel count, 5th-95th percentile range) print to the progress log. With two or more cursors, a two-component decomposition line is drawn between them and the component lifetimes and mean fractions are reported.

Sessions save to `.npz` allowing you to come back later and pick up where you left off.

**Phasor panel controls:**

- **Clear all / Undo**: remove everything or step back one cursor at a time
- **Save session**: writes phasor arrays + cursor state to `.npz`
- **Radius / Minor:major sliders**: resize the cursor in real time; stats update immediately

**Spatial filtering:**

A filter row sits above the cursor controls. Select a method, set parameters, and click **Apply**. Reset restores the original unfiltered data.

| Method | Parameter | Description |
|---|---|---|
| `gaussian` | σ (0.5-10 px) | Gaussian smoothing via phasorpy's NaN-aware implementation |
| `median` | size (3-15 px, odd) | Median filter; removes outlier pixels while preserving edges |
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
| Timelapse batch fit | Fits a time series of PTUs as one FOV with a shared reference lifetime |
| Z-stack batch fit | Fits an axial stack of `region_zX.ptu` slices as one FOV |
| About | Version info and roadmap |

---

### Machine IRF Setup (Required)

The machine IRF is a calibrated instrument response built from your specific microscope configuration. Build it once per system/session setup and reuse it. This is the most reliable IRF method and the one I'd recommend over any of the XLSX-based alternatives. It stores the full shape instead of recreating one from a few datapoints. It isnt perfect and an actual recorded IRF will always be ideal, but it's a big step up from trying to estimate the IRF or relying on the sometimes spotty XLSX IRF exports.

You need matched `.ptu` + `.xlsx` pairs. The more the better, but 10-20 is generally sufficient.

| Goal | Minimum pairs |
|---|---|
| Peak-placement only | 4-6 |
| Stable IRF shape + placement | 10-12 |
| Robust production use | 15-20 |

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
| `--irf-xlsx PATH`, `--irf-export PATH` | LAS X `.xlsx` or delimited text export (`.csv`, `.tsv`, `.txt`, `.dat`, `.ascii`, `.asc`) for analytical IRF fitting |
| `--xlsx PATH`, `--analysis-export PATH` | LAS X `.xlsx` or delimited text export for comparison |
| `--no-xlsx-irf` | Use the analysis export for comparison only; don't use its IRF |
| `--estimate-irf {raw,parametric,machine_irf,machine_irf_sigma_full,machine_irf_sigma_half,none}` | Estimate IRF from decay rising edge, or reuse the machine IRF shape (default: `none`) |
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
| `--fit-start-ns FLOAT` | Fit window start in ns (default: auto from IRF onset) |
| `--fit-end-ns FLOAT` | Fit window end in ns (default: auto) |
| `--exclude-ns SPEC` | Bands to drop from the fit, e.g. `"7.2-8.8"` or `"7.2-8.8,11.0-11.5"`. See [Fit window and exclusion bands](#fit-window-and-exclusion-bands) |
| `--correct-pileup` | Apply Coates pile-up correction to the decay before fitting |
| `--free-tau` | Let lifetimes float per pixel instead of locking them to the summed fit |
| `--intensity-threshold INT` | Minimum photons per pixel mask |
| `--tau-display-min FLOAT` | Min lifetime for exported tau images (ns) |
| `--tau-display-max FLOAT` | Max lifetime for exported tau images (ns) |
| `--tvb-ptu PATH` | Reference PTU of a fluorophore-free background (buffer / culture-medium well). Its summed decay is fit as a scaled time-varying background `V·b(t)` instead of a flat offset (FLIMfit-style). |
| `--tvb-channel INT` | Detector channel for `--tvb-ptu` (default: same as `--channel`) |

#### Time-varying background correction

By default the fit treats the baseline as a flat constant `Z`. When a sample has structured background (autofluorescence from the medium, scatter, plate fluorescence), that background has its own decay shape, and a flat offset cannot remove it. Pass `--tvb-ptu` with a measurement of a fluorophore-free region and FLIMKit fits the full model `B = V·b(t) + Z`, where `b(t)` is the normalized measured background profile and `V` is a non-negative scale recovered per fit (and per pixel). The same option is available in the GUI (the "Time-varying background PTU" picker on the Single-FOV, Batch and Tile-Stitch panels) and the Python API (`tvb_profile=` / `fit_tvb=` on `fit_summed`, `fit_per_pixel`, and the distribution variants). The per-pixel scale is written out as a `*_tvb_scale.tif` map.

#### Output Arguments

| Argument | Description |
|---|---|
| `--out NAME` | Output file prefix (default: `flim_out`, anchored to PTU directory) |
| `--no-plots` | Suppress plot generation |
| `--channel INT` | Detection channel (default: auto-detect) |

---

### Timelapse and Z-stack Fitting

Both fit a stack of PTUs as a single field of view: all slices are pooled to fit one shared reference lifetime, then each slice is fitted per-pixel with τ locked, so lifetime is held constant across the stack while amplitude and intensity vary. Timelapse uses time as the stack axis, z-stack uses depth. They share the same code path (`flimkit/FLIM/batch.py`).

Reach them from the terminal UI (`python main.py --cli` → "Timelapse batch fit" / "Z-stack batch fit"), from the GUI (the Analysis toggle next to the input file in the Single FOV tab switches to Z-stack), or programmatically:

```python
from flimkit.interactive import timelapse_flim_fit, zstack_flim_fit

zstack_flim_fit()      # parses sys.argv
timelapse_flim_fit()
```

Expected filename patterns: `region_tX[_sY][_zZ].ptu` for timelapse, `region_zX.ptu` for a z-stack (as exported in Leica `.sptw` workspaces).

| Argument | Description |
|---|---|
| `--ptu-dir PATH` | Folder of stack PTUs (required) |
| `--output-dir PATH` | Base output directory (required) |
| `--ref-tau1 FLOAT` | Reference τ₁ in ns; skips the pooled global fit |
| `--ref-tau2 FLOAT` | Reference τ₂ in ns; required alongside `--ref-tau1` when `--nexp` ≥ 2 |
| `--fit-start-ns FLOAT` | Fit window start in ns (default: auto from IRF onset) |
| `--fit-end-ns FLOAT` | Fit window end in ns (default: auto) |
| `--exclude-ns SPEC` | Bands to drop from the fit, e.g. `"7.2-8.8"` or `"7.2-8.8,11.0-11.5"` |
| `--correct-pileup` | Coates pile-up correction |
| `--no-stack` | Skip saving the `(T,H,W)` / `(Z,H,W)` map stacks |
| `--bound-fraction` | Z-stack only: compute bound fraction α₂/(α₁+α₂) |

`--nexp`, `--tau-min`, `--tau-max`, `--machine-irf`, `--estimate-irf`, `--irf-fwhm`, `--irf-bins`, `--irf-fit-width`, `--optimizer`, `--restarts`, `--de-population`, `--de-maxiter`, `--workers`, `--no-polish`, `--channel`, `--min-photons`, `--cost-function` and `--no-plots` behave as in `fit_cli.py`.

Outputs land under `output-dir` in one folder per group: per-slice amplitude/intensity/lifetime maps, the pooled reference fit as `*_reference_fit.json`, stacked maps, and a series summary as `*_zseries.csv` / `.json` / `.png` (`*_timeseries.csv` for timelapse).

These treat each `_sY` position as an independent field of view. If the positions are overlapping tiles of one larger region, use the multidimensional series fit below instead, which stitches them.

#### Multidimensional Series (stitched tiles over z and time)

For tiled acquisitions with a z and/or time axis, where the tiles overlap and should be stitched into one canvas per plane rather than fitted as separate fields of view.

Stitch tab → pipeline "Multidimensional series".

Tile positions are found by maximising the correlation between neighbouring tiles over candidate shifts. FFT phase correlation is not used, because on real mosaics with ~10% overlap it returns no distinguishable peak.

Give it an XLIF or LIF if you have one and it is used as the starting layout, then refined against the overlap. The refinement is not optional: on the test set the metadata put both tiles at the same y while the images show a 23 px offset, which is the difference between an overlap correlation of 0.77 and 0.04. Metadata still helps for more than two tiles, where chaining pairwise shifts can drift. With no metadata the positions come from the overlap alone, which needs structure in the overlap to match on and is unreliable on sparse or thin samples.

One pooled decay is fitted across the whole series, and every plane is then fitted per-pixel with those τ values locked, so amplitudes stay comparable between timepoints. "Pool decay every N timepoints" subsamples that pooling step, which only needs photon statistics rather than every file.

Filenames follow the same `region_tX[_sY][_zZ].ptu` convention as the timelapse fit, and at least two positions are required.

```python
from flimkit.formats.PTU.stitch import fit_flim_series

manifest = fit_flim_series(ptu_dir, output_dir, args, pool_stride=10)
```

Outputs are one directory per `(t, z)` plane, each holding the usual fitted maps, plus a `*_series_index.json` manifest recording the recovered tile positions, the consensus τ values and every plane written. Tile positions can be supplied directly as `tile_positions=` to skip recovery.

Registration quality is reported as a correlation per tile pair. A low value means the overlap was not found, usually because the tiles genuinely do not overlap or the region is too sparse to register; supply positions from a `.lif` or `.xlif` in that case.

#### Fit window and exclusion bands

Available on `fit_cli.py`, timelapse and z-stack alike. By default the fit spans the decay from the IRF onset to the end of the record. Two things break that: signal before the rise (IRF artefacts) and reflection peaks partway down the tail, which pull a multi-exponential fit toward a spurious short component. `--fit-start-ns` and `--fit-end-ns` set the window explicitly; `--exclude-ns` drops one or more bands inside it, given as comma-separated `lo-hi` pairs in ns.

```bash
--fit-start-ns 0.5 --fit-end-ns 12.0 --exclude-ns "7.2-8.8"
```

Excluded bins are removed from the cost function rather than zeroed, so the fit statistic stays comparable.

On a synthetic 3.5 ns decay with a 5% reflection planted at 8.0 ns, excluding `7.5-8.5` moves the recovered lifetime from 3.5676 ns to 3.4872 ns and χ²_r from 8.33 to 0.73.

In the Python API the controls are `fit_start_ns=`, `fit_end_ns=` and `exclude_ns=` on `fit_summed`, where `exclude_ns` takes a list of `(lo, hi)` tuples; `flimkit.interactive.parse_exclude_ns` converts the string form. `fit_per_pixel` instead takes the resolved bin set as `fit_idx=`, which `fit_summed` reports back in its summary:

```python
popt, summary = fit_summed(..., exclude_ns=[(7.5, 8.5)])
maps = fit_per_pixel(..., fit_idx=summary['fit_idx'])
```

Windowed per-pixel fitting runs on the CPU path; the GPU backends have no window support yet.

---

### Synthetic Data Generation (CLI)

Generates FLIM data with a known ground truth: a sample PTU, a matching IRF PTU, and a JSON file recording the parameters used. Intended for cross-software validation, where the same file is fitted in FLIMKit and elsewhere and the recovered lifetimes are compared against truth.

```bash
python synth_cli.py --out ./validation --tau 3.0,0.8 --amps 0.7,0.3 --photons 1e5
```

| Argument | Description |
|---|---|
| `--out PATH` | Output directory for the PTUs and truth JSON (required) |
| `--name NAME` | Base name for the files (default: `synth`) |
| `--tau SPEC` | Lifetime(s) in ns, comma-separated for multi-exponential (default: `4.1`) |
| `--amps SPEC` | Amplitudes for multi-exponential τ, comma-separated (default: equal) |
| `--photons SPEC` | Summed photon count; comma-separated generates a series, e.g. `"2e4,1e5,5e5"` (default: `1e5`) |
| `--period-ns FLOAT` | Laser period in ns, sets the sync rate (default: 50.0) |
| `--res-ps FLOAT` | TCSPC bin width in ps (default: 25.0) |
| `--irf-fwhm-ns FLOAT` | IRF FWHM in ns (default: 0.15) |
| `--irf-center-ns FLOAT` | IRF peak position in ns (default: 2.0) |
| `--reflection-ns FLOAT` | Plant a reflection peak at this time in ns, e.g. `8.0` |
| `--reflection-frac FLOAT` | Reflection intensity as a fraction of total signal (default: 0.02) |
| `--reflection-width-ns FLOAT` | Reflection peak FWHM in ns (default: 0.15) |
| `--pileup-pp FLOAT` | Apply pile-up at this many photons per pulse, e.g. `0.1` |
| `--background-frac FLOAT` | Flat background as a fraction of total signal (default: 0.0) |
| `--image INT` | Image side length in pixels, square (default: 16) |
| `--no-irf` | Skip writing the IRF PTU |
| `--sdt` | Also write Becker & Hickl `.sdt` versions of sample and IRF |

`--reflection-ns` pairs with `--exclude-ns` on the fitting side: plant a reflection at a known position, then confirm that excluding that band recovers the input lifetime.

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
from flimkit.formats.PTU.reader import PTUFile

ptu = PTUFile('data.ptu', verbose=True)
decay = ptu.summed_decay(channel=None)           # auto-detect channel
stack = ptu.pixel_stack(channel=None, binning=1) # (Y, X, H)
print(ptu.n_bins, ptu.tcspc_res, ptu.time_ns)
```

#### Signal Extraction (xarray)

```python
from flimkit.formats.PTU.tools import signal_from_PTUFile
import numpy as np

signal = signal_from_PTUFile('data.ptu', dtype=np.uint32, binning=4)
# signal.attrs['frequency'] - modulation frequency in MHz
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
from flimkit.formats.PTU.stitch import stitch_flim_tiles, load_flim_for_fitting
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
| `lm_restarts` | 8 | Levenberg-Marquardt multi-start restarts |
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

### Fit Diagnostics

The cost function above selects the optimizer objective. The fields below are
post-fit diagnostics and do not change how the model is fitted.

The existing `reduced_chi2_pearson`, `reduced_chi2_tail_pearson`, and per-pixel
`chi2_r` fields are retained for compatibility with historical Leica LAS X
comparisons. Their one-count model floor means they are not generally expected
to equal one for sparse decays.

`calibrated_chi2_pearson`, `calibrated_chi2_tail_pearson`, and the per-pixel
`calibrated_chi2_r` map divide the same residual sum by its expected
contribution under a fixed Poisson model:

$$Q_{\mathrm{cal}} = \frac{\sum_i (Y_i-m_i)^2/\max(m_i,1)}{\sum_i \min(m_i,1)}.$$

The fixed-model expectation is one. Parameter fitting and data-selected windows
can introduce a smaller additional shift, so this diagnostic is not a classical
chi-square p-value.

---

## Module Reference

### `flimkit.formats` - Format Dispatch

The format layer sits behind one interface: every reader returns the same `(Y, X, H)` decay cube plus metadata, so the fitter, phasor, stitching and GUI never branch on file type.

#### `flim_file.py`
- **`FLIMFile(path, ...)`** - format-agnostic entry point. Sniffs the format and delegates to the matching reader, exposing the same `.summed_decay()` / `.pixel_stack()` / `.n_bins` / `.tcspc_res` interface regardless of source.
- **`detect_format(path)`** - identify the format from extension, magic bytes, and sibling files (ISS needs its triplet)
- **`file_modality(path)`** - whether the file is time-domain (fit) or frequency-domain (phasor only)
- **`supported_formats()`**, **`supported_extensions()`**, **`file_dialog_filetypes()`** - format registry, used to build the GUI file pickers

---

### `flimkit.formats.PTU` - PicoQuant PTU

#### `reader.py`
- **`PTUFile(path, verbose=False)`** - wraps Christoph Gohlke's `ptufile` and exposes the FLIMFile interface. Pins the TCSPC bin grid, integrates frames, and selects the photon channel.
  - `.summed_decay(channel=None)` - summed decay histogram
  - `.pixel_stack(channel=None, binning=1)` - (Y, X, H) histogram stack
  - `.raw_pixel_stack(channel=None, binning=1)` - (Y, X, H) stack (uint32)
  - `.n_bins`, `.tcspc_res`, `.time_ns` - TCSPC metadata
- **`read_pck(path)`** - reads PicoQuant Check / IRF `.pck` histograms (`ptufile` exposes only their tags, so this stays in FLIMKit)

#### `decode.py`
- `get_flim_histogram_from_ptufile()` - `(Y, X, H)` stack + metadata for the tile-stitch pipeline
- `create_time_axis()` - build time axis from PTU metadata

#### `tools.py`
- **`signal_from_PTUFile(path, dtype, binning)`** - load PTU and return an `xarray.DataArray` with labelled dimensions (`Y`, `X`, `H`) and `frequency` attribute

#### `stitch.py`
- **`stitch_flim_tiles(xlif_path, ptu_dir, output_dir, ...)`** - stitch multi-tile PTU data into a mosaic using XLIF metadata. Three-pass phase-correlation registration (Preibisch et al. 2009): column Y drift, row Y residuals, row X backlash. Nearest-centre ownership for canvas assembly.
- **`fit_flim_tiles(...)`** - full fitting pipeline on a stitched mosaic (two-pass: pooled DE fit → per-pixel NNLS)
- **`load_flim_for_fitting(output_dir, load_to_memory)`** - load previously stitched data

---

### `flimkit.formats.BH` - Becker & Hickl SDT

#### `reader.py`
- **`BHFile(path, ...)`** - wraps Christoph Gohlke's `sdtfile` and exposes the FLIMFile interface. Handles block layout, TAC range and repetition rate; auto-selects the populated channel.
- **`read_bh(path, binning=1, channel=None, sync_rate=None)`** - `(Y, X, H)` cube + metadata
- **`get_flim_data(path, ...)`**, **`get_intensity_image(path, ...)`** - cube and summed-intensity helpers
- **`create_time_axis(n_bins, tcspc_resolution)`** - time axis from SDT metadata

#### `writer.py`
- Writes `.sdt` files, used by `flimkit.synth` for the `--sdt` output so synthetic ground truth can be opened in SPCImage

### `flimkit.formats.PS` - Photonscore LINCam

#### `reader.py`
- **`PSFile(path, ...)`** - reads the `.photons` D7 container via `photonsfile`. Position-sensitive detector, so the image is formed by binning each photon's (x, y) and the decay by histogramming its micro-time.
- **`read_ps(path, binning=1, channel=None, pixels=512, n_bins=256, period_ns=None)`** - `(Y, X, H)` cube + metadata. `pixels` sets the spatial binning grid, `n_bins` the TCSPC histogram depth.
- **`get_flim_data(path, ...)`**, **`get_intensity_image(path, ...)`**, **`create_time_axis(...)`**

### `flimkit.formats.ISS` - ISS FastFLIM / Vista

> Experimental: written from ISS specifications and checked only against synthetic files, not real acquisitions (issue #19).

#### `reader.py`
- **`ISSFile(path, ...)`** - time-domain triplet (`.TAGTIME` / `.TAGCHANNEL` / `.TAGDECAY`); all three must sit alongside each other
- **`read_iss(path, binning=1, channel=None)`** - `(Y, X, H)` cube + metadata
- **`get_flim_data(path, ...)`**, **`get_intensity_image(path, ...)`**

#### `fdflim.py`
- **`ISSFdFlim(path)`** - frequency-domain `.ifli` (`VistaFLImage`). Already phasor data, so there is no decay to fit.
- **`phasor_from_ifli(path, channel=None, harmonic=0, calibrate=True)`** - per-pixel phase/modulation as phasor coordinates, applying the file's reference calibration

#### `image.py`
- **`read_ifi(path, channel=None)`**, **`get_intensity_image(path, channel=None)`** - ISS intensity images

---

### `flimkit.synth` - Synthetic Ground Truth

Generates FLIM data with known parameters, for validation. Driven by [`synth_cli.py`](#synthetic-data-generation-cli).

- **`generate(out_dir, name='synth', ny=16, nx=16, with_irf=True, sdt=False, **kwargs)`** - write a sample PTU, matching IRF PTU and truth JSON; `sdt=True` also writes `.sdt` versions
- **`generate_series(out_dir, photon_counts, ...)`** - a photon-count series from one parameter set, for testing count-dependent bias
- **`build_decay(tau_ns, amps=None, n_bins=2000, tcspc_res_ns=0.025, ...)`** - the noiseless expected decay, with optional reflection peak, pile-up and background
- **`sample_cube(expected, ny, nx, seed=0)`** - Poisson-sample the expected decay into a `(Y, X, H)` cube
- **`gaussian_irf(n_bins, center_bin, fwhm_bins)`** - synthetic IRF
- **`write_ptu(...)`**, **`write_sdt(...)`**, **`write_irf_ptu(...)`** - file writers

---

### `flimkit.FLIM` - Reconvolution Fitting

#### `fitters.py`
- **`fit_summed(decay, tcspc_res, n_bins, irf_prompt, ...)`** - fit a summed FLIM decay via reconvolution. Pass 1: Differential Evolution global search → Levenberg-Marquardt polish. Returns `(best_params, summary_dict)`.
- **`fit_per_pixel(stack, tcspc_res, n_bins, irf_prompt, global_popt, n_exp, ...)`** - per-pixel fitting with τ values fixed from the global fit. Uses NNLS - fast, convex, unique solution. Pass 2.

**Two-pass model:**

```
y(t) = [IRF(t + Shift_IRF) + Bkgr_IRF] ⊗ [Σ αᵢ·exp(−t/τᵢ) + Bkgr]
```

Pass 1 (summed): DE → LM polish → fixes τ₁...τₙ  
Pass 2 (per-pixel): NNLS fits α₁...αₙ and background with fixed τ values

Primary per-pixel output: `tau_mean_amp` = Σ(fracᵢ × τᵢ) - amplitude-weighted mean lifetime

#### `assemble.py`
- **`assemble_tile_maps(tile_results, canvas_h, canvas_w, n_exp)`** - assemble per-tile results into a single canvas
- **`derive_global_tau(canvas, n_exp)`** - ROI-level lifetime statistics from the assembled canvas
- **`save_assembled_maps(canvas, global_summary, output_dir, roi_name, n_exp, ...)`** - save canvas as TIFFs and NPY

#### `irf_tools.py`
- **`build_machine_irf_from_folder(folder, align_anchor, reducer, ...)`** - build machine IRF from paired PTU/XLSX files. Aligns to decay peak, aggregates by median, saves as `.npy` + `_meta.json`.
- **`irf_from_xlsx_analytical(xlsx, ...)`** - fit the analytical IRF model (Gaussian + exponential tail)
- **`gaussian_irf_from_fwhm(n_bins, tcspc_res, fwhm_ns, peak_bin)`** - generate Gaussian IRF from FWHM

---

### `flimkit.phasor` - Phasor Analysis

#### `signal.py`
- **`return_phasor_from_PTUFile(ptu_file)`** - compute phasor coordinates from a PTU file
- **`get_phasor_irf(irf_xlsx)`** - read IRF from FLIM microscope software Excel export
- **`calibrate_signal_with_irf(signal, real, imag, irf_time_ns, irf_counts, frequency)`** - phase/modulation correction via IRF phasor
- **`calibrate_signal_with_machine_irf(signal, real, imag, machine_irf_npy, frequency)`** - calibrate using a machine IRF `.npy`. Reads companion `_meta.json` for time resolution; interpolates onto the signal time axis.

#### `filters.py`
- **`phasor_filter(real, imag, method, *, mean=None, sigma=1.0, size=3, wavelet='db4', level=1, threshold_mode='soft')`** - apply a spatial filter to calibrated phasor G/S arrays. When `mean` is supplied, the phasorpy 0.10 NaN-aware C implementation is used for Gaussian and median; otherwise falls back to scipy. Wavelet denoising uses PyWavelets with a MAD noise estimator. Returns `(real_f, imag_f)`.

#### `interactive.py`
- **`phasor_cursor_tool(real_cal, imag_cal, mean, frequency, ...)`** - interactive phasor cursor widget. Works in Jupyter (ipywidgets) and standalone scripts (matplotlib.widgets). Click-to-place elliptic cursors, adjustable radius/angle, per-cursor τ_φ maps, two-component decomposition, Undo/Peaks/Export/Save.

#### `peaks.py`
- **`find_phasor_peaks(real_cal, imag_cal, mean, frequency, ...)`** - automatic peak detection on 2-D phasor histograms via Gaussian smoothing and local maxima detection

---

### `flimkit.UI` - Desktop GUI

#### `gui.py`
- **`launch_gui()`** - entry point for the Tkinter GUI
- **`FLIMKitApp`** - main application class. Tabs: Single FOV Fit, Tile Stitch/Fit, Batch ROI Fit, Machine IRF Builder, Phasor Analysis
- **`FOVPreviewPanel`** - right-panel widget showing intensity image and summed decay. Switches to `PhasorViewPanel` when the Phasor tab is active. Caches the last fitted IRF prompt (`_irf_prompt`) so per-ROI fits can reuse it.

#### `roi_tools.py`
- **`RoiManager`** - stores region geometry and per-region statistics. Serialises to/from JSON for `.roi_session.npz` persistence.
  - `.add_region(name, tool, coords)` - register a new region, returns its integer ID
  - `.compute_region_mask(region_id, image_shape)` - boolean (H×W) mask for a region
  - `.to_json()` / `.from_json(json_str)` - serialise/deserialise for session files
- **`RoiAnalysisPanel`** - tab panel for region drawing, statistics display, and per-ROI fitting.
  - Drawing modes: Select, Rectangle, Ellipse, Polygon, Freehand
  - Per-region stats: τ_mean, τ_median, τ_stdev, photon count (all from the loaded lifetime/intensity maps)
  - **`_fit_roi_decay()`** - shows the Fit Options dialog, builds a union mask for all selected regions, extracts the summed decay, and runs `fit_summed` in a background thread. On completion writes τ stats back to each merged region and updates the session file.
  - **`_show_roi_fit_result(result)`** / **`_view_last_fit_result()`** - open (or reopen from cache) the dark-themed fit result popup with decay plot, residuals, and summary table.
  - Export: CSV (including fit columns), GeoJSON (single or all regions), GeoJSON import
- **`_ask_roi_fit_options(params)`** - modal dialog for overriding n_exp, τ bounds, and cost function before a per-ROI fit. Returns an updated params dict or None if cancelled.

#### `phasor_panel.py`
- **`PhasorViewPanel(parent, max_cursors=6)`** - embedded Tkinter widget with `FigureCanvasTkAgg`. Top axes: FOV intensity image (colourised once cursors are placed); bottom axes: phasor histogram. Controls: Clear, Undo, Save session, Radius slider, Minor/major slider, spatial filter row (method selector, σ/size spinboxes, Apply, Reset).
  - `.set_data(real_cal, imag_cal, mean, frequency, display_image, min_photons)` - load phasor data; call on main thread
  - `.load_session(session, min_photons)` - restore a saved `.npz` session
  - `.get_session_dict()` - export current state for saving

---

### `flimkit.image` - Image Utilities

#### `tools.py`
- **`make_intensity_image(ptu_path, rotate_90_cw, save_image)`** - 2-D intensity image from PTU
- **`make_cell_mask(intensity_image, flow_threshold, cellprob_threshold, resize_to, gpu, ...)`** - binary cell mask via Cellpose-SAM segmentation (GPU when available, CPU fallback)
- **`apply_intensity_threshold(intensity_image, threshold)`** - boolean mask for photon-count gating
- **`pick_intensity_threshold(intensity_image)`** - interactive slider for visual threshold selection

---

### `flimkit.utils` - Shared Utilities

#### `plotting.py`
- **`plot_summed(...)`** - main summed-fit figure: log-scale decay + model overlay, weighted residuals, parameter table
- **`plot_pixel_maps(...)`** - per-pixel lifetime and amplitude maps
- **`plot_lifetime_histogram(...)`** - lifetime distribution histogram

#### `enhanced_outputs.py`
- **`save_fit_summary_txt(...)`** - human-readable fit results text file
- **`save_weighted_tau_images(...)`** - intensity-weighted and amplitude-weighted τ TIFFs with optional display range clipping

#### `lifetime_image.py`
- **`make_lifetime_image(canvas, output_dir, roi_name, tau_min_ns, tau_max_ns, ...)`** - colourised lifetime image with NaN-aware smoothing and gamma correction

#### `xlsx_tools.py`
- **`load_xlsx(path, debug=False)`** - parse a FLIM microscope FLIM export XLSX. Auto-detects column layout; returns `decay_t/c`, `irf_t/c`, `fit_t/c`, `res_t/c`.

#### `xml_utils.py`
- **`parse_xlif_tile_positions(xlif_path, ptu_basename)`** - tile positions from XLIF (microns)
- **`get_pixel_size_from_xlif(xlif_path)`** - pixel size (m) and pixel count
- **`compute_tile_pixel_positions(tiles, pixel_size_m, tile_size)`** - convert physical positions to pixel coordinates and compute canvas size

---

## Project Structure

```
├── main.py                        # Guided terminal UI
├── fit_cli.py                     # FLIM fitting CLI
├── phasor_cli.py                  # Phasor analysis CLI
├── synth_cli.py                   # Synthetic known-truth PTU/SDT generator
├── build_and_sign.py              # PyInstaller build + codesign
├── validate_installation.py       # Installation sanity check (10 checks)
├── hardware_limits.py             # Hardware stress test - throughput & RAM headroom
├── requirements.txt
│
├── flimkit/
│   ├── configs.py                 # Default fitting parameters
│   ├── interactive.py             # Guided fitting launcher
│   ├── phasor_launcher.py         # Guided phasor launcher
│   ├── machine_irf/               # Machine IRF files - generated per system
│   │
│   ├── UI/
│   │   ├── gui.py                 # Tkinter desktop GUI
│   │   ├── roi_tools.py           # ROI drawing panel, RoiManager, per-ROI decay fitting
│   │   └── phasor_panel.py        # Embedded phasor view panel
│   │
│   ├── synth.py                   # Synthetic known-truth data generation
│   │
│   ├── formats/
│   │   ├── flim_file.py           # FLIMFile, detect_format - format dispatch
│   │   ├── PTU/
│   │   │   ├── reader.py          # PTUFile (wraps ptufile), read_pck
│   │   │   ├── decode.py          # Histogram extraction for tile stitching
│   │   │   ├── tools.py           # signal_from_PTUFile (xarray)
│   │   │   └── stitch.py          # Multi-tile stitching + registration
│   │   ├── BH/
│   │   │   ├── reader.py          # BHFile (wraps sdtfile)
│   │   │   └── writer.py          # .sdt writer, used by synth
│   │   ├── PS/
│   │   │   └── reader.py          # PSFile (wraps photonsfile)
│   │   └── ISS/
│   │       ├── reader.py          # ISSFile - TD triplet (experimental)
│   │       ├── fdflim.py          # .ifli FD-FLIM phasor (experimental)
│   │       └── image.py           # .ifi intensity image (experimental)
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
│       ├── xlsx_tools.py          # FLIM microscope software Excel parsing
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

FLIMKit can be packaged as a standalone executable - no Python needed on the target machine.

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

## Plugins

Analysis tools reach the Tools menu through a registry, `flimkit.plugins`. FLIMKit's own tools are registered the same way an add-on is, so the file a contributor writes is the file a third party writes.

A plugin is a Python module that decorates a function:

```python
from flimkit.plugins import tool

FLIMKIT_PLUGIN_API = 1

@tool(id='hello_example', label='Hello Plugin...', menu='Tools', order=900)
def open_hello(app):
    from tkinter import messagebox
    messagebox.showinfo('Hello', 'This window came from an add-on, not from FLIMKit.')
```

`id` has to be unique across everything loaded. `menu` is a slash path, so `'Tools/Batch Processing'` nests one level down and any depth works. `order` sorts entries within a menu, low first, ties broken by label. The function is called with the GUI object, and `app.root` is the Tk parent to hang a window off. Keep the tkinter import inside the body so the module still imports on a headless machine.

Declare `FLIMKIT_PLUGIN_API` to match `flimkit.plugins.API_VERSION`. A mismatch is refused rather than half-loaded.

The registrations that ship with FLIMKit live in `flimkit/plugins/builtin/` and are listed in `BUILTIN`. A working example is in `examples/plugins/hello_tool.py`.

### Current images and ROIs

Plugins can exchange the images and regions currently shown by FLIMKit without reading private GUI fields:

```python
from flimkit.plugins import (
    export_rois_geojson,
    get_current_images,
    import_rois_geojson,
    tool,
)

@tool(id='bridge_example', label='Bridge Example...', menu='Tools', order=900)
def open_bridge(app):
    current = get_current_images(app)
    intensity = current['images'].get('intensity')
    lifetime = current['images'].get('lifetime')
    lifetime_unit = current['units'].get('lifetime')

    rois = export_rois_geojson(app)
    imported_ids = import_rois_geojson(app, rois, mode='append')
```

`get_current_images(app)` returns separate `images` and `units` dictionaries so metadata cannot collide with an image name. The available image names are `intensity` and `lifetime`; their units are `photons` and `ns`, respectively. Arrays are 2D copies. Intensity maps with trailing dimensions are reduced to 2D by summing those axes; lifetime maps must already be 2D. An image that has not been calculated is omitted from both dictionaries. Changing a returned array does not change the image held by FLIMKit.

This binding is limited to fitted lifetime and photon-count intensity images. Raw per-pixel decay histograms are not included. They require a separate transfer and metadata contract for the time-bin width, repetition rate and instrument response function.

`export_rois_geojson(app)` returns a GeoJSON `FeatureCollection`. Coordinates are image pixels in `[x, y]` order with the origin at the top-left. Fractional coordinates are preserved. ROI measurements are stored once under each feature's `statistics` property; import also accepts older payloads with flattened statistic fields. Rectangles and ellipses include their exact FLIMKit bounds in the feature properties; ellipse geometry is also represented by a 64-point polygon for other programs.

`import_rois_geojson(app, payload, mode='append')` accepts a GeoJSON `Feature` or `FeatureCollection` and returns the new FLIMKit region IDs. A plain GeoJSON polygon without FLIMKit properties becomes a polygon ROI, which is the normal path for data from Fiji. `mode='replace'` validates the whole payload before clearing existing regions. Invalid or unsupported geometry raises `ValueError` without partly importing the payload.

These functions may be called from a plugin's background thread. FLIMKit moves access to its GUI thread and blocks the calling thread until the GUI operation completes or raises an error. Plugin callers that need cancellation should manage it outside these synchronous bindings.

Loading order is built-ins, then installed packages that declare a `flimkit.plugins` entry point, then `~/.flimkit/plugins`, then `FLIMKIT_PLUGIN_PATH` and the folders in `plugins.paths`. Ids have to be unique across all of them, and the first registration of an id wins, so a later plugin cannot take an id off an earlier one.

Loading is isolated per plugin. If one raises on import, its registrations are rolled back, the traceback is kept in `flimkit.plugins.load_report()`, and the rest still load. A plugin that calls `sys.exit()` cannot take the app down with it.

`FLIMKIT_NO_PLUGINS=1` skips loading entirely, which gives a reproducible baseline for a published analysis.

### File formats

A reader class registers with `@file_format`, and FLIMKit's own readers keep working exactly as they did:

```python
from flimkit.plugins import file_format

@file_format(id='mine', label='My Format', exts=('.mine',), modality='time')
class MyReader:
    def __init__(self, path, **kwargs):
        ...
```

`modality` is `time`, `frequency` or `intensity`, and it is what `file_modality()` reports. The extension then works everywhere a path is accepted, including `FLIMFile(path)` and the file dialogs.

A built-in extension always wins. Registering `.ptu` does not take `.ptu` away from the PicoQuant reader.

For a format that has no extension of its own, register a sniffer:

```python
from flimkit.plugins import format_sniffer

@format_sniffer(tier='magic')
def sniff(path):
    with open(path, 'rb') as fh:
        if fh.read(7) == b'MYMAGIC':
            return 'mine'
    return None
```

`tier='magic'` runs after the built-in extension table and the built-in magic-byte checks, which is the safe place. `tier='extension'` runs before the extension table and can therefore take a file away from a built-in reader, so use it only for a format that genuinely shares an extension with something else. A sniffer that raises is reported and skipped.

### Phasor filters

```python
from flimkit.plugins import phasor_filter

@phasor_filter(id='mine', label='My Filter')
def mine(real, imag, sigma=1.0):
    return real, imag
```

The filter is then usable anywhere the `gaussian`, `median` and `wavelet` methods are, and `flimkit.phasor.filters.phasor_filter_methods()` lists it. Only the keyword arguments your function declares get passed to it. The three built-in methods cannot be overridden.

### Running at startup

A plugin that needs to be doing something from the moment FLIMKit opens, rather than waiting for a menu click, registers a startup callback:

```python
from flimkit.plugins import startup

@startup('my_server', order=200)
def start(app):
    ...
```

The callback runs once, with the GUI object, after the window is built. `order` sorts them low first. A startup that raises is reported on the console and the remaining ones still run, so a broken plugin cannot stop FLIMKit opening.

Do not block in a startup callback. It runs on the UI thread before the window is handed to the user, so anything long-lived belongs on a daemon thread that the callback starts and returns from.

### Buttons in the ROI panel

Some actions belong beside the controls they relate to rather than in a menu. A plugin can add a button to the ROI panel's action grid:

```python
from flimkit.plugins import panel_button

@panel_button('send_somewhere', 'Send to Somewhere', panel='roi', order=200)
def send(app):
    ...
```

`panel` currently accepts `'roi'` only, and an unknown panel is refused at registration rather than ignored. Buttons appear under FLIMKit's own, three to a row, sorted by `order` then label. The callback receives the same GUI object a `@tool` callback does.

`id` has to be unique across everything loaded, the same as tools. Registering an id twice is refused, and the error names the plugin that got there first.

### Installing a plugin

Put the `.py` file in `~/.flimkit/plugins/`. A folder with an `__init__.py` works too, if the plugin needs more than one file. Names starting with `_` or `.` are skipped, and the rest load in alphabetical order after the built-ins.

FLIMKit never creates that folder and does not load from it until you say so. Open `Help > Plugins...` and press Enable, or set `plugins.allow_user_plugins` to `true` in `~/.flimkit/config.json`. If the folder already has files in it the first time you start FLIMKit, you get asked once. Enabling takes effect on the next start.

`Help > Plugins...` lists what loaded, what failed and why, so a plugin that raises on import can be diagnosed without going near the log files.

`FLIMKIT_PLUGIN_PATH` takes a colon-separated list of extra folders, scanned last. It is meant for development, and it is not covered by the enable setting: setting an environment variable is already a deliberate act.

A wheel works in that folder too, which is the only way to install a packaged add-on into the compiled app, since it has no `pip`. Download the `.whl` from the add-on's releases, drop it in, and restart. FLIMKit puts it on the import path before it looks for entry points, so the add-on registers exactly as it would if it had been pip installed.

Two limits on that route. The wheel has to be pure Python, tagged `py3-none-any`; one built for a specific platform cannot be imported from a zip and is refused with that reason rather than failing later. And its dependencies have to be ones FLIMKit already bundles, because there is nothing in the compiled app to fetch the rest with. `numpy`, `scipy`, `matplotlib`, `pandas`, `tifffile` and tkinter are there.

Running from source, `pip install` is the better route and this is unnecessary.

### Settings a plugin owns

```python
from flimkit.plugins import plugin_config

cfg = plugin_config('my_plugin')
cfg.set('threshold', 12)
cfg.save()
threshold = cfg.get('threshold', 10)
```

That writes to a `plugin:my_plugin` section of `~/.flimkit/config.json`, which FLIMKit itself never reads. A plugin cannot reach the `expert` or `preferences` sections through this, so it cannot change how the fitters behave behind your back.

### Shipping a plugin as a package

A plugin that has dependencies of its own, or that you want people to install with `pip`, declares an entry point in its own `pyproject.toml`:

```toml
[project.entry-points.'flimkit.plugins']
my_plugin = 'my_plugin.register'
```

The module named on the right is imported at startup, so put the decorated functions there. Installed packages load after the built-ins and before anything in `~/.flimkit/plugins`, and they are not gated by the user-folder switch, since `pip install` is already a deliberate act. They do respect the master switch and the per-plugin disable list, under the entry point name.

This is how a plugin distributed to other people should be shipped. The folder is for a script you wrote yourself.

The frozen macOS and Windows builds cannot see entry points at all, since there is no site-packages to look in. A plugin that has to work in the compiled app goes in the folder.

### Preferences

`File > Preferences...` has a Plugins tab with the three settings that decide what gets loaded:

| Setting | Config key | Default |
|---|---|---|
| Load plugins at startup | `plugins.enabled` | `true` |
| Load from `~/.flimkit/plugins` | `plugins.allow_user_plugins` | `false` |
| Extra plugin folders | `plugins.paths` | empty |

Turning the first one off skips everything, built-ins included, which is the config equivalent of `--no-plugins`. Folders in `plugins.paths` are scanned after `FLIMKIT_PLUGIN_PATH` and are subject to the same treatment: they are an explicit choice, so they load without the `~/.flimkit/plugins` switch. All three take effect on the next start.

### Turning one off

`Help > Plugins...` has a checkbox per plugin. Unticking one adds it to `plugins.disabled` in the config and it stops loading on the next start. Built-ins can be turned off the same way, so ticking `core_tools` off empties the Tools menu.

From the command line:

```bash
python main.py --no-plugins              # load nothing, built-ins included
python main.py --plugins /path/to/dir    # extra folder, repeatable
```

`--no-plugins` is the same switch as `FLIMKIT_NO_PLUGINS=1`.

### Compatibility

`FLIMKIT_PLUGIN_API` is 1 and the hooks documented above are frozen at that version. New hooks get added; the ones here keep their arguments and their meaning. `flimkit_tests/tests/fixtures/plugin_api_v1/` holds a plugin written against v1 that the test suite loads on every run, so a change that would break an installed plugin fails the build rather than reaching a release.

### Trust

A plugin is ordinary Python. It runs inside FLIMKit with your account's access to your files and your network, and there is no sandbox between the two. The trust decision is the same one you make installing a Fiji plugin or a pytest plugin: read it, or get it from someone you would trust with the machine.

---

## QuPath Bridge

ROIs can already be exported as GeoJSON and imported back, which is enough if you are happy moving files by hand. The [QuPath bridge](https://github.com/FLIMKit/flimkit-qupath-bridge) removes that step: FLIMKit serves its images and ROIs over a loopback HTTP connection, and QuPath reads and writes them directly.

It runs inside a live QuPath session rather than as a script, so it works with the image you have open and the annotations you have drawn.

### Installing

Two halves, one on each side. Both are attached to the bridge's release.

1. Install `flimkit_qupath_bridge-*.whl` into the environment FLIMKit runs in, or drop it in `~/.flimkit/plugins/`.
2. Drop `qupath-extension-flimkit-bridge-*.jar` into QuPath's extensions directory, normally `~/QuPath/v0.7/extensions`.

QuPath 0.7.0 or newer is required, and FLIMKit 0.11.0 or newer for the plugin bindings and the startup hook.

### Using it

The bridge starts with FLIMKit. There is nothing to launch and no port to configure. FLIMKit writes its address and a generated token to `~/.flimkit/qupath-bridge.json`, and QuPath reads that file, so `Extensions > FLIMKit bridge > Connect` needs nothing typed in. If port 8765 is busy an ephemeral one is used and recorded in the same file.

From QuPath:

- **Add FLIMKit images to project** puts the intensity and lifetime maps into the open project as float32 images, in real units rather than a colourmapped render, so they can sit beside a brightfield or mIF image in the viewer grid.
- **Send annotations to FLIMKit** posts the annotations on the current image.
- **Fetch ROIs from FLIMKit** pulls FLIMKit's regions in.

From FLIMKit, the **Send to QuPath** button in the ROI panel reports whether QuPath has connected and what is being served. If no QuPath has paired it says so rather than failing quietly.

### Co-registration

FLIMKit expects ROIs in FLIM image-pixel coordinates, so anything drawn on another image has to be transformed into that space first. That happens on the QuPath side.

This needs QuPath's [alignment extension](https://github.com/qupath/qupath-extension-align), which QuPath does not ship and which has to be installed separately. The bridge deliberately contains no alignment code of its own.

1. Open the brightfield or mIF image and add the FLIMKit images to the same project.
2. Align them with the alignment extension and transfer the annotations onto the FLIM image.
3. Send the annotations on the FLIM image to FLIMKit.

Without the alignment extension you can still exchange images and ROIs, but only between images that already share a coordinate system.

### Security

The bridge listens on `127.0.0.1` only, and refuses any request whose `Host` header is not localhost, which stops a web page reaching it by resolving its own hostname to your machine. Every endpoint except the status check requires the token.

Both programs therefore have to be on the same machine. If they are not, forward the port over SSH rather than exposing it:

```bash
ssh -L 8765:127.0.0.1:8765 you@the-flimkit-machine
```

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
Restart the app - the default IRF path is resolved at startup and won't update mid-session.

**Per-pixel fitting is very slow**  
That's expected for large FOVs on CPU. Try increasing `--binning` to aggregate pixels before fitting, or switch to summed-only mode if you don't need spatial maps. If you have a supported GPU (Apple Silicon, NVIDIA, AMD) and ran `python install.py`, GPU acceleration is detected and used automatically, no extra flags needed. Note that `--free-tau-perpixel` with n_exp ≥ 2 uses batched Adam on GPU; it only falls back to CPU when no backend is detected.

**Tile stitching produces visible seams**  
Check that the max drift setting isn't too restrictive. If registration looks fine but seams persist, it's likely a sample contrast issue at tile boundaries rather than a registration failure.

**ROI holes are lost on GeoJSON import**  
Known limitation... Only the outer boundary is imported. Donut-shaped ROIs with holes lose the hole geometry on import.

**Phasor calibration looks off**  
Make sure the IRF file is from the same acquisition session. XLSX-based IRFs from FLIM microscope software can vary between sessions, which is why the machine IRF exists.

**Lifetimes are slightly higher than FLIM microscope software for the same data**  
This is expected and systematic. FLIMKit anchors the IRF at the steepest-rise point of the leading edge, which differs from how FLIM microscope software places the IRF. The offset is consistent across acquisitions and does not indicate a fitting problem.

---

## References

If you use FLIMKit in published work, please also cite the relevant dependencies where appropriate:

**Lifetime distribution fitting** - theoretical basis for Gaussian and Lorentzian α(τ) models:
> Lakowicz, J.R. (2006). *Principles of Fluorescence Spectroscopy* (3rd ed.). Springer. §4.11.2 (Lifetime Distributions), pp. 141-144.

**PhasorPy** - phasor computation, calibration, and cursor analysis:
> Gohlke, C. et al. PhasorPy. Zenodo. https://doi.org/10.5281/zenodo.13862586

**ptufile / sdtfile / lfdfiles / tifffile** - instrument file decoding for PicoQuant `.ptu`, Becker & Hickl `.sdt`, the SimFCS and ISS formats, and TIFF:
> Gohlke, C. https://github.com/cgohlke/ptufile, https://github.com/cgohlke/sdtfile, https://github.com/cgohlke/lfdfiles, https://github.com/cgohlke/tifffile

**photonsfile** - Photonscore LINCam `.photons` (D7) decoding:
> Hunt, A. and A. Akram. photonsfile. Zenodo. https://doi.org/10.5281/zenodo.21360199

**Tile stitching** - phase-correlation registration algorithm:
> Preibisch, S., Saalfeld, S. and Tomancak, P. (2009). Globally optimal stitching of tiled 3D microscopic image acquisitions. *Bioinformatics* 25(11), 1463-1465. https://doi.org/10.1093/bioinformatics/btp184

**Cellpose-SAM** - cell segmentation model used for masking:
> Pachitariu, M. and Stringer, C. (2025). Cellpose-SAM: segment anything in microscopy images. *bioRxiv*. https://doi.org/10.1101/2025.04.28.651001

---

## Acknowledgements

FLIMKit is designed, developed, and maintained by Alex Hunt. Anthropic's Claude AI was used as an assistant for parts of the GUI implementation, compiled app builds, and Docker packaging; all scientific design, fitting/phasor methods, validation, and the overall architecture are the author's own work.

FLIMKit reads several instrument formats. PicoQuant `.ptu` and Becker & Hickl `.sdt` reading is delegated to Christoph Gohlke's `ptufile` and `sdtfile` libraries, the SimFCS (`.b&h`, `.bhz`, `.ref`, `.r64`) and ISS (`.ifli`, `.iss-tdflim`) formats to his `lfdfiles`, and all TIFF reading, including ImSpector FLIM TIFF and PhasorPy OME-TIFF, to his `tifffile`; thank you to Christoph Gohlke for maintaining them, and for PhasorPy, which FLIMKit uses as its phasor backbone. Photonscore `.photons` reading is delegated to `photonsfile`, which was written for FLIMKit and spun out as a standalone library so it can be used without FLIMKit. Per-format provenance is in each reader's `NOTICE.md` (`flimkit/formats/<FORMAT>/NOTICE.md`).

---

## Contact

Alex Hunt - alexander.hunt@ed.ac.uk
