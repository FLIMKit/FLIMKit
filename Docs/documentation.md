# FLIMKit Documentation

> **v0.13.5** - Python toolkit for Fluorescence Lifetime Imaging Microscopy

> **Warning:** Active development. Cross-validate results with other software before drawing conclusions.

---

## Table of Contents

1. [Overview](#overview)
2. [Supported Input Formats](#supported-input-formats)
3. [Requirements & Installation](#requirements--installation)
4. [Quick Start](#quick-start)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [Workflows](#workflows)
   - [Desktop GUI](#desktop-gui)
   - [Guided Terminal UI](#guided-terminal-ui-mainpy)
   - [Machine IRF Setup](#machine-irf-setup-required)
   - [FLIM Reconvolution Fitting (CLI)](#flim-reconvolution-fitting-cli)
   - [Timelapse and Z-stack Fitting](#timelapse-and-z-stack-fitting)
   - [Synthetic Data Generation (CLI)](#synthetic-data-generation-cli)
   - [Phasor Analysis (CLI)](#phasor-analysis-cli)
   - [Python API](#python-api)
7. [Configuration Reference](#configuration-reference)
8. [Module Reference](#module-reference)
9. [Project Structure](#project-structure)
10. [Compiled App](#compiled-app-macos--windows--linux)
11. [Plugins](#plugins)
12. [Testing](#testing)
13. [Outputs & File Formats](#outputs--file-formats)
14. [Troubleshooting](#troubleshooting)
15. [Contact](#contact)

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
| `phasorpy` (0.12) | Phasor computation, calibration, cursor masking, spatial filtering, lifetime conversion |
| `PyWavelets` | Wavelet-based phasor denoising |
| `ptufile` | PicoQuant `.ptu`, `.bin`, `.phu` decoding |
| `sdtfile` | Becker & Hickl `.sdt` decoding |
| `lfdfiles` | SimFCS `.b&h`, `.bhz`, `.ref`, `.r64` and ISS `.ifli`, `.iss-tdflim` decoding |
| `photonsfile` | Photonscore LINCam `.photons` (D7) decoding |
| `inquirer` | Interactive terminal prompts |
| `numba` | Speeds up the `.photons` decode inside `photonsfile`, with a numpy fallback |
| `shapely` | Repairs self-intersecting ROI rings on GeoJSON export |
| `lz4` | LZ4-compressed Becker & Hickl `.sdt` blocks |
| `ipywidgets` | Jupyter notebook interactive support, optional |
| `cellpose` (≥ 3.0) | Deep-learning cell segmentation (Cellpose-SAM) for cell masking, optional |
| `opencv-python-headless` | Image I/O, resizing, and general image processing |
| `openpyxl` | Excel XLSX parsing for FLIM microscope software IRF extraction |
| `pandas` | Excel/XLSX IRF file parsing |
| `tifffile` | TIFF image I/O |
| `zarr` | OME-Zarr export of the result maps |
| `tqdm` | Progress bars |

### Installation

#### No Python, no terminal

Download the build for your machine from the
[Releases](https://github.com/FLIMKit/FLIMKit/releases/latest) tab, unzip it,
and run it. Python is inside it, so there is nothing else to install.
`FLIMKit-windows.zip`, `FLIMKit-macos.zip` and `FLIMKit-linux.zip` are attached
to every release, built by GitHub Actions on each tag.

On macOS the app is self-signed, so the first launch needs right-click then
Open. A double-click will be refused.

This route gets the desktop application only. It carries no GPU backend unless
the build machine had one, which is covered under GPU acceleration in the
compiled app.

#### From PyPI

For analysis in scripts, notebooks and the terminal:

```bash
pip install flimkit                 # readers, fitting, phasors, stitching, the CLI
pip install "flimkit[gui]"          # adds the desktop window
```

This is the right route for a server, a container, or anywhere you only want
the library. It installs no Tkinter packages, so it works on a machine with no
display and no `python3-tk`.

Extras combine, so ask for whichever apply:

| Command | Packages | What you get |
|---|---|---|
| `pip install flimkit` | 52 | Readers, fitting, phasors, stitching and the terminal CLI. No desktop, no GPU. |
| `pip install "flimkit[gui]"` | 54 | The desktop window as well. |
| `pip install "flimkit[torch]"` | 62 | Headless with GPU fitting. |
| `pip install "flimkit[gui,torch]"` | 64 | Desktop and GPU. |
| `pip install "flimkit[segmentation]"` | 70 | Cellpose cell masking. Cellpose depends on PyTorch, so this brings GPU fitting with it. `flimkit[cellpose]` is the same thing under the package's own name. |
| `pip install "flimkit[notebook]"` | 71 | The Jupyter phasor cursor tool. |
| `pip install "flimkit[all]"` | 91 | `gui`, `notebook` and `segmentation` together, and GPU by way of Cellpose. |
| `pip install "flimkit[test]"` | 56 | pytest, for running the suite against an installed copy. |

Counts are from a resolve on macOS and will differ a little by platform,
mostly in how much PyTorch brings with it.

A pip install fits on the CPU. GPU acceleration needs a backend, and what pip
can give you depends on the platform:

| Platform | Command | What you get |
|---|---|---|
| Apple Silicon | `pip install "flimkit[mlx]"` | MLX on Metal, the fastest path on a Mac |
| Apple Silicon or Intel Mac | `pip install "flimkit[torch]"` | PyTorch MPS |
| Linux, NVIDIA | `pip install "flimkit[torch]"` | CUDA. PyPI's Linux wheel pulls the CUDA runtime itself |
| Windows, NVIDIA | see below | pip alone gives CPU only |
| AMD, ROCm | see below | not on PyPI at all |

Windows and ROCm need a wheel from PyTorch's own index rather than PyPI, which
package metadata cannot ask for. That is what `install.py` is for:

```bash
git clone https://github.com/FLIMKit/FLIMKit.git
cd FLIMKit
python install.py
```

`install.py` installs the requirements, then detects the hardware and installs
the matching GPU backend (MLX on Apple Silicon, CUDA on NVIDIA, ROCm on AMD,
CPU-only otherwise). No flags needed for a standard install. Clone it too if
you want the terminal CLIs at the repository root, `fit_cli.py`,
`phasor_cli.py` and `synth_cli.py`, which are not part of the package.

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

## Step-by-Step Guide

A worked analysis of one field of view, `Ado_1.ptu`, from opening the file to exporting the maps, followed by a project folder with a z-stack, the tile stitching form, and a list of every file FLIMKit reads and writes. Every screenshot is the desktop GUI running from source (v0.13.5 plus the commits after it), and every number quoted was produced by those runs. Your numbers will differ with your data and your machine IRF, but the order of the steps and the reasons for each setting carry over.

`Ado_1.ptu` is a single 1024 x 1024 px field of view: 1,714,555 photons in channel 1, 529 TCSPC bins of 96.97 ps, a 19.505 MHz laser (51.27 ns period) and 127.5 s of acquisition.

### Step 1: Open the file

Start the GUI (`python main.py`, or the compiled app) and stay in Single FOV Fit mode.

![FLIMKit on start-up](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/01_start.jpg)

The window has four areas. The Project list on the left fills when you open a folder ([Step 13](#step-13-work-from-a-project-folder)). The settings form sits in the middle, with the Progress log, Fit Summary and Images tabs underneath. The FOV Preview on the right shows the intensity image, the lifetime map and the summed decay.

Input Files:

| Option | What it does |
|---|---|
| Analysis: Single FOV | One file, one field of view. |
| Analysis: Z-stack | A folder of `region_zX.ptu` slices fitted as one field of view ([Step 14](#step-14-fit-a-z-stack)). |
| Data file | The FLIM file. Any format in [Supported Input Formats](#supported-input-formats) works. Browse, type the path, or drag the file onto the box. |
| LAS X export (optional) | The `.xlsx` (or delimited text) exported from the LAS X FLIM window for this same file. It is only needed for the Analytical model IRF, and it lets you compare against the LAS X fit. |

Load `Ado_1.ptu`. The preview draws the intensity image and the summed decay straight away, and the status line under the preview gives the image size and photon count.

![Ado_1.ptu loaded](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/02_loaded.jpg)

If a `Ado_1.roi_session.npz` already sits next to the file, FLIMKit restores the last fit, the ROIs and the form settings from it instead of starting clean. The log says `[Auto-Load] No session found` when there isn't one.

Look at the decay before you change anything. This one rises at about 2.6 ns, decays over roughly three decades, and has a small peak at about 47 ns. That peak is the start of the next laser period, and FLIMKit finds it and cuts the fit window short of it on its own ([Step 6](#step-6-run-a-fast-fit-first)).

### Step 2: Choose the IRF

![IRF options](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/03_irf.jpg)

The instrument response function is the largest single source of systematic error in a reconvolution fit, and it matters most for the shortest component. The ⓘ next to each section heading opens the same explanations inside the app.

| Option | When to use it |
|---|---|
| Analytical model (LAS X export) | Builds the IRF from the LAS X export named in Input Files. The default for FALCON data when you have the export for this file. |
| Machine IRF (.npy pre-built) | A stored IRF for your microscope, built once from matched pairs ([Machine IRF Setup](#machine-irf-setup-required)). Use it when the optical path hasn't changed since it was built. |
| Machine IRF + full σ broadening | The same stored IRF, which the fit may broaden with a Gaussian of up to 3.0 bins. For when the stored IRF is narrower than the real response, after a change of objective or pinhole for example. |
| Machine IRF + half σ broadening (σ≤0.5) | Broadening capped at 0.5 bins. For a stored IRF that is close, where a free σ would start absorbing the decay. |
| Measured IRF file (scatter PTU or .pck) | An IRF recorded alongside the sample from a scattering solution or a reflective surface. The most defensible choice when you have one. |
| Estimate from decay - raw | Takes 21 bins around the decay peak as the IRF. A last resort: it can't separate the instrument from the fastest part of the decay, so short lifetimes come out long. |
| Estimate from decay - parametric | Fits a pulse shape to a 1.5 ns window around the peak. Smoother than raw, with the same bias. |
| Gaussian (fallback) | A Gaussian of the configured width. |

The Machine IRF path box appears under the options when a machine IRF method is picked, filled with your default. The default can be changed in File > Preferences... > Files ([Machine IRF Setup](#machine-irf-setup-required)). For `Ado_1.ptu` I used Machine IRF, since there is no LAS X export for this file.

### Step 3: Set the fitting parameters

![Fitting parameters and masking](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/04_fit_params.jpg)

| Option | What it does |
|---|---|
| Fit model: n-exp | A sum of 1, 2 or 3 discrete exponentials convolved with the IRF. The right choice for most samples. |
| Fit model: Gaussian dist. | One continuous distribution of lifetimes, reported as a centre and a width. For a fluorophore in a heterogeneous environment where a discrete fit needs too many components. Slower, since each evaluation integrates over a 200-point lifetime grid. |
| Fit model: Lorentzian dist. | The same with heavier tails, so a small sub-population far from the centre doesn't drag the centre towards it. |
| Fit model: n-exp tail | Discrete exponentials fitted past the decay peak with no IRF. Fast, and the IRF drops out as a source of error, but components shorter than about the IRF width come out biased. The IRF section is hidden when this is picked. |
| Components | 1, 2 or 3 for n-exp and tail, or 1 (unimodal) / 2 (bimodal) for the distributions. Start low and add a component only when the residuals show structure ([Step 6](#step-6-run-a-fast-fit-first)). |
| Fitting mode: Full | Fits the summed decay, then every pixel. This is what makes the lifetime map. |
| Fitting mode: Fast | The summed decay only. Seconds instead of minutes, with no map. Use it to check the IRF, the model and the component count first. |
| τ bounds (ns) | The lower and upper bounds on the fitted lifetimes (0.145 and 45 ns by default). The part of the decay that is fitted is set separately, in Expert Settings. |
| Output prefix | Where the output files go and what they're called. It fills with the file name, so outputs land next to the data. A bare name is placed in the data folder; a full path is used as it is. |

### Step 4: Masking and thresholding

![Masking and the run buttons](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/05_masking_run.jpg)

| Option | What it does |
|---|---|
| Apply cell mask (Cellpose-SAM) | Segments cells in the intensity image and fits only inside them, so background pixels don't add noise-dominated lifetimes to the statistics. It runs on the GPU when there is one. Needs the `segmentation` extra. |
| Intensity threshold (min photons/px) | Skips pixels below this count. Blank means no threshold. |
| Apply Coates pile-up correction | Corrects the early-photon bias at high count rates. The log prints the photons per pulse at the start of each fit: tick this above about 5%. The corrected decay is no longer Poisson, so χ²_r is unreliable with it on, though the lifetimes are not. |
| Time-varying background PTU | A separate acquisition of a fluorophore-free region (medium, buffer). Its decay shape is fitted as a scaled background instead of a flat offset ([Time-varying background correction](#time-varying-background-correction)). |

`Ado_1.ptu` ran at 0.0007 photons per pulse (0.07%), so pile-up correction stays off. I left the mask off for the first fits to show what it changes in [Step 8](#step-8-mask-out-the-background).

### Step 5: Expert settings

Expert Settings opens a dialog shared by the single FOV and tile pipelines. When anything in it differs from the defaults, an orange "Custom expert settings active" line appears above the run button.

![Expert Fit Settings](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/06_expert.jpg)

| Option | What it does |
|---|---|
| Optimizer: Differential Evolution (DE) | A global search over the bounded parameter space with a Levenberg-Marquardt polish at the end. It doesn't depend on a starting guess. The default. |
| Optimizer: Levenberg-Marquardt (LM) | A local fit from several starting points. Much faster, and it can miss the global minimum on a multi-exponential decay with few restarts. |
| DE population / DE max iterations | DE search size (30 and 5000). |
| LM random restarts | Number of LM starting points (8). |
| Spatial binning (NxN) | Sums N x N pixels before the per-pixel fit. 1 is no binning. See [Step 7](#step-7-full-fit-and-binning). |
| CPU workers | Cores for DE. -1 uses all of them. The compiled app is limited to 1. |
| Min photons/pixel | Pixels below this count are left out of the per-pixel fit (10). |
| Cost function | Poisson deviance (default) or the legacy Neyman χ². Poisson is correct for photon counts, above all at low counts. |
| Channel filter | Detector channel. Blank uses all channels. |
| IRF FWHM (ns) | Width for the estimated and Gaussian IRFs. Blank is one TCSPC bin. |
| IRF alignment | Where the IRF is anchored against the decay. Steepest rise (default) or the legacy decay peak. |
| IRF shift bound (±bins) | How far the fit may shift the IRF in time. 2 is recommended; 5 is the old default. |
| Align measured IRF peak to the decay rising edge | For a scatter PTU or `.pck` recorded in a separate acquisition, whose time zero won't match the sample's. |
| Fit window start / end (ns) | The part of the decay that is fitted. Blank lets FLIMKit choose from the IRF onset and the end of the period. |
| Exclude bands (ns) | Stretches of the decay to leave out, such as a reflection peak, written `7.2-8.8` or `7.2-8.8,11.0-11.5` ([Fit window and exclusion bands](#fit-window-and-exclusion-bands)). |
| Free τ per pixel | Lets lifetimes float in every pixel instead of locking them to the summed fit. Slower, and it shows spatial variation in τ when n_exp > 1. |
| Pile-up in the model | Fits pile-up as part of the model instead of rescaling the decay. Needs free τ per pixel and n_exp > 1. |
| Background in the model | Fits the offset instead of subtracting it. One exponential with fixed τ, CPU only. |
| Free t0 | Tail fits only. Lets the start time float, which correlates with the amplitudes. |
| Confirm / Reset Defaults / Cancel | Apply, restore the defaults, or close without changes. |

For `Ado_1.ptu` I left everything at the defaults until [Step 7](#step-7-full-fit-and-binning).

### Step 6: Run a fast fit first

With Fast mode and 2 components, Run Single-FOV Fit took a few seconds.

![2-exponential fast fit](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/07_fast_2exp.jpg)

The Progress log records every choice the fit made, so it's the first place to look when a result seems wrong.

![Progress log for the fast fit](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/08_log.jpg)

Section [1] reads the file header, and section [2] gives the count rate and photons per pulse used to decide on pile-up correction. Section [5] shows the fit window chosen: here FLIMKit found the next-period artefact at bin 484 (46.93 ns) and fitted bins 19-464 (1.84-44.99 ns) automatically.

The 2-exponential result was τ₁ = 4.04 ns and τ₂ = 0.61 ns with χ²_r(tail) = 18.29, and the residuals have a clear wave across the first 20 ns. That structure means the model is missing something. Changing to 3 components and running again dropped χ²_r(tail) to 1.53 and flattened the residuals, so I kept three. A lower χ² on its own doesn't justify an extra component, since more parameters always fit better. Check that the new lifetimes are physically distinct and that the residuals improved.

### Step 7: Full fit and binning

With 3 components, switch Fitting mode to Full and run again. A progress window counts the pixels as they're fitted.

![Per-pixel fit running](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/09_running.jpg)

At binning 1 only 34,680 of 507,977 signal pixels (6.8%) reached the 10-photon minimum, which the log flags as `PHOTON-STARVED`. When most of the image is starved, FLIMKit says so when the fit finishes and suggests a binning.

![Too few photons per pixel](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/10_photon_warning.jpg)

Binning sums an N x N block of pixels into one decay before fitting, trading resolution for photons per pixel. With Spatial binning set to 8 in Expert Settings, the map filled in.

![Full fit at 8 x 8 binning](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/11_binned_map.jpg)

The per-pixel fit keeps the summed-fit lifetimes and fits only the amplitudes in each pixel, unless Free τ per pixel is on. The map is the amplitude-weighted mean lifetime per pixel. The background between the cells now fills with noise-dominated values, which the next step removes.

### Step 8: Mask out the background

Tick Apply cell mask (Cellpose-SAM) and run again.

![Masked fit](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/12_masked.jpg)

The map now covers the segmented cells only. The summed decay is built from those pixels too, so the global fit changes with it: τ = 5.81, 1.94 and 0.398 ns and χ²_r(tail) = 1.244. Cellpose missed part of one cell at the top left here, so check the mask (saved as `<prefix>_cell_mask.png`) before trusting the per-cell statistics.

Fit Summary tab, row by row:

| Row | Meaning |
|---|---|
| τ1, τ2, τ3 | Component lifetimes from the summed fit, in ns. |
| α1, α2, α3 | Component amplitudes, in counts. |
| f1, f2, f3 (amp frac) | Amplitude fractions, αᵢ / Σα. |
| τ_mean (amp-weighted) | Σαᵢτᵢ / Σαᵢ. Weighted towards the short, high-amplitude components (1.125 ns here). |
| τ_mean (int-weighted) | Σαᵢτᵢ² / Σαᵢτᵢ. Weighted towards the long components, closer to what a phasor or a mean arrival time gives (2.834 ns here). |
| Background (fitted) | Constant offset per bin. |
| IRF shift | How far the IRF was moved to match the decay, in bins (0.913 bins, about 88 ps). |
| IRF σ (broadening) | Extra Gaussian width. 0 unless a broadening IRF method is used. |
| IRF FWHM (eff.) | Effective IRF width after any broadening. |
| χ²_r(tail) Neyman / Pearson | Reduced χ² over the tail of the decay, kept for comparison with LAS X. See [Fit Diagnostics](#fit-diagnostics). |

Running the same fit twice gives identical numbers, since both optimisers are seeded. I refitted Ado_1 from a project folder in [Step 13](#step-13-work-from-a-project-folder) and every value matched to the last digit.

### Step 9: Adjust the display

The FLIM Color Scale panel under the preview changes the picture, never the fitted values.

![Display controls with the decay plot hidden](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/15_display.jpg)

| Control | What it does |
|---|---|
| τ range Min / Max, Auto, Update | Colour limits in ns. Auto sets them to the 2nd and 98th percentiles of the map (0.90-1.90 ns for Ado_1). Type values and press Update to set them yourself. Values outside the range are clamped, not removed. |
| Γ | Gamma on the colour scale. 1.0 is linear. |
| Colormap | Colour map for the lifetime image. |
| Show Decay Plot | Hides the decay and residuals so the images get the whole panel. |
| View: FLIM / Intensity | With the decay plot hidden, which of the two images fills the panel. |
| τ weighting: Amplitude / Intensity | Recomputes the map as the amplitude- or intensity-weighted mean lifetime from the stored per-pixel amplitudes. No refit. |
| Intensity: Min / Max, Auto, colormap | The same for the intensity image. |

The colour scale is saved into the session file, so it comes back when the file is reopened.

### Step 10: Measure regions of interest

Open the ROI Analysis tab, pick a drawing mode and drag on either image.

![Two ROIs on Ado_1](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/13_roi.jpg)

| Control | What it does |
|---|---|
| Select | Click a region on the image to select it. |
| Rectangle / Ellipse | Drag a box. |
| Polygon | Click the vertices, then right-click to close the shape (three points minimum). |
| Freehand | Draw the outline. |
| Clear All | Removes every region. |
| Delete Selected / Rename... | Act on the region selected in the list. |
| Import from GeoJSON | Loads regions drawn elsewhere, QuPath for example. Only the outer boundary of each shape is kept, so holes are lost. |
| Export as CSV | The Regions table as a spreadsheet. |
| Export as GeoJSON / Export All as GeoJSON | The selected or all regions, with their statistics, for QuPath. |
| Fit ROI Decay / View Fit | Fits the summed decay of the selected region(s) on its own ([Per-ROI decay fitting](#per-roi-decay-fitting)). |
| Send to a viewer | Added by the `flimkit-bridge` plugin. Sends the current field of view to a connected viewer such as QuPath ([QuPath Bridge](#qupath-bridge)). |

The table gives, per region, the mean, median and standard deviation of the pixel lifetimes and the photon count. The ellipse over the lower cell gave τ_mean = 1.22 ns (median 1.18, SD 0.21) from 163,093 photons, and the rectangle over the upper cell 1.22 ns (median 1.19, SD 0.23) from 138,519. Regions are written to the session file as soon as they're drawn.

### Step 11: Export

Export Images..., under the Fit Summary table, saves the images shown in the preview.

![Export Results dialog](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/14_export.jpg)

| Option | What it does |
|---|---|
| Images to Export | The images the fit produced, Intensity and Lifetime here. All / None tick or clear them. |
| Include scale bar | Draws a µm scale bar when the file carries a pixel size. `Ado_1.ptu` doesn't, so the log said `No pixel size available - scale bar will be omitted`. |
| Include ROI annotations | Draws the regions on the images. |
| PNG | The images as displayed, with the current colour map and limits. For slides and quick looks. |
| OME-TIFF | `<scan>_lifetime.ome.tiff` as 32-bit floats in ns, with unfitted pixels left as NaN, and `<scan>_intensity.ome.tiff` as 32-bit photon counts. The pixel size goes into the OME metadata when the file carries one. These are the ones to measure from in Fiji/ImageJ. |
| OME-Zarr | One compressed store named after the scan, each image a channel ([OME-Zarr export](#ome-zarr-export)). |
| Save Location | The folder. |

Every export is named after the scan, so the PNG export of `Ado_1.ptu` writes `Ado_1_intensity.png`, `Ado_1_lifetime.png` and `Ado_1_summed_decay.png`, and exporting several files into one folder doesn't overwrite anything. What else the fit writes by itself is listed in [Step 16](#step-16-what-flimkit-reads-and-writes).

### Step 12: Phasor analysis

Switch the mode to Phasor Analysis. The file and the machine IRF carry over from the fit form.

![Phasor settings](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/16_phasor_setup.jpg)

| Option | What it does |
|---|---|
| Input Mode: New PTU file / Resume session (.npz) | Start from a file, or reopen a saved phasor session with its cursors. |
| PTU file | The FLIM file. |
| IRF XLSX (optional) / Machine IRF (optional) | The calibration. Phasors need an IRF to place the universal circle correctly. The XLSX wins if both are given. |
| Min photons (fraction) | Pixels whose mean count per time bin is below this are left out of the plot. The default 0.01 is about 5 photons for a 529-bin decay. |
| Max cursors | How many cursors can be placed. |
| Load & Analyse | Computes and plots the phasor. |

![Find Peaks and FRET settings](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/17_phasor_options.jpg)

| Option | What it does |
|---|---|
| Find Peaks: Smooth σ, Threshold, Find Peaks | Smooths the phasor histogram and places cursors on the peaks above the threshold. |
| FRET Analysis: Donor τ, Acceptor τ, Donor fretting | The donor lifetime, an optional acceptor lifetime (blank for donor-only), and the fraction of donors that transfer. |
| Overlay Trajectory / Fit Donor FRET / Clear Overlay | Draw the FRET efficiency trajectory, fit it to the data, or remove it. |

On the plot, a click places an elliptical cursor and the matching pixels light up in the image above. The toolbar above the plot has Clear all and Undo, Save session, Ellipse or Polygon cursors, the Radius and Minor/major sliders for the cursor shape, and Fit Cursor Decay / View Fit to fit the summed decay of the pixels under a cursor.

`Ado_1.ptu` gave 48,357 valid pixels at 19.5 MHz. A cursor of radius 0.05 on the main cloud of the raw phasor took 6,259 pixels with a phase lifetime τ_φ of 2.20-2.71 ns (median 2.45 ns).

![Unfiltered phasor](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/18_phasor_raw.jpg)

The raw cloud is wide because each pixel carries few photons. Phasor filter smooths the G and S coordinates in image space, which pulls the cloud together without changing where it sits:

| Filter | Parameter | What it does |
|---|---|---|
| none | | Raw phasor. |
| gaussian | σ | Gaussian smoothing of G and S. |
| median | size | Median filter of G and S. Removes outliers and keeps edges. |
| wavelet | | Wavelet soft-thresholding (Daubechies db4). It takes no parameter. This is the filter closest to the LAS X phasor display. |

Apply runs the filter and Reset goes back to the raw data. With the wavelet filter and the same cursor, the count went to 13,771 pixels and τ_φ to 2.20-2.70 ns (median 2.43 ns).

![Wavelet-filtered phasor](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/19_phasor_wavelet.jpg)

The phasor session is saved as `<file>_phasor.npz` next to the data whenever a cursor or the filter changes. Loading the same file again recomputes the phasor with the current IRF and then puts the saved cursors and filter back on it, the same way a fit session comes back, and the log says `[Auto-Load] Restored 1 cursor(s) and the wavelet filter from Ado_1_phasor.npz`. Clear all and Reset start again from nothing.

### Step 13: Work from a project folder

A project is a folder of acquisitions. File > Open Project Folder... lists every FLIM file in it, groups `region_zX` slices into z-stacks and picks up `.xlif` files for tiles. It writes a `project.json` into the folder to remember them. For this I made a folder holding `Ado_1.ptu`, `Ado_2.ptu` and the eight slices `Series008_z1.ptu` to `Series008_z8.ptu`.

![Project with a z-stack selected](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/20_project_zstack_loaded.jpg)

Each row shows a session mark (○ nothing saved, ● a fit, ◐ a phasor, ◉ both) and the scan type (F for a field of view, Z for a z-stack, T for tiles from an XLIF). The footer counts scans and saved sessions. Clicking a row loads it into the right form, restoring its fit and ROIs when there is a session.

To fit several files the same way, fit one, keep it selected, and press Apply fit settings... under the list.

![Apply Fit Settings](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/23_apply_settings.jpg)

The dialog shows the model, IRF, lifetime bounds and display scales it will copy, and lists the other single-FOV files. Files that already have a fit are marked ● and their fit is replaced; ROIs aren't copied. I refitted Ado_1 in the project (3 components, machine IRF, cell mask, 8 x 8 binning) and applied it to Ado_2, which fitted to τ = 5.43, 1.58 and 0.264 ns with χ²_r(tail) = 2.29.

![Settings applied](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/24_apply_done.jpg)

Clicking Ado_2 afterwards reloads its saved fit. The residual plot isn't stored in the session, so it stays empty until the file is refitted, but the values are all there.

![Ado_2 reopened from the project](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/25_project_reopen.jpg)

### Step 14: Fit a z-stack

Selecting the Z row switches Analysis to Z-stack and fills the folder. The slider under the preview steps through the slices. The fit pools every slice to fit one set of lifetimes, then fits each slice per pixel with those lifetimes locked, so only the amplitudes change with depth ([Timelapse and Z-stack Fitting](#timelapse-and-z-stack-fitting)). The form is the same as for a single FOV, and the button reads Run Z-stack Fit.

With the machine IRF and 2 components, the eight 256 x 256 slices (1,731,281 photons pooled) gave τ₁ = 3.48 ns and τ₂ = 0.572 ns, and per-slice amplitude-weighted means of 1.90-2.00 ns.

![Z-stack result](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/21_zstack_result.jpg)

![Z-stack log](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/22_zstack_log.jpg)

The residuals show why χ²_r(tail) came out at 688.9: this 25.7 ns period has a reflection peak at about 20 ns, and the whole period was fitted (bins 0-265).

Expert Settings apply to z-stacks the same way as to a single FOV. Setting Exclude bands to `19.6-21.2` dropped those bins from the fit, 248 of 265 fitted, and moved the pooled lifetimes to τ₁ = 3.66 ns and τ₂ = 0.604 ns. The calibrated tail χ² in `Series008_reference_fit.json` went from 159.6 to 21.1. It isn't near 1, so look at the residuals for what is left before trusting the absolute values. Spatial binning applies to every slice as well. The Images tab steps through the per-slice maps.

### Step 15: Stitch tiles from an XLIF

Tile Stitch/Fit mode takes a tiled acquisition: the `.xlif` from the LAS X project's Metadata folder and the folder of PTU tiles exported with it.

![Tile Stitch/Fit form](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/26_stitch_form.jpg)

| Option | What it does |
|---|---|
| XLIF metadata | Tile positions and layout. |
| PTU tile directory | The folder of tiles. Tiles from several scans in one folder are sorted out by name. |
| Base output dir | A sub-folder named after the ROI is made inside it. |
| Rotate tiles 90° CW | Rotates each tile 90° clockwise before it is placed. On by default and recommended for Leica FLIM data. |
| Stitch tiles only | Registers and stitches the intensity image and the decay cube, without fitting. |
| Stitch then fit full ROI | Fits the whole mosaic at once. Needs tens of GB of RAM on large mosaics. |
| Per-tile fit | Fits a global decay for the lifetimes, then each tile with those lifetimes locked, and stitches the maps. The recommended option. |
| Multidimensional series | Per-tile fitting repeated over z and/or time, one stitched plane per (t, z) ([Multidimensional Series](#multidimensional-series-stitched-tiles-over-z-and-time)). |

Below the pipeline choice come the same IRF, model and masking sections as for a single FOV, then:

| Option | What it does |
|---|---|
| Per-pixel fitting | Needed for the lifetime maps and for ROI analysis on the mosaic. |
| Export τ-weighted / intensity-weighted / amplitude-weighted map | 32-bit TIFF lifetime maps. |
| Save individual component maps | One TIFF per τᵢ. |
| Lifetime display / Intensity display | Display limits for the exported images. Blank is automatic. |
| Phase-correlation registration, Max shift (px) | Corrects stage drift before stitching ([Tile registration](#tile-stitching-and-fitting)). Raise the shift above 120 px if the drift is larger. |
| Pool decay every N timepoints | Multidimensional series only. Subsamples the files used for the pooled lifetime fit. |
| IRF XLSX dir | A folder with one `<tile_name>.xlsx` per tile. Blank uses the IRF method above. The machine IRF is safer, since per-tile exports vary. |

### Step 16: What FLIMKit reads and writes

Files FLIMKit reads, apart from the FLIM data in [Supported Input Formats](#supported-input-formats):

| File | Where it's read | What it carries |
|---|---|---|
| `.xlsx`, `.csv`, `.tsv`, `.txt`, `.dat`, `.ascii`, `.asc` | LAS X export box, IRF XLSX, per-tile XLSX folder | The LAS X decay export: the IRF for the analytical model and the LAS X fit for comparison. |
| `.npy` machine IRF | Machine IRF path | The stored instrument response ([Machine IRF Setup](#machine-irf-setup-required)). |
| Scatter `.ptu` or `.pck` | Measured IRF file | A recorded IRF. |
| Background `.ptu` | Time-varying background PTU | A fluorophore-free decay fitted as background. |
| `.xlif` (or `.lif`) | Tile Stitch/Fit, project folders | Tile positions and layout. |
| `<file>.roi_session.npz` | Automatically when the file is opened, or File > Restore NPZ... | A saved fit, its ROIs, form settings and display scales. |
| `<file>_phasor.npz` | Phasor > Resume session | A saved phasor with its cursors. |
| `.geojson` | ROI Analysis > Import from GeoJSON, File > Import GeoJSON... | Regions from QuPath or any GeoJSON source. |
| `project.json` | File > Open Project Folder... | The project's scan list and output folder. |

Files FLIMKit writes:

| File | Written by | What's in it |
|---|---|---|
| `<prefix>_summed_<n>exp.png` | Every fit | Summed decay, IRF, fitted curve and residuals. |
| `<prefix>_pixelmaps_<n>exp.png` | Full fits | A figure of the per-pixel maps. |
| `<prefix>_lifetime_hist_<n>exp.png` | Full fits | Histogram of the per-pixel lifetimes. |
| `<prefix>_cell_mask.png` | Fits with the cell mask on | The Cellpose mask used. |
| `<file>.roi_session.npz` | Every fit, every ROI change, colour-scale changes | Fit results (summed and per-pixel arrays, decay, IRF, time axis, intensity, lifetime map), form settings, display scales and ROIs. Reopening the file restores all of it. File > Save NPZ / Save NPZ As... writes it on demand. |
| `<file>_phasor.npz` | Phasor mode, automatically, or Save session | Calibrated G and S before filtering, the mean intensity, frequency, the cursors (ellipses and polygons), their size and the filter applied. Restored when the file is loaded again. |
| `<scan>_intensity.png`, `<scan>_lifetime.png`, `<scan>_summed_decay.png` | Export Images..., PNG | The images as displayed, for slides. |
| `<scan>_intensity.ome.tiff`, `<scan>_lifetime.ome.tiff` | Export Images..., OME-TIFF | Intensity as 32-bit photon counts, lifetime as 32-bit floats in ns with NaN where nothing was fitted, and the pixel size when known. |
| `<scan>.ome.zarr` | Export Images..., OME-Zarr | One compressed store, each image a channel, the fit summary in the metadata. |
| ROI `.csv` | Export as CSV, File > Export > Export ROI Table CSV | Per region: ID, name, type, τ mean/median/SD, photons and photon SD, and the per-ROI fit (τ_mean, χ²_r, τᵢ and αᵢ) when there is one. |
| ROI `.geojson` | Export as GeoJSON / Export All as GeoJSON, and the File > Export menu | Each region as a GeoJSON feature with its name, type, colour and statistics (τ median and SD, photons and SD). Self-intersecting outlines are repaired and flagged. Opens in QuPath. |
| Fit summary `.csv` | File > Export > Export Summed Fit CSV | The Fit Summary table. |
| `project.json` | Opening a project folder | Scan list, types, source paths and output folder. |
| `<stack>_reference_fit.json` | Z-stack fits | The pooled lifetimes, bounds, photons and slices used. |
| `<stack>_intensity_stack.npy`, `_tau_mean_amp_stack.npy`, `_alpha_<i>_stack.npy`, `_chi2_r_stack.npy`, `_calibrated_chi2_r_stack.npy` | Z-stack fits | `(Z, H, W)` arrays, one plane per slice. |
| `<stack>_zseries.csv` / `.json` / `.png` | Z-stack fits | Per-slice summary: mean amplitudes, mean τ and its SD, χ²_r and pixels fitted, with a plot. |
| `reference_decay.npz` | Z-stack fits | The pooled decay the reference lifetimes were fitted to. |
| `z0001/`, `z0002/`, ... | Z-stack fits | Per-slice fit plots and maps. |
| Stitched `.npy` cube, intensity and τ TIFFs, CSV summary | Tile Stitch/Fit and Batch ROI | See [Tile stitching and fitting](#tile-stitching-and-fitting). |

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

The **Fit ROI Decay** button fits the summed decay from the selected region(s) independently, rather than using the whole-FOV fit.

**Selecting regions:**
- Single region: click it in the list, then click **Fit ROI Decay**.
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

To use an IRF kept somewhere else, a shared drive or one per microscope for example, set it in File > Preferences... > Files > Default machine IRF (.npy). Browse picks the file, Use the built-in IRF clears the setting, and Save checks the file loads before accepting it. The choice is stored in `~/.flimkit/config.json`, so it applies to every form straight away and stays set after a restart. The terminal tools read the same setting. If the chosen file later goes missing, FLIMKit says so in the log and falls back to the built-in one.

![Default machine IRF in Preferences](https://raw.githubusercontent.com/FLIMKit/FLIMKit/main/Docs/images/guide/27_prefs_machine_irf.jpg)

A newly built `machine_irf_default` is picked up after a restart, unless a different default is set in Preferences.

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
| `--binning INT` | Spatial binning (NxN) before the per-slice per-pixel fit (default: 1) |
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

The GPU backends honour the window too. All four per-pixel kernels take `fit_idx` and restrict themselves to it, so excluding a reflection peak does not cost the GPU.

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
| `MACHINE_IRF_DEFAULT_PATH` | The Preferences choice if set, else the user copy if present, else the bundled default | Resolved at startup, and again when Preferences are saved |
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
The region model itself lives in `flimkit/utils/roi.py` so that headless and web use can reach it without Tkinter. `roi_tools` re-exports `RoiManager` and the patch helpers, so existing imports keep working.

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

#### `roi.py`
- **`RoiManager`** - stores region geometry and per-region statistics. Serialises to/from JSON for `.roi_session.npz` persistence. No GUI toolkit, so headless scripts and either frontend can use it.
  - `.add_region(name, tool, coords)` - register a new region, returns its integer ID
  - `.compute_region_mask(region_id, image_shape)` - boolean (H×W) mask for a region
  - `.to_json()` / `.from_json(json_str)` - serialise/deserialise for session files
  - `.to_geojson(region_ids=None)` / `.add_geojson(payload, mode='append')` - GeoJSON export and import, with self-intersecting polygon repair
- **`get_rectangle_patch(...)`** / **`get_ellipse_patch(...)`** / **`get_polygon_patch(...)`** - matplotlib patches from region coordinates

#### `display.py`
- **`load_zstack_display_slices(group_dir, ptu_dir=None, region=None)`** - per-slice pixel maps, decays and reference fit for a z-stack output directory
- **`compute_weighted_lifetime(pixel_maps, intensity, n_exp=2, weighting='amplitude')`** - amplitude- or intensity-weighted mean lifetime map
- **`apply_color_scale(image, vmin=None, vmax=None, gamma=1.0, percentile_auto=(2, 98))`** - normalise a map for display, with percentile autoscaling
- **`get_colormap(name='viridis')`** - colormap lookup against the `COLORMAPS` table
- **`compute_region_stats(lifetime_map, intensity_map, region_mask, full_stats=False)`** - τ and photon statistics inside a mask
- **`mask_to_rgba(mask, color=(1.0, 1.0, 1.0), alpha=0.3)`** - boolean mask as an RGBA overlay

#### `config_snapshot.py`
- **`_C()`** - cached dict snapshot of `configs.py`, read by both frontends when building fit arguments

#### `session.py`
- **`_reconstruct_dict_from_session(session_data, key)`** - rebuild a nested dict from the flattened `*_json` and `*_arr_*` keys in a session NPZ
- **`_safe_array_from_json(value)`** - coerce a session value back to a NumPy array
- **`_parse_summary(captured_log)`** - `name = value unit` lines from captured fit output as table rows

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
│   │   ├── roi_tools.py           # ROI drawing panel, per-ROI decay fitting
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
│       ├── roi.py                 # RoiManager - region geometry, masks, GeoJSON
│       ├── display.py             # Display scaling, colormaps, region stats
│       ├── config_snapshot.py     # Cached snapshot of configs.py
│       ├── session.py             # Session NPZ dict/array helpers
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

Machine IRFs are stored in `~/.flimkit/machine_irf/` (created automatically). The app ships with a bundled default until you build your own. After saving a new machine IRF, restart the app, or point File > Preferences... > Files > Default machine IRF (.npy) at it, which takes effect at once.

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

Two limits on that route. The wheel has to be pure Python, tagged `py3-none-any`; one built for a specific platform cannot be imported from a zip and is refused with that reason rather than failing later. And its dependencies have to be ones FLIMKit already bundles, because there is nothing in the compiled app to fetch the rest with. `numpy`, `scipy`, `matplotlib`, `pandas`, `tifffile`, `zarr` and tkinter are there.

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

Two halves, one on each side.

1. `pip install flimkit-bridge` into the environment FLIMKit runs in. That also gives you the `flimkit-bridge` command, which is how you run the bridge without the desktop app. The wheel is attached to the release as well, for offline installs or for `~/.flimkit/plugins/`.
2. Drop `qupath-extension-flimkit-bridge-*.jar` into QuPath's extensions directory, normally `~/QuPath/v0.7/extensions`.

`pip install flimkit-qupath-bridge` used to be step 1 and still works, but that package is being sunset. There is nothing QuPath-specific left in it: it is a shim that re-exports `flimkit-bridge`, warns on import, and is removed in 0.7.0. Install `flimkit-bridge` instead.

QuPath 0.7.0 or newer is required, and FLIMKit 0.13.0 or newer, which pip pulls in.

### Using it

The server itself is a separate package, [flimkit-bridge](https://github.com/FLIMKit/flimkit-bridge). The QuPath extension and the Fiji add-on are both clients of it, so there is one server, one port and one place a fix has to land.

The bridge starts with FLIMKit. There is nothing to launch and no port to configure. FLIMKit writes its address and a generated token to `~/.flimkit/qupath-bridge.json`, and QuPath reads that file, so `Extensions > FLIMKit bridge > Connect` needs nothing typed in. If port 8765 is busy an ephemeral one is used and recorded in the same file.

QuPath also pairs itself. Opening or dropping any file makes QuPath ask the bridge whether FLIMKit recognises it, so a `.ptu` opens through FLIMKit without connecting first. Pass `-Dflimkit.bridge.imageserver=false` to QuPath to turn that off.

From QuPath:

- **Add FLIMKit images to project** puts the intensity and lifetime maps into the open project as float32 images, in real units rather than a colourmapped render, so they can sit beside a brightfield or mIF image in the viewer grid.
- **Send annotations to FLIMKit** posts the annotations on the current image.
- **Fetch ROIs from FLIMKit** pulls FLIMKit's regions in.
- **Phasor** opens the phasor window for the image in view.

From FLIMKit, the **Send to QuPath** button in the ROI panel reports whether QuPath has connected and what is being served. If no QuPath has paired it says so rather than failing quietly.

### The phasor window

The phasor window draws the same density plot the desktop app does, for whichever time-domain file QuPath has open. Click on it to place an elliptical cursor and drag to move one. Each cursor reports its pixel count, phase and modulation lifetimes, and mean G and S, and six is the limit because that is how many colours the palette has.

**Draw region** switches the plot to tracing. Drag to draw an outline and it becomes a cursor when you let go, so a population that is not an ellipse can still be selected. The outline goes to FLIMKit as a polygon in G and S, points closer than three pixels apart are dropped, and fewer than three points is not a region. Drawn regions cannot be dragged afterwards. Remove and redraw instead. The button releases itself after each outline, so the plot goes back to placing ellipses.

**Create annotations** turns the cursors into QuPath annotations on the image, one per cursor, classified as `Phasor` and carrying the same measurements the cursor list shows. The phasor is computed on binned pixels, so the label image comes back at that resolution and is traced with the binning as its downsample, which puts the outlines in full-resolution image coordinates.

**Settings** picks the filter and the IRF.

| Setting | Values | What it does |
|---|---|---|
| Phasor filter | `none`, `gaussian`, `median`, `wavelet`, plus anything a plugin registers | Spatial smoothing of the G and S coordinates |
| Gaussian sigma | 0.1 to 10 px | Width of the gaussian kernel |
| Median window | 3 to 15 px | Size of the median window |
| IRF calibration | `none`, plus every machine IRF installed | Rotates and scales the phasor onto the calibrated frame |

Calibration runs before filtering, the same order the desktop app and `phasor_cli.py` use, so the same file gives the same coordinates whichever front end you drive it from.

The IRF list is empty on a fresh install. A machine IRF describes one microscope, so none are distributed with FLIMKit, and you have to measure your own under **Tools > Machine IRF Builder** in the desktop app before it appears here. An uncalibrated phasor is still useful for comparing populations within one image, but the absolute lifetimes it reports are not.

Changing a setting recomputes the phasor rather than reusing what was already on screen. The bridge caches per dataset and per setting, so switching back to a combination you have already looked at is immediate.

### Without the FLIMKit window

QuPath does not need the FLIMKit desktop app. `flimkit-bridge` starts the same server on its own, with no window and no display, which is what you want on a headless machine or when QuPath is the only front end you use.

```bash
pip install flimkit-bridge
flimkit-bridge
```

Note that `flimkit` is the desktop app and always opens a window. `flimkit-bridge` is the headless one.

It writes the same `~/.flimkit/bridge.json`, so `Extensions > FLIMKit bridge > Connect` finds it exactly as it finds a running FLIMKit. Extensions built before the server moved out read `qupath-bridge.json`, which is written alongside it until those versions are retired. `--port` and `--token` are there if you need them, `--force` takes over the discovery file from a bridge that has been left running, and `--no-announce` serves alongside one instead.

QuPath drives everything in this mode, stitching and fitting included, over the same HTTP API. `Extensions > FLIMKit bridge > Stitch and fit a mosaic...` takes the `.lif`, `.xlif` or `.xlef` that holds the tile positions and runs the whole pipeline on the FLIMKit side.

The **Pipeline** setting in that dialog picks how. `tile_fit`, the default, fits each tile after a global summed fit and assembles the maps. `stitch_fit` stitches the raw photons into one canvas and fits that, which means writing the whole photon cube to disk first: 124 tiles at 512 square make a 5581 square canvas, which is 57 GB at 459 bins. Pick `stitch_fit` when you want the stitched photon cube itself, and `tile_fit` otherwise.

### Co-registration

FLIMKit expects ROIs in FLIM image-pixel coordinates, so anything drawn on another image has to be transformed into that space first. That happens on the QuPath side.

This needs QuPath's [alignment extension](https://github.com/qupath/qupath-extension-align), which QuPath does not ship and which has to be installed separately. The bridge deliberately contains no alignment code of its own.

Align on the intensity image rather than the lifetime map. The alignment extension cannot render 32-bit float and throws rather than declining, and the lifetime map has to be float to carry nanoseconds. Photon counts are whole numbers, so intensity crosses as 16-bit whenever that is lossless, which the extension opens without complaint. The transform is valid for the lifetime map as well, since every image of that field shares one pixel grid, and intensity is the better registration target anyway because it carries the structure a lifetime map often lacks.

1. Open the brightfield or mIF image and add the FLIMKit images to the same project.
2. Align the brightfield against the intensity image and transfer the annotations onto it.
3. Send the annotations on the aligned image to FLIMKit.

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
| OME-Zarr | Lossless and compressed, one channel per exported map, OME-NGFF 0.4 |
| GeoJSON | ROI geometries and statistics, imports directly into QuPath |
| CSV | Fit summaries and per-ROI statistics |
| NPZ | Session files (fitting results, phasor arrays, cursor state) for session restoration |
| NPY | Raw FLIM histogram cubes and assembled lifetime maps |
| TXT | Human-readable fit summaries |

### OME-Zarr export

Export Images offers OME-Zarr alongside PNG and OME-TIFF. It writes one store
holding a single `(c, y, x)` float32 array, with each image you ticked as a
named channel, so intensity and lifetime end up as two channels of the same
image rather than two files. Channel names, the pixel size and the summed-fit
results are written into the store metadata: the fit summary sits under the
`flimkit` key in `.zattrs`, so the numbers travel with the image.

The store is named after the scan, `R146_FOV1.ome.zarr` for `R146_FOV1.ptu`,
falling back to `results.ome.zarr` when there is no input path to take a name
from. Exporting several fields of view into one folder therefore leaves one
store per field rather than overwriting.

The layout is OME-NGFF 0.4 on a Zarr v2 store, which is the version the
readers in Fiji, napari and QuPath are written against. The store validates
against the `ome-zarr-models` 0.4 schema and reads back through the reference
`ome-zarr` reader with its channel names and axes intact; I have not opened one
in Fiji, napari or QuPath myself. Chunks are compressed with Blosc/zstd, which
is where the size saving comes from: on a 512 square float32 intensity and
lifetime pair the store is about a third of the two uncompressed OME-TIFFs.

Zarr is a base dependency, so nothing extra needs installing and the compiled
application carries it. If it is somehow missing the export dialog says so and
does nothing rather than writing a partial store.

---

## Troubleshooting

**Gatekeeper blocks the compiled app on macOS**  
Right-click → Open on first launch. After that it should run normally.

**Machine IRF not found after saving**  
Restart the app, since a newly built default is picked up at startup, or set it in File > Preferences... > Files, which applies at once.

**Per-pixel fitting is very slow**  
That's expected for large FOVs on CPU. Try increasing `--binning` to aggregate pixels before fitting, or switch to summed-only mode if you don't need spatial maps. If you have a supported GPU (Apple Silicon, NVIDIA, AMD) and ran `python install.py`, GPU acceleration is detected and used automatically, no extra flags needed. `--free-tau-perpixel` with n_exp ≥ 2 is the exception: the backend prepares the batch and then runs SciPy per pixel on the CPU, so a GPU buys almost nothing there. Measured on an RTX A2000, 436.7s against 460.7s for 16,384 pixels, where the fixed-tau kernel is 9x and the distribution scan 22x.

The per-pixel GPU fit works in blocks, and `FLIMKIT_GPU_BLOCK_BYTES` sets the budget for one block in bytes. The default is 32 MB on CUDA and ROCm and 256 MB on MLX, and either way it is clamped to half of the free device memory. The CUDA default came off an RTX A5000, where anything above 32 MB costs a flat 2x because the card has 6 MB of L2 and the fast region is where the basis and a block stay resident. It does not generalise: an RTX A2000, with less L2, shows no cliff at all and is about 10 per cent slower at 32 MB than at 256 MB. Every budget returns identical lifetimes, so this is speed only.

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

FLIMKit is designed, developed, and maintained by Alex Hunt. Anthropic's Claude AI was used as an assistant for parts of the GUI implementation, the add-on system, compiled app builds, code debugging, and Docker packaging; all scientific design, fitting/phasor methods, validation, and the overall architecture are the author's own work.

FLIMKit reads several instrument formats. PicoQuant `.ptu` and Becker & Hickl `.sdt` reading is delegated to Christoph Gohlke's `ptufile` and `sdtfile` libraries, the SimFCS (`.b&h`, `.bhz`, `.ref`, `.r64`) and ISS (`.ifli`, `.iss-tdflim`) formats to his `lfdfiles`, and all TIFF reading, including ImSpector FLIM TIFF and PhasorPy OME-TIFF, to his `tifffile`; thank you to Christoph Gohlke for maintaining them, and for PhasorPy, which FLIMKit uses as its phasor backbone. Photonscore `.photons` reading is delegated to `photonsfile`, which was written for FLIMKit and spun out as a standalone library so it can be used without FLIMKit. Per-format provenance is in each reader's `NOTICE.md` (`flimkit/formats/<FORMAT>/NOTICE.md`).

---

## Contact

Alex Hunt - alexander.hunt@ed.ac.uk
