# FLIMKit

> **Warning:** Active development. Cross-validate results with other software before drawing conclusions. API and file formats may change without deprecation.

FLIMKit is a Python toolkit for FLIM data from FLIM microscope systems (or any PTU-based system). Built as a drop-in for FLIM microscope software, with two workflows:

- **Reconvolution fitting**: mono/bi/tri-exponential lifetime fitting with full IRF deconvolution, per-pixel and summed modes, multi-tile ROI stitching, batch processing, and optional time-varying background correction (FLIMfit-style, from a measured fluorophore-free reference)
- **Lifetime distribution fitting**: Gaussian and Lorentzian continuous α(τ) distributions (Lakowicz §4.11.2), per-ROI and per-pixel maps with GPU acceleration
- **Phasor analysis**: calibrated phasor plots, interactive elliptical cursors, spatial filtering (gaussian/median/wavelet), two-component decomposition, automatic peak detection, session save/load

Four entry points: desktop GUI, guided terminal UI, CLI scripts, Python API.

[Examples repo](https://github.com/alex1075/FLIMKit-Examples.git) | [Full documentation](documentation.md)

## Installation

Python ≥ 3.12 required (3.14 recommended — official builds use 3.14).

```bash
git clone https://github.com/alex1075/FLIMKit.git
cd FLIMKit
python install.py             # auto-detects GPU and installs the right backend
python validate_installation.py   # 10 checks - all should pass
```

For development work (PyInstaller + test dependencies):

```bash
python install.py --dev
```

Or download the compiled app from the Releases tab (no Python needed).

## Docker / TrueNAS SCALE

A pre-built image is available on Docker Hub. It runs the full desktop GUI in a browser via xpra (no installation needed on the client).

**Pull and run:**

```bash
docker run -d \
  -p 14500:14500 \
  -v /path/to/your/data:/data \
  --name flimkit \
  alex1075/flimkit:latest
```

Then open **http://localhost:14500** in your browser. PTU files and data should be placed in the folder you mount to `/data` - use the file dialog inside the app to navigate there.

**TrueNAS SCALE (Custom App):**

1. Apps → Discover Apps → Custom App
2. Paste the contents of `docker-compose.yaml` from this repo
3. Edit the volume paths to match your pool (e.g. `/mnt/tank/microscopy:/data`)
4. Deploy - TrueNAS will pull the image automatically

**Build from source** (required if you want to push your own changes):

```bash
python build_docker.py        # builds linux/amd64, pushes to Docker Hub
```

Requires Docker Desktop with buildx. On Apple Silicon, buildx cross-compiles for `linux/amd64` automatically.

## Usage

### Desktop GUI

```bash
python main.py
```

Five tabs: **Single FOV Fit**, **Tile Stitch / Fit**, **Batch ROI Fit**, **Machine IRF Builder**, **Phasor Analysis**. The right panel shows an FOV preview (intensity image + summed decay) and switches to the interactive phasor view when that tab is active.

### Terminal UI

```bash
python main.py --cli
```

### CLI

```bash
python fit_cli.py --ptu data.ptu --machine-irf machine_irf_default.npy --nexp 2
python phasor_cli.py --ptu data.ptu --irf irf.xlsx
```

### Python API

```python
from flimkit.phasor_launcher import launch_phasor
state = launch_phasor('data.ptu', irf_path='irf.xlsx')

# With spatial phasor filtering
state = launch_phasor('data.ptu', irf_path='irf.xlsx',
                      phasor_filter='gaussian', filter_kwargs={'sigma': 1.5})
```

## Machine IRF (do this first)

Before fitting, build a machine IRF for your system once and reuse it across sessions. You need matched `.ptu` + `.xlsx` pairs, 10-20 is a good number.

In the GUI, go to **Machine IRF Builder**, point it at your pairs folder, and save as `machine_irf_default`. From source this goes to `flimkit/machine_irf/`; compiled app saves to `~/.flimkit/machine_irf/`.

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

## GPU Acceleration

Per-pixel fitting uses a batched matrix solver that runs on GPU when a supported backend is detected. This applies to both single-FOV and tile-ROI pipelines. The same `fit_per_pixel()` function is used in both.

| Backend | Hardware | Notes |
|---|---|---|
| MLX | Apple Silicon (M1/M2/M3/M4) | Detected automatically |
| PyTorch MPS | Apple Silicon | Fallback if MLX not installed |
| PyTorch CUDA | NVIDIA | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| PyTorch ROCm | AMD | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |

`python install.py` detects your hardware and installs the right backend automatically. No extra flags needed at runtime.

**Limitations:** `--free-tau-perpixel` mode uses batched Adam on GPU when a backend is available (n_exp ≥ 2). `fit_summed` (single global fit) is always CPU - it's fast enough not to matter.

**Compiled app and GPU:** The compiled app bundles whatever GPU libraries are installed on the *build* machine. A binary built on Apple Silicon will have MLX/MPS; one built on a CUDA machine will have CUDA. If you need GPU in the compiled app, build it yourself on the target hardware. See [Compiled App](#compiled-app-macos--windows--linux).

## Tests

Not strictly necessary, but useful after making code changes.

```bash
python install.py --dev          # install test requirements first

cd flimkit_tests
python run_tests.py              # all tests
python run_tests.py -c           # with coverage report
python run_tests.py integration  # integration tests only
```

## Outputs

| Format | Description |
|---|---|
| PNG | Intensity and lifetime map images |
| OME-TIFF | Lossless export with metadata - opens in Fiji/ImageJ |
| GeoJSON | ROI geometries and stats - imports directly into QuPath |
| CSV | Fit summaries and per-ROI statistics |
| NPZ | Session files for restoring analysis state |

## Roadmap

Done: single FOV fitting, tile stitching, batch ROI processing, phasor analysis, GUI, session restoration, compiled app, ROI analysis with QuPath export.

Up next: config persistence, stat histograms, auto-region detection, batch n-exp in GUI. Chemical validation and publication pending.

See [ROADMAP.md](ROADMAP.md) for details.

## Notes on FLIM microscope software comparison

Fitted lifetimes from FLIMKit will typically read slightly higher than FLIM microscope software for the same data. This is a consequence of IRF placement: FLIMKit anchors the IRF at the steepest-rise point of the leading edge, which differs from how FLIM microscope software places the IRF. The difference is systematic and reproducible across acquisitions.

## References

**Lifetime distribution fitting** - Gaussian and Lorentzian α(τ) models:
> Lakowicz, J.R. (2006). *Principles of Fluorescence Spectroscopy* (3rd ed.). Springer. §4.11.2, pp. 141-144.

**Time-varying background correction** - measured background decay `B = V·b(t) + Z`:
> Görlitz, F. et al. (2017). Open Source High Content Analysis Utilizing Automated Fluorescence Lifetime Imaging Microscopy. *J. Vis. Exp.* (119), e55119. https://doi.org/10.3791/55119

**PhasorPy**: Gohlke, C. et al. Zenodo. https://doi.org/10.5281/zenodo.13862586

**Tile stitching**: Preibisch et al. (2009). *Bioinformatics* 25(11). https://doi.org/10.1093/bioinformatics/btp184

**Cellpose-SAM**: Pachitariu & Stringer (2025). *bioRxiv*. https://doi.org/10.1101/2025.04.28.651001

## Acknowledgements

FLIMKit is designed, developed, and maintained by Alex Hunt. Anthropic's Claude AI was used as an assistant for parts of the GUI implementation, compiled app builds, and Docker packaging; all scientific design, fitting/phasor methods, validation, and the overall architecture are the author's own work.

## Contact

Alex Hunt: alexander.hunt@ed.ac.uk
