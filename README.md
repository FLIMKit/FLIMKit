# FLIMKit

> **Warning:** Active development. Cross-validate results with other software before drawing conclusions. API and file formats may change without deprecation.

FLIMKit is a Python toolkit for FLIM data from Leica SP8/FALCON (or any PTU-based system). Built as a drop-in for Leica LAS X FLIM analysis, with two workflows:

- **Reconvolution fitting**: mono/bi/tri-exponential lifetime fitting with full IRF deconvolution, per-pixel and summed modes, multi-tile ROI stitching, and batch processing
- **Phasor analysis**: calibrated phasor plots, interactive elliptical cursors, two-component decomposition, automatic peak detection, session save/load

Four entry points: desktop GUI, guided terminal UI, CLI scripts, Python API.

[Examples repo](https://github.com/alex1075/FLIMKit-Examples.git) | [Full documentation](documentation.md)

## Installation

Python ≥ 3.12 required.

```bash
git clone https://github.com/alex1075/FLIMKit.git
cd FLIMKit
pip install -r requirements.txt
python validate_installation.py   # 9 checks — all should pass
```

Or download the compiled app from the Releases tab (no Python needed).

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
```

## Machine IRF (do this first)

Before fitting, build a machine IRF for your system once and reuse it across sessions. You need matched `.ptu` + `.xlsx` pairs, 10–20 is a good number.

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

`python install.py` detects your hardware and installs the right backend. GPU is used automatically, no extra flags needed.

**Limitations:** `--free-tau-perpixel` mode (LM solver) is CPU-only regardless of GPU availability. `fit_summed` (single global fit) is always CPU, it's fast enough not to matter.

**Compiled app and GPU:** The compiled app bundles whatever GPU libraries are installed on the *build* machine. A binary built on Apple Silicon will have MLX/MPS; one built on a CUDA machine will have CUDA. If you need GPU in the compiled app, build it yourself on the target hardware. See [Compiled App](#compiled-app-macos--windows--linux).

## Tests

Not strictly necessary, but useful after making code changes.

```bash
cd flimkit_tests
python run_tests.py              # all tests
python run_tests.py -c           # with coverage report
python run_tests.py integration  # integration tests only
```

## Outputs

| Format | Description |
|---|---|
| PNG | Intensity and lifetime map images |
| OME-TIFF | Lossless export with metadata — opens in Fiji/ImageJ |
| GeoJSON | ROI geometries and stats — imports directly into QuPath |
| CSV | Fit summaries and per-ROI statistics |
| NPZ | Session files for restoring analysis state |

## Roadmap

Done: single FOV fitting, tile stitching, batch ROI processing, phasor analysis, GUI, session restoration, compiled app, ROI analysis with QuPath export.

Up next: config persistence, stat histograms, auto-region detection, batch n-exp in GUI. Chemical validation and publication pending.

See [ROADMAP.md](ROADMAP.md) for details.

## Contact

Alex Hunt: alexander.hunt@ed.ac.uk
