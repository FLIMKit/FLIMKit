# Changelog

## [Unreleased]

### In development
- ISS file format support (branch `feature/iss-format-support`, version `0.9.15.dev0+iss`): new `flimkit/ISS/` package mirroring `flimkit/PTU/` to read ISS lifetime data. Primary target is the time-domain triplet (`.TAGTIME`/`.TAGCHANNEL`/`.TAGDECAY`) returning the same `(Y, X, H)` decay cube + metadata as the PTU reader; secondary is the frequency-domain phasor `.ifli`. Planning stage, blocked on real ISS sample files. Tracked in issue #19. Specs provided by Jeff Liao (ISS) via Anand Yethiraj (Guelph).

### Added
- Time-varying background correction (FLIMfit-style `B = V·b(t) + Z`): supply a measured fluorophore-free reference PTU and the fit removes a scaled, time-varying background instead of only a flat offset
- Available across summed and per-pixel fits (discrete fixed-τ / free-τ / 1-exp grid scan, and unimodal distribution), on CPU, PyTorch and MLX, with a non-negative per-pixel `tvb_scale` map
- Exposed via CLI (`--tvb-ptu` / `--tvb-channel`), all GUI fit tabs (Single-FOV, Batch-FOV, Tile-Stitch, Batch-ROI; the tile pipeline aligns the background per tile), and the Python API (`tvb_profile=` / `fit_tvb=`)
- Per-pixel scale exported as `*_tvb_scale.tif`; summed-fit summary gains a TVB-scale line
- 28 regression tests (`test_tvb.py`) covering loader, model, bounds, cost classes, summed/per-pixel recovery, CPU/GPU parity, and exports

### Notes
- Anthropic's Claude AI assisted with parts of the implementation. All scientific design and method selection are the author's own work.

---

## [0.9.13] - 2026-06-09

### Added
- Gaussian and Lorentzian lifetime distribution fitting (Lakowicz §4.11.2): continuous α(τ) distributions as an alternative to the discrete multi-exponential model
- Per-ROI distribution fitting via Differential Evolution + Levenberg-Marquardt with full IRF reconvolution
- Per-pixel distribution maps: 2D GPU grid scan for unimodal fits (MLX/Torch); scipy parallel LM fallback for bimodal
- Expert settings toggle: Discrete / Gaussian distribution / Lorentzian distribution, with 1 or 2 components
- Distribution-specific results display: τ̄_amp, τ̄_int, σ/Γ, FWHM, amplitude fractions per component
- 11 new tests covering kernel properties, basis grid construction, τ̄ recovery, and per-pixel map shape

### Notes
- Anthropic's Claude AI assisted with parts of the implementation. All scientific design and method selection are the author's own work.

---

## [0.9.12] - 2026-05-20

### Changed
- Switched cell segmentation masking from Cellpose to Cellpose-SAM

---

## [0.9.11] - 2026-05-10

### Changed
- Updated PhasorPy calls to match API changes in v0.11
- Added phasor spatial filtering options (Gaussian, median, wavelet) to GUI and API

---

## [0.9.10] - 2026-05-05

### Fixed
- GPU fitter tests and CI pipeline

---

## [0.9.9] - 2026-05-01

### Fixed
- Fixed bugs in image tools, interactive module, and test suite introduced in 0.8.9
- Improved test coverage and mock data for decode, ground truth, and integration test cases

---

## [0.9.x] - 2026-04-07 to present

### Added
- Phasor panel embedded in the GUI: live phasor histogram and intensity image update as cursors are placed, and FRET
- ROI analysis panel for placing and measuring regions directly on fitted FLIM images
- Session restoration for stitched ROIs (FOV preview and summed fit now restore correctly)
- Undo/redo system with menu and button states (Ctrl+Z / Ctrl+Shift+Z)
- Project tree view in left sidebar browser for multi-PTU scans
- Four ROI drawing tools: rectangle, ellipse, polygon, freehand
- QuPath GeoJSON export and import for ROI round-tripping
- Per-region lifetime and photon count statistics with CSV export
- Progress bars throughout fitting and stitching pipelines
- Auto-save to NPZ for all session types
- Keyboard shortcuts: undo/redo, zoom, menu accelerators
- Better error messages across the codebase

### Changed
- Refactored `gui.py` with improved layout, tab structure, and controls
- PTU reader updated and optimised
- `lifetime_image.py` updated with improved per-pixel lifetime image generation
- `assemble.py` streamlined for efficiency
- Stitched FLIM image export pixel size bug fixed (was saving larger than actual pixel size)

### Notes
- Anthropic's Claude AI assisted with parts of the GUI implementation. All scientific design, fitting/phasor methods, validation, and overall architecture are the author's own work.

---

## [0.8.8] - 2026-04-06

### Added
- CI/CD build workflow for macOS, Linux, and Windows via GitHub Actions
- Machine IRF creation from existing fitted data (`irf_tools.py`)
- Chunking support for memory-limited machines during per-pixel fitting
- Batch ROI processing pipeline

### Fixed
- Ubuntu build: added tkinter and libGL dependencies, hidden-import fix
- GUI windowing issues on multi-monitor setups
- ROI fitting algorithm accuracy

### Changed
- Switched to `--onedir` PyInstaller builds for a proper `.app` bundle (fixes the two-dock-icon issue)
- Standardised on Python 3.12 (attempts at 3.13 and 3.14 dropped)
- Tile stitching algorithm overhauled. Removed stitch line artefacts, added three-pass phase-correlation registration
- Removed deprecated `stitch.py` and `fix_missing_tiles.py`, replaced with integrated modules
- Machine IRF now supported as input to phasor analysis

---

## [0.6.x] - 2026-03-13

### Added
- Phasor analysis: interactive cursor tool, automatic peak detection, two-component decomposition, batch processing, session save/load
- `phasor_launcher.py` and `phasor_cli.py`
- `phasor/signal.py` for phasor computation helpers
- Intensity thresholding for fitting and image export

### Fixed
- Corrected phasor image region visualisation
- Various fitter and test fixes

### Changed
- Poisson cost function now default (better statistics for low-count bins)
- `configs.py` extended for machine IRF configuration
- Requirements updated with phasor dependencies

---

## [0.3.x] - 2026-03-03 to 2026-03-12

### Added
- Versioning system (`_version.py`)
- Test suite infrastructure
- Raw photon reader for image stitching
- IRF reconstruction workflow (`irf_reconstruction_validation.ipynb`)

### Fixed
- Missing source lines, CLI call issues, shebang paths

### Changed
- Renamed package from `pyflim` to `flimkit`
- Re-added stitching with optimisations

---

## [Initial Commit] - 2026-02-27

First commit - basic FLIM fitting, CLI, IRF estimation from XLSX, user prompts and defaults.
