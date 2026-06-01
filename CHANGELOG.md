# Changelog

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
