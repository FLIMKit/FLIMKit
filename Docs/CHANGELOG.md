# Changelog

## [0.11.0] - 2026-08-15

### Added
- Plugins can run code at startup. `@startup(id, order)` registers a callback that runs with the app once the window is built, so an add-on that has to be doing something from launch, such as a server, no longer has to wait for someone to pick a menu item. A startup that raises is reported on the console and the rest still run, so one broken add-on cannot stop FLIMKit opening. Callbacks run on the UI thread, so anything long-lived belongs on a thread the callback starts and returns from.
- Plugins can add buttons to the ROI panel. `@panel_button(id, label, panel, order)` puts a button in the panel's action grid, for actions that belong beside the ROI controls rather than in a menu. `panel` accepts `roi`, and an unknown panel is refused at registration rather than ignored. The callback receives the same app object a `@tool` callback does.
- Bridges to QuPath and Fiji, documented under QuPath and Fiji Bridges. FLIMKit serves its intensity and lifetime images and its ROIs over a loopback HTTP connection, and the other program reads and writes them directly instead of going through exported files. The QuPath bridge runs inside a live QuPath session and can put the FLIMKit images into the open project, so they sit beside a brightfield image in the viewer.

### Fixed
- Freehand ROIs whose outline crosses itself no longer export as self-intersecting GeoJSON polygons. RFC 7946 does not allow a ring to cross itself, so anything backed by JTS or GEOS refused them; QuPath rejected an entire FeatureCollection because one region in it was invalid. Such a ring is now repaired to its outer boundary and the feature carries `repaired: self-intersecting`. Rings that were already valid are exported unchanged. Three of five hand-drawn regions in a real session were affected. This needed a geometry library, so `shapely` is a new dependency.

## [0.10.1] - 2026-08-14

### Added
- Public plugin bindings for fitted intensity and lifetime images and for GeoJSON ROI import and export. `get_current_images` returns 2D image copies with `photons` and `ns` units in separate namespaces; raw per-pixel decay histograms remain outside this binding. `export_rois_geojson` and `import_rois_geojson` hide private GUI fields, refresh ROI views after import and safely marshal background plugin calls to the GUI thread.
- Add-ons can be installed into the compiled app as a wheel. A `.whl` or `.zip` dropped into `~/.flimkit/plugins`, or named in `plugins.paths` or `FLIMKIT_PLUGIN_PATH`, goes on the import path before the entry point scan, so a packaged add-on registers exactly as a `pip install` would. That is the only route into the frozen builds, which have no `pip`. The wheel has to be pure Python, tagged `py3-none-any`, and its dependencies have to be ones FLIMKit already bundles; a platform wheel is refused with that reason rather than failing later inside an import.
- Free-tau per-pixel fitting honours the fit window on the GPU path as well, by restricting the residual and the degrees of freedom to the window. That path runs `least_squares` per pixel across a thread pool rather than on the GPU proper, so it was a slice rather than a kernel change.
- The GPU per-pixel path honours a fit window. Fixed-tau and the one-exponential grid scan now take the window and exclusion bands, so dropping a reflection peak out of the fit no longer forces the CPU path, which is the case where the GPU was wanted most. `intensity` and the `min_photons` mask are still computed over the whole decay, matching the CPU, and only the projection and chi-squared see the window. CPU/GPU parity is tested with a window active on both paths. Free-tau, one-exponential with a time-varying background, and tail fits still fall back and say which one applies.
- In-app explanations of the fit settings. An information icon on the Fitting Parameters, IRF and Masking section headers, and next to the optimizer choice in Expert Fit Settings, opens a single window covering six topics: fit model, component count, fitting mode, optimizer, IRF source and masking. Every option in those sections gets a paragraph on what it does and when to pick it, along with the caveats that are easy to miss: that a lower chi-squared alone does not justify an extra component, that a tau near the IRF width is unresolved rather than measured, and that the Coates correction leaves the decay non-Poisson while the default cost function still assumes it is. The text lives in `flimkit/UI/fit_help.py` and the icons sit inside label rows that already existed, so no panel grew.

### Fixed
- GeoJSON ROI round trips now preserve rectangle, ellipse, polygon and freehand geometry, names, fractional coordinates, colours and supported statistics. Ellipses export as polygon approximations with exact bounds, plain external polygons import as polygon ROIs, and invalid replacement payloads no longer clear existing regions.
- One-component free-tau per-pixel fitting gave different answers depending on whether a GPU backend was selected: the CPU path has always used the 200-point lifetime grid scan for a single exponential, while the backend path used the SciPy solver, so the two were running different algorithms on the same data. Both now use the grid scan and agree exactly. Found and fixed by Zhen Yuan (#43, #45).
- Per-pixel Gaussian and Lorentzian distribution fits ignored the fit window and exclusion bands. The summed distribution fit honoured them, so an excluded reflection peak was dropped from the global fit and left in every per-pixel map. The window now reaches the per-pixel path on CPU, Torch and MLX; background estimation, `intensity` and the `min_photons` mask still use the whole decay. Found and fixed by Zhen Yuan (#44, #46).
- `fit_per_pixel_dist` accepted `use_gpu` and never read it, so `use_gpu=False` was ignored and a backend was used anyway. Fixed alongside the distribution window work.

## [0.10.0] - 2026-08-11

### Added
- Add-on system (`flimkit/plugins/`). Analysis tools, file formats, format sniffers and phasor filters register themselves through a registry instead of being hard-coded, and FLIMKit's own tools go through the same registry a third party would use: the five Tools menu entries are now registrations in `flimkit/plugins/builtin/core_tools.py`, and the menu builds itself from what is registered. A plugin is a Python file with a decorated function (`@tool`, `@file_format`, `@format_sniffer`, `@phasor_filter`), and `examples/plugins/hello_tool.py` is a working one.
- Plugins load from four places, in order: the built-ins, installed packages declaring a `flimkit.plugins` entry point, `~/.flimkit/plugins`, then folders named in `plugins.paths` or `FLIMKIT_PLUGIN_PATH`. The home folder is never created by FLIMKit and nothing in it loads until `plugins.allow_user_plugins` is set, in `File > Preferences > Plugins` or the config file. `FLIMKIT_NO_PLUGINS=1` and `python main.py --no-plugins` skip everything, built-ins included, which gives a reproducible baseline for a published analysis.
- Loading is isolated per plugin: one that raises on import has its registrations rolled back, the traceback kept, and the rest still load. A plugin calling `sys.exit()` cannot take the app down. `Help > Plugins...` reports what loaded, what failed and why, and has a checkbox per plugin written to `plugins.disabled`.
- Built-in formats and phasor filters cannot be displaced. Registering `.ptu` does not take `.ptu` off the PicoQuant reader, and the `gaussian`, `median` and `wavelet` filters stay as they are. A plugin gets its own `plugin:<name>` config section and cannot reach the `expert` or `preferences` settings through it.
- The v1 hook signatures are frozen. `flimkit_tests/tests/fixtures/plugin_api_v1/` holds a plugin written against them that the suite loads on every run, so a change that would break an installed plugin fails the build. New hooks get added; the existing ones keep their arguments and their meaning.
- `pyproject.toml`, so `importlib.metadata` has something to read and an installed plugin can announce itself. Dependencies still come from `requirements.txt` and the version from `flimkit/_version.py`, so `install.py`, the Dockerfiles and CI are unchanged.

- Calibrated goodness of fit. `calibrated_chi2_pearson`, `calibrated_chi2_tail_pearson` and a per-pixel `calibrated_chi2` map divide the residual sum by its expected contribution under a fixed Poisson model, so the expectation is one rather than the one-count floor that keeps the existing `reduced_chi2_pearson` below it on sparse decays. The old fields stay for comparison with historical Leica LAS X numbers. Contributed by Zhen Yuan.

### Fixed
- Torch matrix products in the time-varying-background path were losing precision on CUDA, where TF32 is the default for `float32` matmul. They now run at full precision under a lock that restores the previous setting, so the CUDA result matches CPU and MLX. Contributed by Zhen Yuan.
- The GPU backend cache is reset between TVB parity tests, which were leaking a backend from one test into the next.

## [0.9.19] - 2026-08-07

### Added
- Multidimensional series stitching: a fourth Stitch pipeline, "Multidimensional series", for tiled acquisitions with a z and/or time axis. It walks every (timepoint, z) plane, stitches the overlapping tiles for that plane into one canvas and fits it, rather than treating overlapping positions as independent fields of view (issue #21). One decay is pooled across the whole series and the lifetimes are held fixed for every plane, so amplitudes stay comparable between timepoints; `pool_stride` subsamples that pooling since it only needs photon statistics. Output is one directory per (t, z) plus a `*_series_index.json` manifest, and the FOV preview gains a plane slider that loads each from disk. Filenames follow the same `region_tX[_sY][_zZ].ptu` convention as the timelapse and z-stack batch fits, sharing `parse_timelapse_filename` with them. Verified on a 231 timepoint, 2 tile, 3 z-plane set (1386 files).
- Tile positions from an XLIF or LIF are refined against the tile overlap before stitching. On the test set the nominal stage coordinates put both tiles at the same y when the images show a 23 px offset, which drops the overlap correlation from 0.77 to 0.04, so refinement is not optional. With no metadata the positions are recovered from the overlap alone, which needs structure in the overlap and is unreliable on sparse samples.

### Fixed
- The multistart optimiser was searching unscaled parameters, so lifetimes in seconds and amplitudes near one were treated as comparable step sizes. Contributed by Zhen Yuan.

## [0.9.18] - 2026-07-22

### Added
- Tail fit: bare exponentials fitted past the decay peak with no IRF and no reconvolution, `y(t) = sum A[i] exp(-(t-t0)/tau[i]) + Bkgr`. Reports the Leica quantities (per-component intensities `I[k] = A[k]tau[k]`, `I_sum`, `A_sum`, `t0`) alongside the usual amplitude- and intensity-weighted mean lifetimes. `t0` is pinned to the decay peak by default because it is strongly correlated with the amplitudes; free it from expert settings or `--fit-t0`. Available in FOV, batch and stitch modes, and as `--fit-model tail` on the CLI.
- Measured-IRF alignment. A scatter PTU or `.pck` taken in a separate acquisition carries a different sync delay, so its peak can sit hundreds of bins from the decay rising edge, which the fit could not close with its two-bin shift. The offset is now measured and applied, so the fit no longer comes back quietly wrong.
- Tile positions can come from a `.lif` as well as an `.xlif`, proven bit-identical on a 90 tile mosaic.
- LAS X CSV and tab-delimited IRF exports are accepted, including files with a ragged preamble. Contributed by Zhen Yuan.
- `SECURITY.md`, a pull request template, and contributor credits in the acknowledgements.

### Changed
- tkinter is decoupled from the core modules, so the fitting, format and phasor code imports on a headless machine.

## [0.9.17] - 2026-07-17

### Added
- User-settable fit window and exclusion bands (`--fit-start-ns`, `--fit-end-ns`, `--exclude-ns`), so a reflection peak mid-decay can be dropped from the fit. The per-pixel path now honours the window at all: it previously projected over every bin, IRF rise and artefacts included.
- Pile-up can go in the forward model (`n_sync=`) instead of correcting the data, which keeps the fit Poisson.
- Z-stack analysis mode: fit an axial stack of `region_zX.ptu` slices (one PTU per z-slice, as exported in Leica `.sptw` workspaces) as a single field of view. All slices are pooled to fit one shared reference lifetime, then each slice gets per-pixel amplitude/intensity maps with τ locked, so lifetime is held constant through the stack while composition can vary with depth. Writes per-slice maps and lifetime images, `(Z, H, W)` stacks, and a z-series CSV/JSON/plot. In the GUI it is a toggle next to the input file in the Single FOV tab, with a z-slider in the FOV preview; also in the terminal UI and via `zstack_flim_fit` on the CLI. Mirrors the timelapse pipeline with z as the stack axis instead of time.
- ISS `.ifli` files route straight to phasor analysis with fitting disabled, since frequency-domain data has no decay to fit.
- `ptufile`, `sdtfile` and `lfdfiles` dependencies (all ship wheels; `sdtfile` is pure-Python), also added to the PyInstaller hidden imports and `validate_installation.py`.
- A test workflow in CI, and Docker images built and published from CI.

### Changed
- PicoQuant `.ptu` and Becker & Hickl `.sdt` reading now delegate to Christoph Gohlke's maintained, cited `ptufile` and `sdtfile` libraries. Before switching, both were validated against FLIMKit's own decoders on real data: `.ptu` matched `ptufile` across 32 files (Leica, PicoQuant, Chroma, Zeiss; image and point-mode) and `.sdt` cubes were bit-identical to `sdtfile`. This puts format correctness on established libraries and, for `.ptu`, is faster and handles large multi-frame files that were slow or memory-heavy before (a Zeiss 1442-frame file that previously took ~18 min / ~5 GB now decodes in ~3 s). Same `(Y, X, H)` cube and metadata, so the fitter, phasor, stitching and GUI are unchanged.
- FLIMKit's original hand-rolled `.ptu` and `.sdt` decoders are preserved as an independent reference and cross-check in the separate `flim-native-decoders` repository, with the comparison scripts that reproduce the match.
- `.pck` Check / IRF files are still read by FLIMKit's own `read_pck` (`ptufile` exposes only their tags). ISS keeps its own reader, since no published library covers it.
- Photonscore `.photons` reading moved to the `photonsfile` package, spun out of FLIMKit and published to PyPI.
- Format detection and the supported-format table moved into `flimkit/formats/flim_file.py`; documentation moved into `Docs/`; Docker files moved into `docker/`.

### Fixed
- Drag and drop, and the name auto-population when loading a single FOV file outside a project.

## [0.9.16] - 2026-07-14

### Fixed
- Pile-up correction was being fed the photon count instead of the excitation-pulse count. Coates needs `N_sync`, the number of pulses the histogram accumulated over, and every call site was passing the TTTR record count; in the B&H and Photonscore readers that value was literally the total photon count. On a real PTU the true rate is 0.00069 photons per pulse and FLIMKit reported 97.4% pile-up and advised correcting it. In the per-pixel path the per-pixel figure collapsed to one pulse, which zeroed the decay outright. `N_sync` now comes from the real pulse count per format, and Coates refuses any count that cannot be one.

## [0.9.15] - 2026-07-03

### Added
- ISS file format support (branch `feature/iss-format-support`, version `0.9.15.dev0+iss`): new `flimkit/ISS/` package mirroring `flimkit/PTU/` to read ISS lifetime data. Primary target is the time-domain triplet (`.TAGTIME`/`.TAGCHANNEL`/`.TAGDECAY`) returning the same `(Y, X, H)` decay cube + metadata as the PTU reader; secondary is the frequency-domain phasor `.ifli`. Planning stage, blocked on real ISS sample files. Tracked in issue #19. Specs provided by Jeff Liao (ISS).
- Becker & Hickl `.sdt` format support (branch `feature/bh-format-support`, version `0.9.15.dev0+bh`): new `flimkit/formats/BH/` package reading B&H SPC TCSPC FLIM histograms. The decoder (`decode.py`) is original FLIMKit code written from B&H's official SPCM file-structure documentation; `reader.py` returns the same `(Y, X, H)` decay cube + metadata as the PTU/ISS readers and is wired through `FLIMFile`, so the fitter, intensity images and FLIM images work unchanged. Validated against real B&H samples (SPC-150NX / SPC-150N / SPC-180NX FIFO-image files); summed and per-pixel fits run end to end. Adds `lz4` for LZ4-compressed blocks. Thank you to Becker & Hickl (Dr. Jens Balke and Enzo Marscheck) for the format documentation and sample data.
- Photonscore `.photons` format support (`flimkit/formats/PS/`, branch `feature/new-file-formats`): pure-Python reader for Photonscore LINCam files (the D7 container), decoded with no native dependency and validated bit-exact against the Photonscore SDK on a 284-million-photon sample. Position-sensitive detector, so the image is formed by binning each photon's (x, y) position and the decay by histogramming the TCSPC micro-time; `dt` to time from the file's `TacChannel` attribute. Optional numba acceleration for the varint decode with a numpy fallback. Backed by the now-public D7 specification (github.com/photonscore/d7, Apache-2.0). Wired through `FLIMFile` and the GUI file pickers / folder scans, so the fitter, intensity images and FLIM images work unchanged. Thank you to Photonscore for the SDK, a sample file, and for open-sourcing the format. Tracked in issue #20.
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
