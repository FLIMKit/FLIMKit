# FLIMKit Development Roadmap

**Last Updated:** June 29, 2026

## In Progress

### ISS file format support
Read ISS (FastFLIM / Vista) data so FLIMKit works with the ISS lifetime platform alongside PicoQuant. New `flimkit/ISS/` package mirroring `flimkit/PTU/`, returning the same `(Y, X, H)` decay cube + metadata so the rest of the pipeline is unchanged. Primary target is the time-domain triplet (`.TAGTIME`/`.TAGCHANNEL`/`.TAGDECAY`); secondary is the frequency-domain phasor `.ifli` (feeds the phasor module). Being developed on branch `feature/iss-format-support` (version `0.9.15.dev0+iss`). Blocked on real ISS sample files for validation. Tracked in issue #19. Came from a CONFOCALMICROSCOPY listserv thread where Jeff Liao (ISS), who provided the format specs. Thank you both.

## High Priority - To Do

### 1. Config File Management
YAML/JSON settings that persist. UI panel to edit colors, default tools, output dirs. Per-project overrides + reset option. (Infrastructure exists; needs persistence layer)

### 2. Automatic Error Log Exporter
On crash: timestamped logs with system info, app version, and what you were doing. Export for debugging. (Basic logging exists; needs crash handler integration)

### 3. Filter by Stats
UI controls to filter/select regions by tau, photon count, and other statistics. Display filtered results.

### 4. Stat Histograms
Histogram visualization for ROI statistical distributions (tau, photon counts, etc.).

## Medium Priority - To Do

### 1. Auto-Detect Regions
Automatic boundary detection for regions of interest based on intensity or lifetime gradients.

### 2. T2-mode TTTR decoding
Decode PicoQuant T2-mode records (used for FCS and point timing) alongside the current T3 modes, by deriving each photon's microtime from the recorded sync events. Blocked on a real T2 test file, ideally a matched T2/T3 pair of the same sample to validate against. Could later extend to older PicoQuant formats (`.pt3`/`.ht3`/`.phu`).

### 3. Becker & Hickl format support
Read Becker & Hickl TCSPC FLIM data (SPCImage), broadening FLIMKit beyond PicoQuant and Leica to B&H instruments. The `.sdt` reader is now in development on branch `feature/bh-format-support` (version `0.9.15.dev0+bh`): a new `flimkit/formats/BH/` package decodes the SPC histogram/image blocks into the same `(Y, X, H)` cube + metadata as the PTU/ISS readers, wired through `FLIMFile`, and is validated against real B&H sample files (SPC-150NX/150N/180NX). Written from B&H's official SPCM file-structure documentation. Still to come: raw `.spc` FIFO photon-stream decoding (a record-level decoder per module family, like the PTU reader). Thank you to Becker & Hickl (Dr. Jens Balke and Enzo Marscheck) for the documentation and sample data.

## Completed 

**High Priority:**
- Undo/Redo System (Ctrl+Z & Ctrl+Shift+Z with menu & button states)
- Project Tree View (Left sidebar browser for multi-PTU scans)

**Medium Priority:**
- Batch FOV analysis
- Tested full IRF support (6 methods: FLIM microscope XLSX, Machine IRF, Scatter PTU, Estimate raw, Estimate parametric, Gaussian)
- Keyboard shortcuts (Undo/redo, zoom, menu accelerators)
- Better error messages (Extensive throughout codebase)

**Core Features:**
- Multi-format PTU decoding (PicoHarp, HydraHarp, TimeHarp 260, MultiHarp T3) and SymPhoTime `.pck` IRF import
- Region drawing (4 tools: rectangle, ellipse, polygon, freehand)
- Per-region stats (tau, photon counts)
- CSV/GeoJSON export-import
- Progress bars
- Auto-save to NPZ
- Multi-panel UI


## Known issues

- **Stitched FLIM image export** - FLIM images from stitched ROIs are currently saving with a larger pixel size than the original PTU. This is a known issue related to how the per pixel lifetime is calculated (binning). For now the workaround is to use batch ROI fit (under tools) to get true per pixel lifetime maps, which are exported correctly. This will be fixed in a future update. - NOW FIXED IN 0.9.4
- **Stitched session restoration** - When loading a saved session from a stitched ROI, only the FOV preview and summed fit will restore correctly. ROIs and fit settings don't restore as they should. This will be fixed in a future update. - NOW FIXED IN 0.9.4
- **Importing ROIs from GeoJSON** - When importing ROIs from GeoJSON, only the exterior geometry is imported. Any donught like shapes with holes will lose the hole geometry and be imported as the outer boundary only. 
