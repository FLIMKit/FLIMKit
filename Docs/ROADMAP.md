# FLIMKit Development Roadmap

**Last Updated:** August 8, 2026

## High Priority - To Do

### 1. Filter by Stats
UI controls to filter/select regions by tau, photon count, and other statistics. Display filtered results.

### 2. Stat Histograms
Histogram visualization for ROI statistical distributions (tau, photon counts, etc.).

## Medium Priority - To Do

### 1. Auto-Detect Regions
Automatic boundary detection for regions of interest based on intensity or lifetime gradients.

### 2. T2-mode TTTR decoding
Decode PicoQuant T2-mode records (used for FCS and point timing) alongside the current T3 modes, by deriving each photon's microtime from the recorded sync events. Blocked on a real T2 test file, ideally a matched T2/T3 pair of the same sample to validate against. Could later extend to older PicoQuant formats (`.pt3`/`.ht3`/`.phu`).

## Known issues

- **Importing ROIs from GeoJSON** - When importing ROIs from GeoJSON, only the exterior geometry is imported. Shapes with holes lose the hole geometry and are imported as the outer boundary only.
