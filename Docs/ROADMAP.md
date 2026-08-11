# FLIMKit Development Roadmap

**Last Updated:** August 10, 2026

## High Priority - To Do

### 1. Filter by Stats
UI controls to filter/select regions by tau, photon count, and other statistics. Display filtered results.

### 2. Stat Histograms
Histogram visualization for ROI statistical distributions (tau, photon counts, etc.).

### 3. Reading pixel data out of a Leica `.lif`
The metadata already comes out. Tile positions and pixel size are read from the XML header (`parse_lif_tile_positions`, `get_pixel_size_from_lif`), and those positions were checked bit-identical to the `.xlif` ones. The FLIM data is the part that is out of reach: it sits in a proprietary container inside the same file, so a scan saved only into a `.lif` still needs its `.ptu` files alongside it. Requested by a collaborator working from LAS X saved files.

Reading their XML header is one thing, decoding their data container is another, and I am not going to do the second without Leica documenting it and agreeing to it. Becker & Hickl, ISS and Photonscore all supported. After that the remaining unknown is the `_s{N}` series flattening order, which decides how a series index maps onto the tile grid.

### 4. Add-on system
Let users define their own decay models and cost functions, and let analysis tools, file formats and phasor filters register themselves the same way. Built in house as a small `flimkit.plugins` registry rather than by pulling in a plugin framework, since there is no packaging metadata for entry points to hang off and the PyInstaller build could not discover them anyway.

Models and cost functions are the goal, and they are one job rather than two. The differential evolution costs are twelve hand-written classes covering model against cost function against reparameterisation, so both hooks need the same groundwork first: a parameter-layout object, and one generic cost class in place of the twelve. Tracked in issue #3. Tools, formats and filters come first, since they are close to free by comparison and they prove the registry.

## Medium Priority - To Do

### 1. Auto-Detect Regions
Automatic boundary detection for regions of interest based on intensity or lifetime gradients.

### 2. T2-mode TTTR decoding
Decode PicoQuant T2-mode records (used for FCS and point timing) alongside the current T3 modes, by deriving each photon's microtime from the recorded sync events. Blocked on a real T2 test file, ideally a matched T2/T3 pair of the same sample to validate against. Could later extend to older PicoQuant formats (`.pt3`/`.ht3`/`.phu`).

### 3. Mosaic reconstruction without the XLEF
Rebuild a Leica tile mosaic from the tiles alone when the XLEF is missing, using row segmentation, global column-frame recovery and loop closure. Prototyped outside FLIMKit and verified against the XLIF positions on two dense tissue scans. It does not work on sparse or thin tissue, where the overlaps are too signal-starved to register, so it needs either the metadata or repeated line and frame structure to fall back on. Not yet ported into FLIMKit.

### 4. Panel GUI migration
Move the desktop GUI off tkinter and onto Panel. The spike passed and the early phases sit on `feature/panel-ui`, but nothing has landed on `main` yet, `flimkit/UI/gui.py` is still tkinter, and the remaining phases are parked until after the JOSS submission and the September workshop.

### 5. TVB grid-scan cost sensitivity
The per-pixel TVB grid scan selects a lifetime by taking the minimum over projected costs that sit very close together. TF32 rounding on CUDA was enough to move the selected bin by 0.06 to 0.13 ns, which is now prevented by forcing full float32 precision for those two matrix multiplications (#30, #34). That fixes CPU and GPU agreement but not the sensitivity itself. Worth reporting how flat the cost curve is near the minimum, or refining the grid around it, so a near-degenerate fit is visible rather than silent.

## Known issues

- **Importing ROIs from GeoJSON** - When importing ROIs from GeoJSON, only the exterior geometry is imported. Shapes with holes lose the hole geometry and are imported as the outer boundary only.
- **CI test depends on a network download** - `test_stitch_and_fit_workflow` builds a tissue mask, which makes Cellpose fetch model weights at run time. When the GitHub runner is rate limited the test fails with HTTP 429 and the whole run goes red for reasons unrelated to the change being tested. Caching the weights or stubbing the fetch would fix it.
