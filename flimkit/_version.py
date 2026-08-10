__version__ = '0.9.19+anisotropy.3'
fitter_version = '18'

roadmap = '''Flim program roadmap:
Version history:
0.1.0 -> Initial version with basic FLIM fitting functionality
0.1.1 -> Added command line interface (CLI) for single FOV fitting
0.1.2 -> Added support for IRF estimation from XLSX files
0.1.3 -> Added some user prompts and created defaults
0.1.4 -> Updated interactive prompts and added some error handling
Will be updated to 0.2.0 when ROI stitching works
0.3.0 -> ROI stiching + fitting
0.4.0 -> batch processing of multiple ROIs or FOVs
0.5.0 -> tbd
Need to add Phasor once added to the codebase, and update version to 1.0. And added some tests.
0.8.0 -> added per-tile machine IRF fitting workflow, updated docs and examples
0.8.1 -> added some error handling and user prompts to per-tile machine IRF workflow
0.8.2 -> added some more error handling and user prompts to per-tile machine IRF workflow, and updated example notebook
0.9.0 -> made flim fitted image visible in GUI
0.9.1 -> added session restoration in the GUI
0.9.2 -> added some error handling and begin session restoration for roi analysis panel in the GUI
0.9.3 -> complete session restoration for roi analysis panel
0.9.5 -> selectable ROIs in FLIM image in GUI and export of region stats
0.9.7 -> added FRET (using PhasorPy)
0.9.8 -> added pile-up correction function
0.9.9 -> UI fixes
0.9.10 -> fixed GPU tests
0.9.11 -> updated PhasorPy calls to match latest version (0.10.0) and added phasor filtering options
0.9.12 -> changed masking to use cellpose-SAM
0.9.13 -> added Gaussian and Lorentzian lifetime distribution fitting (Lakowicz §4.11.2)
0.9.14 -> bug fixes galore
0.9.15 -> new file format support: ISS (TD-FLIM .TAGTIME/.TAGCHANNEL/.TAGDECAY, FD-FLIM phasor .ifli; see issue #19), Becker & Hickl .sdt (SPC TCSPC FLIM, FIFO-image histograms; decoder written from B&H SPCM file-structure docs) and Photonscore .photons (LINCam D7 container; pure-Python, no native dependency, validated bit-exact vs the Photonscore SDK; see issue #20)
0.9.16 -> fixed pile-up correction: Coates was being fed the photon/record count instead of the excitation-pulse count (N_sync), which destroyed the decay tail and mis-reported pile-up (0.07% read as 97%). N_sync is now taken from the true sync count per format, and Coates refuses any count that cannot be one.
0.9.17 -> user-settable fit window + exclusion bands (--fit-start-ns/--fit-end-ns/--exclude-ns), so reflection peaks mid-decay can be dropped from the fit; the per-pixel path now honours the window at all (it previously projected over every bin, IRF rise and artefacts included). Pile-up can now go in the forward model (n_sync=) instead of correcting the data, which keeps the fit Poisson.
0.9.18 -> added the n-exponential tail fit: bare exponentials fitted past the decay peak with no IRF and no reconvolution, y(t) = sum A[i] exp(-(t-t0)/tau[i]) + Bkgr. Reports the Leica quantities (per-component intensities I[k] = A[k]tau[k], I_sum, A_sum, t0) alongside the usual amplitude- and intensity-weighted mean lifetimes. t0 is pinned to the decay peak by default because it is strongly correlated with the amplitudes; free it from expert settings or --fit-t0. Available in FOV, batch and stitch modes, and as --fit-model tail on the CLI.
0.9.19 -> multidimensional (z-stack and/or timelapse) tile stitching: a fourth Stitch pipeline that walks every (timepoint, z) plane, stitches the overlapping tiles for that plane and fits it, so overlapping positions are no longer treated as separate fields of view. One decay is pooled across the series and the lifetimes held fixed for every plane, so amplitudes stay comparable between timepoints. Tile positions come from the XLIF/LIF when one is given and are then refined against the tile overlap, which matters because the nominal stage coordinates can be out by tens of pixels; without metadata they are recovered from the overlap alone. Results browse with a plane slider in the FOV preview.
1.0.0 -> release version with all core features implemented and tested
1.5.0 (or 2.0) -> validation of fitting results with known fluorophores.'''