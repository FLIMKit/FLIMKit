__version__ = '0.9.16'
fitter_version = '17'

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
1.0.0 -> release version with all core features implemented and tested
1.5.0 (or 2.0) -> validation of fitting results with known fluorophores.'''