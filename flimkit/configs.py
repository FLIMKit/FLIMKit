import sys
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

FLIM_CMAP = LinearSegmentedColormap.from_list(
    'flim', ['#000080','#0000ff','#00ffff','#00ff00','#ffff00','#ff0000']
)

MIN_PHOTONS_PERPIX = 10

# Intensity threshold for masking low-signal pixels before fitting.
# Set to None (disabled) or an integer photon count.
# Use --intensity-threshold on the CLI, or 'interactive' to pick visually.
INTENSITY_THRESHOLD = None


# General fitting settings:
Tau_min = 0.145
Tau_max = 45.0

# Set default fitting mode. Options are "summed", "perPixel", and "both". Override with --mode when running the code.
D_mode = 'both'

# Default number of exponentials to fit. Options are 1, 2, or 3. Override with --nexp when running the code. 4+ exponentials are not supported as the fitting becomes unstable and unreliable.
n_exp = 3

# Default binning factor for per-pixel fitting. Set to 1 for no binning. Override with --binning when running the code.
binning_factor = 1
# Default optimizer for per-pixel fitting. Options are "lm_multistart" and "de".
Optimizer = 'de'

# Levenberg-Marquardt settins:
lm_restarts = 8

# Default settings for DE optimizer:
de_population = 30
de_maxiter = 5000
n_workers = -1

# IRF settings:
IRF_FWHM = None
# ns window (centred on the decay peak) the parametric IRF is fit over. Keep it
# tight: a wide window pulls in the fluorescence falling edge, so the fit traces
# the decay instead of the instrument pulse and yields an over-broad IRF.
IRF_FIT_WIDTH = 0.5
IRF_BINS = 21
Estimate_IRF = 'none'

# Machine IRF defaults (spreadsheet-free workflow)

# True when launched as a PyInstaller-compiled executable.
_is_frozen = getattr(sys, 'frozen', False)

# Bundled read-only resources - always accessible via __file__ (even when frozen,
# PyInstaller extracts them to a temp dir that __file__ still points into).
_BUNDLED_MACHINE_IRF_DIR = Path(__file__).resolve().parent / 'machine_irf'

# Writable location for user-generated IRF files.
# When frozen the package dir is inside a read-only bundle, so saves go to
# ~/.flimkit/machine_irf/ instead.  When running from source the behaviour is
# unchanged (saves alongside the package).
USER_MACHINE_IRF_DIR = Path.home() / '.flimkit' / 'machine_irf'
MACHINE_IRF_DIR = USER_MACHINE_IRF_DIR if _is_frozen else _BUNDLED_MACHINE_IRF_DIR

# Default IRF path used when no explicit path is given.
# When frozen: prefer a user-saved copy (if it exists), else fall back to the
# bundled factory default so a fresh install still works out of the box.
_user_default = USER_MACHINE_IRF_DIR / 'machine_irf_default.npy'
MACHINE_IRF_DEFAULT_PATH = (
    (_user_default if _user_default.exists() else _BUNDLED_MACHINE_IRF_DIR / 'machine_irf_default.npy')
    if _is_frozen
    else _BUNDLED_MACHINE_IRF_DIR / 'machine_irf_default.npy'
)
MACHINE_IRF_ALIGN_ANCHOR = 'peak'
MACHINE_IRF_REDUCER = 'median'
MACHINE_IRF_FIT_STRATEGY = 'fixed'
MACHINE_IRF_FIT_BG = True
MACHINE_IRF_FIT_SIGMA = False
MACHINE_IRF_FIT_TAIL = False
MACHINE_IRF_SIGMA_MAX_FULL = 3.0
MACHINE_IRF_SIGMA_MAX_HALF = 0.5
MACHINE_IRF_DE_POPULATION = 30
MACHINE_IRF_DE_MAXITER = 5000

# Cost function for summed fit. Options are "poisson" (recommended) and "chi2" (legacy).
# "poisson" uses Poisson deviance (C-statistic) on raw counts - statistically correct.
# "chi2" normalises by peak and uses Neyman weights - underweights the tail.
Cost_function = 'poisson'

# Display range for exported tau images (FLIM microscope clipping).
# Out-of-range pixels are clipped to the nearest boundary, matching FLIM microscope software behaviour.
# Set to None to keep the full fitted range (no clipping).
TAU_DISPLAY_MIN = None
TAU_DISPLAY_MAX = None

# Display range for exported intensity images (same clipping behaviour).
# Set to None to keep the full range.
INTENSITY_DISPLAY_MIN = None
INTENSITY_DISPLAY_MAX = None

# Phasor-domain spatial filtering applied after calibration.
# Set to None to disable.  Options: None, 'gaussian', 'median', 'wavelet'.
PHASOR_FILTER = None
PHASOR_FILTER_SIGMA = 1.0
PHASOR_FILTER_SIZE = 3
PHASOR_FILTER_WAVELET = 'db4'
PHASOR_FILTER_LEVEL = 1

# Other specific settings:
channels = None
OUT_NAME = 'flim_out'


config_message = f"""Default settings:
Intensity threshold: {INTENSITY_THRESHOLD} photons (None = disabled)
Tau_min: {Tau_min} ns
Tau_max: {Tau_max} ns
Tau display min: {TAU_DISPLAY_MIN} ns (None = no clip)
Tau display max: {TAU_DISPLAY_MAX} ns (None = no clip)
Intensity display min: {INTENSITY_DISPLAY_MIN} (None = no clip)
Intensity display max: {INTENSITY_DISPLAY_MAX} (None = no clip)
Fitting mode: {D_mode}  
Number of exponentials: {n_exp}
Cost function: {Cost_function}
Optimizer: {Optimizer}
Levenberg-Marquardt restarts: {lm_restarts}
Differential Evolution population size: {de_population}
Differential Evolution max iterations: {de_maxiter}
Number of workers for DE optimization: {n_workers}
IRF FWHM: {IRF_FWHM} ns
IRF fit width: {IRF_FIT_WIDTH} ns
IRF bins: {IRF_BINS}
IRF estimation method: {Estimate_IRF}
Machine IRF default path: {MACHINE_IRF_DEFAULT_PATH}
Machine IRF align anchor: {MACHINE_IRF_ALIGN_ANCHOR}
Machine IRF reducer: {MACHINE_IRF_REDUCER}
Machine IRF fit strategy: {MACHINE_IRF_FIT_STRATEGY}
Machine IRF fit bg/sigma/tail: {MACHINE_IRF_FIT_BG}/{MACHINE_IRF_FIT_SIGMA}/{MACHINE_IRF_FIT_TAIL}
Machine IRF DE population/maxiter: {MACHINE_IRF_DE_POPULATION}/{MACHINE_IRF_DE_MAXITER}
Channels to fit: {channels}
Output directory: {OUT_NAME}
Change any of these defaults by passing the corresponding argument when running the code. 
Run `python -m flim_fitter --help` for more details on available arguments and their usage. 
Change defaults in configs.py or override with command line arguments as needed for different datasets and systems.
"""

if __name__ == '__main__':
    print(config_message)