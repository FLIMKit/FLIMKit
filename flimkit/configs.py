import sys
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

FLIM_CMAP = LinearSegmentedColormap.from_list(
    'flim', ['#000080','#0000ff','#00ffff','#00ff00','#ffff00','#ff0000']
)

MIN_PHOTONS_PERPIX = 10

INTENSITY_THRESHOLD = None

Tau_min = 0.145
Tau_max = 45.0

D_mode = 'both'

n_exp = 3

binning_factor = 1
Optimizer = 'de'

lm_restarts = 8

de_population = 30
de_maxiter = 5000
de_n_starts = None
de_seed = 42
n_workers = -1

IRF_FWHM = None
IRF_FIT_WIDTH = 0.5
IRF_BINS = 21
Estimate_IRF = 'none'

_is_frozen = getattr(sys, 'frozen', False)

_BUNDLED_MACHINE_IRF_DIR = Path(__file__).resolve().parent / 'machine_irf'

USER_MACHINE_IRF_DIR = Path.home() / '.flimkit' / 'machine_irf'
MACHINE_IRF_DIR = USER_MACHINE_IRF_DIR if _is_frozen else _BUNDLED_MACHINE_IRF_DIR

_user_default = USER_MACHINE_IRF_DIR / 'machine_irf_default.npy'
FACTORY_MACHINE_IRF_DEFAULT_PATH = (
    (_user_default if _user_default.exists() else _BUNDLED_MACHINE_IRF_DIR / 'machine_irf_default.npy')
    if _is_frozen
    else _BUNDLED_MACHINE_IRF_DIR / 'machine_irf_default.npy'
)

def preferred_machine_irf_path():
    import json
    config_file = Path.home() / '.flimkit' / 'config.json'
    try:
        with open(config_file, encoding='utf-8') as fh:
            chosen = (json.load(fh).get('preferences') or {}).get('machine_irf_path') or ''
    except (OSError, ValueError, AttributeError):
        return None
    if not chosen:
        return None
    chosen = Path(chosen).expanduser()
    if not chosen.is_file():
        print(f'[Config] Default machine IRF {chosen} is missing, using {FACTORY_MACHINE_IRF_DEFAULT_PATH}')
        return None
    return chosen

MACHINE_IRF_DEFAULT_PATH = preferred_machine_irf_path() or FACTORY_MACHINE_IRF_DEFAULT_PATH
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

Cost_function = 'poisson'

TAU_DISPLAY_MIN = None
TAU_DISPLAY_MAX = None

INTENSITY_DISPLAY_MIN = None
INTENSITY_DISPLAY_MAX = None

PHASOR_FILTER = None
PHASOR_FILTER_SIGMA = 1.0
PHASOR_FILTER_SIZE = 3
PHASOR_FILTER_WAVELET = 'db4'
PHASOR_FILTER_LEVEL = 1

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
Differential Evolution starts: {1 if de_n_starts is None else de_n_starts}, first seed {de_seed}
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