_cfg: dict = {}

def _C() -> dict:
    if not _cfg:
        from flimkit.configs import (
            n_exp, Tau_min, Tau_max, D_mode, binning_factor,
            MIN_PHOTONS_PERPIX, Optimizer, lm_restarts, de_population,
            de_maxiter, n_workers, OUT_NAME, IRF_BINS, IRF_FIT_WIDTH,
            IRF_FWHM, channels, TAU_DISPLAY_MIN, TAU_DISPLAY_MAX,
            INTENSITY_DISPLAY_MIN, INTENSITY_DISPLAY_MAX,
            MACHINE_IRF_DIR, MACHINE_IRF_DEFAULT_PATH,
            MACHINE_IRF_ALIGN_ANCHOR, MACHINE_IRF_REDUCER,
        )
        _cfg.update(
            n_exp=n_exp, Tau_min=Tau_min, Tau_max=Tau_max, D_mode=D_mode,
            binning_factor=binning_factor, MIN_PHOTONS_PERPIX=MIN_PHOTONS_PERPIX,
            Optimizer=Optimizer, lm_restarts=lm_restarts,
            de_population=de_population, de_maxiter=de_maxiter,
            n_workers=n_workers, OUT_NAME=OUT_NAME,
            IRF_BINS=IRF_BINS, IRF_FIT_WIDTH=IRF_FIT_WIDTH, IRF_FWHM=IRF_FWHM,
            channels=channels,
            TAU_DISPLAY_MIN=TAU_DISPLAY_MIN, TAU_DISPLAY_MAX=TAU_DISPLAY_MAX,
            INTENSITY_DISPLAY_MIN=INTENSITY_DISPLAY_MIN,
            INTENSITY_DISPLAY_MAX=INTENSITY_DISPLAY_MAX,
            MACHINE_IRF_DIR=MACHINE_IRF_DIR,
            MACHINE_IRF_DEFAULT_PATH=MACHINE_IRF_DEFAULT_PATH,
            MACHINE_IRF_ALIGN_ANCHOR=MACHINE_IRF_ALIGN_ANCHOR,
            MACHINE_IRF_REDUCER=MACHINE_IRF_REDUCER,
        )
    return _cfg
