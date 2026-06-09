from dataclasses import dataclass
import numpy as np

@dataclass
class FRETChannelData:

    real_cal:     np.ndarray
    imag_cal:     np.ndarray
    mean:         np.ndarray
    frequency:    float
    min_photons:  float = 0.01

    def __post_init__(self):
        self.real_cal = np.asarray(self.real_cal, dtype=float)
        self.imag_cal = np.asarray(self.imag_cal, dtype=float)
        self.mean     = np.asarray(self.mean,     dtype=float)
        if self.real_cal.shape != self.imag_cal.shape:
            raise ValueError(
                f"real_cal and imag_cal must have the same shape; "
                f"got {self.real_cal.shape} and {self.imag_cal.shape}."
            )
        if self.mean.shape != self.real_cal.shape:
            raise ValueError(
                f"mean must have the same shape as real_cal; "
                f"got {self.mean.shape} vs {self.real_cal.shape}."
            )
        if self.frequency <= 0:
            raise ValueError(
                f"frequency must be positive (MHz); got {self.frequency}."
            )

    @property
    def valid_mask(self):
        return (self.mean >= self.min_photons) & np.isfinite(self.real_cal)

    @property
    def valid_g(self):
        return self.real_cal[self.valid_mask]

    @property
    def valid_s(self):
        return self.imag_cal[self.valid_mask]


@dataclass
class FRETModelParameters:

    donor_lifetime:        float
    fret_efficiency:       float = 0.0
    donor_fretting:        float = 1.0
    donor_background:      float = 0.0
    background_real:       float = 0.0
    background_imag:       float = 0.0
    acceptor_lifetime:     float | None = None
    donor_bleedthrough:    float = 0.0
    acceptor_bleedthrough: float = 0.0
    acceptor_background:   float = 0.0

    def __post_init__(self):
        if self.donor_lifetime <= 0:
            raise ValueError(
                f"donor_lifetime must be positive (ns); got {self.donor_lifetime}."
            )
        for attr, lo, hi in [
            ('fret_efficiency', 0.0, 1.0),
            ('donor_fretting',  0.0, 1.0),
        ]:
            v = getattr(self, attr)
            if not (lo <= v <= hi):
                raise ValueError(f"{attr} must be in [{lo}, {hi}]; got {v}.")


@dataclass
class FRETBounds:

    fret_efficiency:       tuple[float, float] = (0.0,  1.0)
    donor_fretting:        tuple[float, float] = (0.0,  1.0)
    donor_background:      tuple[float, float] = (0.0, 10.0)
    donor_bleedthrough:    tuple[float, float] = (0.0, 10.0)
    acceptor_bleedthrough: tuple[float, float] = (0.0, 10.0)
    acceptor_background:   tuple[float, float] = (0.0, 10.0)

    def _as_scipy(self, attrs):
        lo = [getattr(self, a)[0] for a in attrs]
        hi = [getattr(self, a)[1] for a in attrs]
        return dict(lb=lo, ub=hi)

    def donor_only_scipy(self):
        return self._as_scipy(
            ('fret_efficiency', 'donor_fretting', 'donor_background')
        )

    def joint_scipy(self):
        return self._as_scipy((
            'fret_efficiency', 'donor_fretting',
            'donor_background', 'donor_bleedthrough',
            'acceptor_bleedthrough', 'acceptor_background',
        ))


@dataclass
class FRETResult:

    fret_efficiency:       float
    donor_fretting:        float
    donor_background:      float
    donor_real_model:      float
    donor_imag_model:      float
    residual:              float
    donor_bleedthrough:    float = 0.0
    acceptor_bleedthrough: float = 0.0
    acceptor_background:   float = 0.0
    acceptor_real_model:   float | None = None
    acceptor_imag_model:   float | None = None
    converged:             bool = True
    message:               str  = ''

    def print_summary(self):
        print('\u2550\u2550\u2550 FRET fit result \u2550\u2550\u2550')
        print(f"  FRET efficiency    : {self.fret_efficiency:.4f}")
        print(f"  Donor fretting     : {self.donor_fretting:.4f}")
        print(f"  Donor background   : {self.donor_background:.4f}")
        print(f"  Donor model  (G,S) : "
              f"({self.donor_real_model:.4f}, {self.donor_imag_model:.4f})")
        if self.acceptor_real_model is not None:
            print(f"  Acceptor model (G,S): "
                  f"({self.acceptor_real_model:.4f}, {self.acceptor_imag_model:.4f})")
            print(f"  Donor bleedthrough  : {self.donor_bleedthrough:.4f}")
            print(f"  Acceptor bleedthrough: {self.acceptor_bleedthrough:.4f}")
            print(f"  Acceptor background : {self.acceptor_background:.4f}")
        print(f"  Residual           : {self.residual:.6g}")
        status = '\u2713 converged' if self.converged else '\u2717 did not converge'
        print(f"  Optimizer          : {status}  \u2014 {self.message}")

def _require_phasorpy_fret_api():
    try:
        from phasorpy.lifetime import (  # noqa: F401
            phasor_from_fret_donor,
            phasor_from_fret_acceptor,
        )
    except ImportError as exc:
        raise ImportError(
            'FRET analysis requires phasorpy >= 0.9 with '
            'phasor_from_fret_donor and phasor_from_fret_acceptor.  '
            "Install with:  pip install 'phasorpy>=0.9'"
        ) from exc

    import inspect
    from phasorpy.lifetime import phasor_from_fret_donor, phasor_from_fret_acceptor
    for fn, required in [
        (phasor_from_fret_donor,    {'frequency', 'donor_lifetime',
                                     'fret_efficiency', 'donor_fretting'}),
        (phasor_from_fret_acceptor, {'frequency', 'donor_lifetime',
                                     'acceptor_lifetime', 'fret_efficiency'}),
    ]:
        missing = required - set(inspect.signature(fn).parameters)
        if missing:
            raise ImportError(
                f"phasorpy.{fn.__name__} is missing expected parameters: "
                f"{missing}.  Please update phasorpy."
            )


def _single_lifetime_phasor(frequency, lifetime):
    from phasorpy.lifetime import phasor_from_lifetime
    real, imag = phasor_from_lifetime(frequency, lifetime)
    return float(np.squeeze(real)), float(np.squeeze(imag))


def _fret_donor_phasor(
    frequency,
    donor_lifetime,
    *,
    fret_efficiency=0.0,
    donor_fretting=1.0,
    donor_background=0.0,
    background_real=0.0,
    background_imag=0.0,
):
    from phasorpy.lifetime import phasor_from_fret_donor
    real, imag = phasor_from_fret_donor(
        frequency, donor_lifetime,
        fret_efficiency=fret_efficiency,
        donor_fretting=donor_fretting,
        donor_background=donor_background,
        background_real=background_real,
        background_imag=background_imag,
        unit_conversion=1e-3,
    )
    return float(np.squeeze(real)), float(np.squeeze(imag))


def _fret_acceptor_phasor(
    frequency,
    donor_lifetime,
    acceptor_lifetime,
    *,
    fret_efficiency=0.0,
    donor_fretting=1.0,
    donor_bleedthrough=0.0,
    acceptor_bleedthrough=0.0,
    acceptor_background=0.0,
    background_real=0.0,
    background_imag=0.0,
):
    from phasorpy.lifetime import phasor_from_fret_acceptor
    real, imag = phasor_from_fret_acceptor(
        frequency, donor_lifetime, acceptor_lifetime,
        fret_efficiency=fret_efficiency,
        donor_fretting=donor_fretting,
        donor_bleedthrough=donor_bleedthrough,
        acceptor_bleedthrough=acceptor_bleedthrough,
        acceptor_background=acceptor_background,
        background_real=background_real,
        background_imag=background_imag,
        unit_conversion=1e-3,
    )
    return float(np.squeeze(real)), float(np.squeeze(imag))


def predict_fret_trajectory(
    frequency,
    donor_lifetime,
    *,
    acceptor_lifetime=None,
    donor_fretting=1.0,
    donor_background=0.0,
    background_real=0.0,
    background_imag=0.0,
    donor_bleedthrough=0.0,
    acceptor_bleedthrough=0.0,
    acceptor_background=0.0,
    n_points=100,
):
    _require_phasorpy_fret_api()
    efficiencies = np.linspace(0.0, 1.0, n_points)

    from phasorpy.lifetime import phasor_from_fret_donor
    donor_g, donor_s = phasor_from_fret_donor(
        frequency, donor_lifetime,
        fret_efficiency=efficiencies,
        donor_fretting=donor_fretting,
        donor_background=donor_background,
        background_real=background_real,
        background_imag=background_imag,
        unit_conversion=1e-3,
    )

    acceptor_g = acceptor_s = None
    if acceptor_lifetime is not None:
        from phasorpy.lifetime import phasor_from_fret_acceptor
        acceptor_g, acceptor_s = phasor_from_fret_acceptor(
            frequency, donor_lifetime, acceptor_lifetime,
            fret_efficiency=efficiencies,
            donor_fretting=donor_fretting,
            donor_bleedthrough=donor_bleedthrough,
            acceptor_bleedthrough=acceptor_bleedthrough,
            acceptor_background=acceptor_background,
            background_real=background_real,
            background_imag=background_imag,
            unit_conversion=1e-3,
        )

    return dict(
        efficiency=efficiencies,
        donor_g=np.asarray(donor_g),
        donor_s=np.asarray(donor_s),
        acceptor_g=np.asarray(acceptor_g) if acceptor_g is not None else None,
        acceptor_s=np.asarray(acceptor_s) if acceptor_s is not None else None,
    )


def fit_donor_fret(
    donor,
    params,
    bounds=None,
    *,
    weight_by_photons=True,
):
    _require_phasorpy_fret_api()
    from phasorpy.lifetime import phasor_from_fret_donor
    from scipy.optimize import least_squares

    if bounds is None:
        bounds = FRETBounds()

    mask = donor.valid_mask
    g_vals = donor.real_cal[mask]
    s_vals = donor.imag_cal[mask]
    if weight_by_photons:
        w = donor.mean[mask]
        w = w / w.sum()
    else:
        w = np.ones(g_vals.size) / g_vals.size

    g_obs = float(np.dot(w, g_vals))
    s_obs = float(np.dot(w, s_vals))

    freq = donor.frequency
    tau_d = params.donor_lifetime
    bg_real = params.background_real
    bg_imag = params.background_imag

    def _residuals(x: np.ndarray) -> np.ndarray:
        E, f, bg = x
        g_m, s_m = phasor_from_fret_donor(
            freq, tau_d,
            fret_efficiency=E,
            donor_fretting=f,
            donor_background=bg,
            background_real=bg_real,
            background_imag=bg_imag,
            unit_conversion=1e-3,
        )
        return np.array([float(g_m) - g_obs, float(s_m) - s_obs])

    x0 = [params.fret_efficiency, params.donor_fretting, params.donor_background]
    scipy_bounds = bounds.donor_only_scipy()
    result = least_squares(
        _residuals, x0,
        bounds=(scipy_bounds['lb'], scipy_bounds['ub']),
        method='trf',
    )

    E_fit, f_fit, bg_fit = result.x
    g_model, s_model = phasor_from_fret_donor(
        freq, tau_d,
        fret_efficiency=E_fit,
        donor_fretting=f_fit,
        donor_background=bg_fit,
        background_real=bg_real,
        background_imag=bg_imag,
        unit_conversion=1e-3,
    )

    return FRETResult(
        fret_efficiency=float(E_fit),
        donor_fretting=float(f_fit),
        donor_background=float(bg_fit),
        donor_real_model=float(g_model),
        donor_imag_model=float(s_model),
        residual=float(result.cost),
        converged=bool(result.success),
        message=result.message,
    )


#Joint Donor+Acceptor Solver


def fit_joint_fret(
    donor,
    acceptor,
    params,
    bounds=None,
    *,
    weight_by_photons=True,
):
    if params.acceptor_lifetime is None:
        raise ValueError(
            'fit_joint_fret requires params.acceptor_lifetime to be set.'
        )
    if donor.real_cal.shape != acceptor.real_cal.shape:
        raise ValueError(
            f"donor and acceptor arrays must have the same shape; "
            f"got {donor.real_cal.shape} vs {acceptor.real_cal.shape}."
        )
    if donor.frequency != acceptor.frequency:
        raise ValueError(
            f"donor and acceptor must share the same frequency; "
            f"got {donor.frequency} vs {acceptor.frequency} MHz."
        )

    _require_phasorpy_fret_api()
    from phasorpy.lifetime import phasor_from_fret_donor, phasor_from_fret_acceptor
    from scipy.optimize import least_squares

    if bounds is None:
        bounds = FRETBounds()

    def _centroid(ch: FRETChannelData) -> tuple[float, float]:
        mask = ch.valid_mask
        g_vals = ch.real_cal[mask]
        s_vals = ch.imag_cal[mask]
        if weight_by_photons:
            w = ch.mean[mask]
            w = w / w.sum()
        else:
            w = np.ones(g_vals.size) / g_vals.size
        return float(np.dot(w, g_vals)), float(np.dot(w, s_vals))

    dg_obs, ds_obs = _centroid(donor)
    ag_obs, as_obs = _centroid(acceptor)

    freq   = donor.frequency
    tau_d  = params.donor_lifetime
    tau_a  = params.acceptor_lifetime
    bg_real = params.background_real
    bg_imag = params.background_imag

    def _residuals(x: np.ndarray) -> np.ndarray:
        E, f, d_bg, d_bt, a_bt, a_bg = x
        dg_m, ds_m = phasor_from_fret_donor(
            freq, tau_d,
            fret_efficiency=E,
            donor_fretting=f,
            donor_background=d_bg,
            background_real=bg_real,
            background_imag=bg_imag,
            unit_conversion=1e-3,
        )
        ag_m, as_m = phasor_from_fret_acceptor(
            freq, tau_d, tau_a,
            fret_efficiency=E,
            donor_fretting=f,
            donor_bleedthrough=d_bt,
            acceptor_bleedthrough=a_bt,
            acceptor_background=a_bg,
            background_real=bg_real,
            background_imag=bg_imag,
            unit_conversion=1e-3,
        )
        return np.array([
            float(dg_m) - dg_obs,
            float(ds_m) - ds_obs,
            float(ag_m) - ag_obs,
            float(as_m) - as_obs,
        ])

    x0 = [
        params.fret_efficiency,
        params.donor_fretting,
        params.donor_background,
        params.donor_bleedthrough,
        params.acceptor_bleedthrough,
        params.acceptor_background,
    ]
    scipy_bounds = bounds.joint_scipy()
    result = least_squares(
        _residuals, x0,
        bounds=(scipy_bounds['lb'], scipy_bounds['ub']),
        method='trf',
    )

    E_fit, f_fit, d_bg_fit, d_bt_fit, a_bt_fit, a_bg_fit = result.x

    dg_model, ds_model = phasor_from_fret_donor(
        freq, tau_d,
        fret_efficiency=E_fit,
        donor_fretting=f_fit,
        donor_background=d_bg_fit,
        background_real=bg_real,
        background_imag=bg_imag,
        unit_conversion=1e-3,
    )
    ag_model, as_model = phasor_from_fret_acceptor(
        freq, tau_d, tau_a,
        fret_efficiency=E_fit,
        donor_fretting=f_fit,
        donor_bleedthrough=d_bt_fit,
        acceptor_bleedthrough=a_bt_fit,
        acceptor_background=a_bg_fit,
        background_real=bg_real,
        background_imag=bg_imag,
        unit_conversion=1e-3,
    )

    return FRETResult(
        fret_efficiency=float(E_fit),
        donor_fretting=float(f_fit),
        donor_background=float(d_bg_fit),
        donor_real_model=float(dg_model),
        donor_imag_model=float(ds_model),
        residual=float(result.cost),
        donor_bleedthrough=float(d_bt_fit),
        acceptor_bleedthrough=float(a_bt_fit),
        acceptor_background=float(a_bg_fit),
        acceptor_real_model=float(ag_model),
        acceptor_imag_model=float(as_model),
        converged=bool(result.success),
        message=result.message,
    )


#Pixelwise Maps


def map_fret_efficiency(
    donor: FRETChannelData,
    params: FRETModelParameters,
    bounds: Optional[FRETBounds] = None,
    *,
    acceptor: Optional[FRETChannelData] = None,
    weight_by_photons: bool = True,
) -> dict:
    if bounds is None:
        bounds = FRETBounds()

    Y, X = donor.real_cal.shape
    efficiency = np.full((Y, X), np.nan)
    fretting   = np.full((Y, X), np.nan)
    residual   = np.full((Y, X), np.nan)
    converged  = np.zeros((Y, X), dtype=bool)

    mask = donor.valid_mask
    if acceptor is not None:
        mask = mask & acceptor.valid_mask

    ys, xs = np.where(mask)

    for y, x in zip(ys, xs):
        px_donor = FRETChannelData(
            real_cal=donor.real_cal[y:y+1, x:x+1],
            imag_cal=donor.imag_cal[y:y+1, x:x+1],
            mean=donor.mean[y:y+1, x:x+1],
            frequency=donor.frequency,
            min_photons=donor.min_photons,
        )
        if acceptor is not None:
            px_acceptor = FRETChannelData(
                real_cal=acceptor.real_cal[y:y+1, x:x+1],
                imag_cal=acceptor.imag_cal[y:y+1, x:x+1],
                mean=acceptor.mean[y:y+1, x:x+1],
                frequency=acceptor.frequency,
                min_photons=acceptor.min_photons,
            )
            r = fit_joint_fret(
                px_donor, px_acceptor, params, bounds,
                weight_by_photons=weight_by_photons,
            )
        else:
            r = fit_donor_fret(
                px_donor, params, bounds,
                weight_by_photons=weight_by_photons,
            )
        efficiency[y, x] = r.fret_efficiency
        fretting[y, x]   = r.donor_fretting
        residual[y, x]   = r.residual
        converged[y, x]  = r.converged

    return dict(
        efficiency=efficiency,
        fretting=fretting,
        residual=residual,
        converged=converged,
    )


#Visualization


def plot_fret_trajectory(
    frequency: float,
    donor_lifetime: float,
    *,
    acceptor_lifetime: Optional[float] = None,
    donor_fretting: float = 1.0,
    n_points: int = 100,
    ax=None,
    donor_kw: Optional[dict] = None,
    acceptor_kw: Optional[dict] = None,
) -> tuple:
    import matplotlib.pyplot as plt

    traj = predict_fret_trajectory(
        frequency, donor_lifetime,
        acceptor_lifetime=acceptor_lifetime,
        donor_fretting=donor_fretting,
        n_points=n_points,
    )

    if ax is None:
        _, ax = plt.subplots()

    _donor_kw = {'color': 'steelblue', 'label': 'donor trajectory'}
    if donor_kw:
        _donor_kw.update(donor_kw)

    lines = [ax.plot(traj['donor_g'], traj['donor_s'], **_donor_kw)[0]]

    if traj['acceptor_g'] is not None:
        _acceptor_kw = {'color': 'tomato', 'label': 'acceptor trajectory'}
        if acceptor_kw:
            _acceptor_kw.update(acceptor_kw)
        lines.append(
            ax.plot(traj['acceptor_g'], traj['acceptor_s'], **_acceptor_kw)[0]
        )

    return ax, lines


def plot_fret_fit(
    donor: FRETChannelData,
    result: FRETResult,
    frequency: float,
    donor_lifetime: float,
    *,
    acceptor: Optional[FRETChannelData] = None,
    acceptor_lifetime: Optional[float] = None,
    n_trajectory: int = 100,
    ax=None,
    scatter_kw: Optional[dict] = None,
    trajectory_kw: Optional[dict] = None,
) -> tuple:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    _scatter_kw = {'marker': '.', 'alpha': 0.3, 'linestyle': 'none', 'color': 'steelblue'}
    if scatter_kw:
        _scatter_kw.update(scatter_kw)

    artists: dict = {}

    artists['donor_scatter'] = ax.plot(
        donor.valid_g, donor.valid_s, **_scatter_kw
    )[0]

    if acceptor is not None:
        _acc_scatter_kw = dict(_scatter_kw)
        _acc_scatter_kw['color'] = 'tomato'
        _acc_scatter_kw.pop('label', None)
        artists['acceptor_scatter'] = ax.plot(
            acceptor.valid_g, acceptor.valid_s, **_acc_scatter_kw
        )[0]

    _traj_kw = trajectory_kw or {}
    plot_fret_trajectory(
        frequency, donor_lifetime,
        acceptor_lifetime=acceptor_lifetime,
        donor_fretting=result.donor_fretting,
        n_points=n_trajectory,
        ax=ax,
        **_traj_kw,
    )

    artists['donor_model'] = ax.plot(
        result.donor_real_model, result.donor_imag_model,
        marker='*', markersize=12, color='steelblue',
        linestyle='none', label='donor fit',
        zorder=5,
    )[0]

    if result.acceptor_real_model is not None:
        artists['acceptor_model'] = ax.plot(
            result.acceptor_real_model, result.acceptor_imag_model,
            marker='*', markersize=12, color='tomato',
            linestyle='none', label='acceptor fit',
            zorder=5,
        )[0]

    return ax, artists
