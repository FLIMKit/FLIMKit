import numpy as np

from flimkit import synth
from flimkit.FLIM.fitters import fit_summed


def test_multistart_recovers_high_count_biexponential_truth():
    expected, irf, _truth = synth.build_decay(
        tau_ns=[3.0, 0.8],
        amps=[0.6, 0.4],
        n_bins=2000,
        tcspc_res_ns=0.025,
        irf_fwhm_ns=0.15,
        irf_center_ns=2.0,
        n_photons=1e6,
        background_frac=0.0,
    )
    decay = np.random.default_rng(0).poisson(expected).astype(float)

    _params, summary = fit_summed(
        decay=decay,
        tcspc_res=0.025e-9,
        n_bins=2000,
        irf_prompt=irf,
        has_tail=False,
        fit_bg=True,
        fit_sigma=False,
        n_exp=2,
        tau_min_ns=0.145,
        tau_max_ns=45.0,
        optimizer='lm_multistart',
        n_restarts=8,
        cost_function='poisson',
        irf_shift_bins=2,
    )

    fitted = np.asarray(summary['taus_ns'])
    truth = np.asarray([3.0, 0.8])
    relative_error = np.abs(fitted - truth) / truth
    termination = summary['optimizer_msg']

    assert relative_error.max() < 0.05, (
        f'multistart failed to recover biexponential truth: '
        f'taus={fitted.tolist()}, relative_error={relative_error.tolist()}, '
        f'termination={termination}'
    )
