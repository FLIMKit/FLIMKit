import json
import numpy as np
from pathlib import Path
from flimkit.FLIM.models import apply_pileup

def gaussian_irf(n_bins, center_bin, fwhm_bins):
    sigma = fwhm_bins / 2.3548
    b = np.arange(n_bins, dtype=float)
    irf = np.exp(-0.5 * ((b - center_bin) / sigma) ** 2)
    s = irf.sum()
    return irf / s if s > 0 else irf

def build_decay(tau_ns, amps=None, n_bins=2000, tcspc_res_ns=0.025,
                irf_fwhm_ns=0.15, irf_center_ns=2.0, n_photons=1e5,
                background_frac=0.0, reflection=None, pileup_pp=None):
    taus = np.atleast_1d(np.asarray(tau_ns, dtype=float))
    if amps is None:
        amps = np.ones_like(taus)
    amps = np.atleast_1d(np.asarray(amps, dtype=float))
    amps = amps / amps.sum()
    t = np.arange(n_bins) * tcspc_res_ns
    kernel = np.zeros(n_bins)
    for a, tau in zip(amps, taus):
        kernel += a * np.exp(-t / tau)
    center_bin = irf_center_ns / tcspc_res_ns
    fwhm_bins = irf_fwhm_ns / tcspc_res_ns
    irf = gaussian_irf(n_bins, center_bin, fwhm_bins)
    model = np.real(np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(irf)))
    model = np.maximum(model, 0.0)
    model = model / model.sum()
    refl_truth = None
    if reflection is not None:
        rc_bin = reflection['center_ns'] / tcspc_res_ns
        rw_bin = max(reflection.get('width_ns', 0.15) / tcspc_res_ns, 0.5)
        band = np.exp(-0.5 * ((np.arange(n_bins) - rc_bin) / (rw_bin / 2.3548)) ** 2)
        band = band / band.sum()
        frac = float(reflection['frac'])
        model = (1.0 - frac) * model + frac * band
        refl_truth = dict(center_ns=reflection['center_ns'],
                          width_ns=reflection.get('width_ns', 0.15), frac=frac)
    if background_frac > 0:
        model = (1.0 - background_frac) * model + background_frac / n_bins
    expected = model * float(n_photons)
    pileup_truth = None
    if pileup_pp is not None and pileup_pp > 0:
        n_sync = float(n_photons) / float(pileup_pp)
        expected = apply_pileup(expected, n_sync)
        pileup_truth = dict(photons_per_pulse=float(pileup_pp), n_sync=n_sync)
    truth = dict(
        tau_ns=taus.tolist(),
        amps=amps.tolist(),
        n_bins=int(n_bins),
        tcspc_res_ns=float(tcspc_res_ns),
        period_ns=float(n_bins * tcspc_res_ns),
        irf_fwhm_ns=float(irf_fwhm_ns),
        irf_center_ns=float(irf_center_ns),
        n_photons_target=float(n_photons),
        background_frac=float(background_frac),
        reflection=refl_truth,
        pileup=pileup_truth,
        wrap_residual=float(kernel[-1] / kernel.max()),
    )
    return expected, irf, truth

def sample_cube(expected, ny, nx, seed=0):
    rng = np.random.default_rng(seed)
    per_px = expected / float(ny * nx)
    cube = rng.poisson(per_px[None, None, :] * np.ones((ny, nx, 1)))
    return cube.astype(np.uint32)

def write_ptu(path, cube, period_ns, tcspc_res_ns, pixel_margin=10.0):
    import ptufile
    cube = np.ascontiguousarray(cube, dtype=np.uint32)
    period_s = period_ns * 1e-9
    res_s = tcspc_res_ns * 1e-9
    max_px = int(cube.sum(axis=2).max()) if cube.size else 0
    pixel_time = max(max_px, 1) * period_s * pixel_margin
    w = ptufile.PtuWriter(str(path), shape=cube.shape,
                          global_resolution=period_s,
                          tcspc_resolution=res_s, pixel_time=pixel_time, mode='w')
    w.write(cube)
    w.close()
    return str(path)

def write_irf_ptu(path, truth, n_photons=2e5, ny=8, nx=8, seed=1):
    nb = truth['n_bins']
    res = truth['tcspc_res_ns']
    center_bin = truth['irf_center_ns'] / res
    fwhm_bins = truth['irf_fwhm_ns'] / res
    irf = gaussian_irf(nb, center_bin, fwhm_bins) * float(n_photons)
    cube = sample_cube(irf, ny, nx, seed=seed)
    return write_ptu(path, cube, truth['period_ns'], res)

def generate(out_dir, name='synth', ny=16, nx=16, with_irf=True, seed=0, **kwargs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected, irf, truth = build_decay(**kwargs)
    cube = sample_cube(expected, ny, nx, seed=seed)
    truth['image_shape'] = [ny, nx]
    truth['n_photons_written'] = int(cube.sum())
    sample_path = out_dir / f'{name}.ptu'
    write_ptu(sample_path, cube, truth['period_ns'], truth['tcspc_res_ns'])
    truth['sample_ptu'] = sample_path.name
    if with_irf:
        irf_path = out_dir / f'{name}_IRF.ptu'
        write_irf_ptu(irf_path, truth)
        truth['irf_ptu'] = irf_path.name
    truth_path = out_dir / f'{name}_truth.json'
    truth_path.write_text(json.dumps(truth, indent=2))
    return dict(sample=str(sample_path), truth=truth, truth_json=str(truth_path))

def generate_series(out_dir, photon_counts, name='synth', with_reflection=True,
                    reflection=None, **kwargs):
    out_dir = Path(out_dir)
    if reflection is None:
        reflection = dict(center_ns=8.0, width_ns=0.15, frac=0.02)
    results = []
    for i, n in enumerate(photon_counts):
        refl = reflection if with_reflection else None
        tag = f'{name}_{int(n):d}ph'
        res = generate(out_dir, name=tag, with_irf=(i == 0), seed=i,
                       n_photons=n, reflection=refl, **kwargs)
        results.append(res)
    return results