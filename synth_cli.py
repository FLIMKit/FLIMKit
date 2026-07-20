#!/usr/bin/env python
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
from flimkit import synth

def _parse_floats(spec):
    return [float(x) for x in str(spec).split(',') if x.strip()]

def main():
    ap = argparse.ArgumentParser(
        description='Generate synthetic FLIM PTUs with a known ground truth '
                    '(sample + matching IRF) for cross-software validation.')
    ap.add_argument('--out', required=True,
                    help='output directory for the PTUs and truth JSON')
    ap.add_argument('--name', default='synth', help='base name for the files')
    ap.add_argument('--tau', default='4.1',
                    help='lifetime(s) in ns, comma-separated for multi-exp (e.g. "3.0,0.8")')
    ap.add_argument('--amps', default=None,
                    help='amplitudes for multi-exp taus, comma-separated (default: equal)')
    ap.add_argument('--photons', default='1e5',
                    help='summed photon count; comma-separated makes a series '
                         '(e.g. "2e4,1e5,5e5,2.5e6")')
    ap.add_argument('--period-ns', type=float, default=50.0,
                    help='laser period in ns (sets the sync rate)')
    ap.add_argument('--res-ps', type=float, default=25.0, help='TCSPC bin width in ps')
    ap.add_argument('--irf-fwhm-ns', type=float, default=0.15, help='IRF FWHM in ns')
    ap.add_argument('--irf-center-ns', type=float, default=2.0, help='IRF peak position in ns')
    ap.add_argument('--reflection-ns', type=float, default=None,
                    help='plant a reflection peak at this time (ns), e.g. 8.0')
    ap.add_argument('--reflection-frac', type=float, default=0.02,
                    help='reflection intensity as a fraction of total signal')
    ap.add_argument('--reflection-width-ns', type=float, default=0.15,
                    help='reflection peak FWHM in ns')
    ap.add_argument('--pileup-pp', type=float, default=None,
                    help='apply pile-up at this many photons per pulse (e.g. 0.1)')
    ap.add_argument('--background-frac', type=float, default=0.0,
                    help='flat background as a fraction of total signal')
    ap.add_argument('--image', type=int, default=16,
                    help='image side length in pixels (square)')
    ap.add_argument('--no-irf', action='store_true', help='skip writing the IRF PTU')
    ap.add_argument('--sdt', action='store_true',
                    help='also write Becker & Hickl .sdt versions (sample + IRF)')
    args = ap.parse_args()
    taus = _parse_floats(args.tau)
    tau_arg = taus[0] if len(taus) == 1 else taus
    amps = _parse_floats(args.amps) if args.amps else None
    res_ns = args.res_ps / 1000.0
    n_bins = int(round(args.period_ns / res_ns))
    reflection = None
    if args.reflection_ns is not None:
        reflection = dict(center_ns=args.reflection_ns, frac=args.reflection_frac,
                          width_ns=args.reflection_width_ns)
    common = dict(tau_ns=tau_arg, amps=amps, n_bins=n_bins, tcspc_res_ns=res_ns,
                  irf_fwhm_ns=args.irf_fwhm_ns, irf_center_ns=args.irf_center_ns,
                  background_frac=args.background_frac, pileup_pp=args.pileup_pp)
    photons = _parse_floats(args.photons)
    print(f'Writing to {args.out}')
    if len(photons) == 1:
        r = synth.generate(args.out, name=args.name, ny=args.image, nx=args.image,
                           with_irf=not args.no_irf, n_photons=photons[0],
                           reflection=reflection, sdt=args.sdt, **common)
        _report(r)
    else:
        results = synth.generate_series(
            args.out, photons, name=args.name,
            with_reflection=reflection is not None,
            reflection=reflection, ny=args.image, nx=args.image, sdt=args.sdt, **common)
        for r in results:
            _report(r)
    print('Done.')

def _report(r):
    t = r['truth']
    refl = f", reflection {t['reflection']['frac']:.1%} @ {t['reflection']['center_ns']}ns" \
        if t['reflection'] else ''
    print(f"  {Path(r['sample']).name}: tau={t['tau_ns']} ns, "
          f"{t['n_photons_written']:,} photons{refl}")

if __name__ == '__main__':
    main()
