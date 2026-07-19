import json
import numpy as np
import pytest
from pathlib import Path
from flimkit import synth
from flimkit.formats.PTU.reader import PTUFile

def test_build_decay_truth_and_wrap():
    expected, irf, truth = synth.build_decay(tau_ns=3.5, n_photons=1e5)
    assert abs(expected.sum() - 1e5) / 1e5 < 1e-6
    assert truth['tau_ns'] == [3.5]
    assert truth['period_ns'] == pytest.approx(50.0)
    assert truth['wrap_residual'] < 1e-3

def test_reflection_lands_at_requested_bin():
    expected, _, truth = synth.build_decay(
        tau_ns=4.0, n_photons=1e5,
        reflection=dict(center_ns=8.0, width_ns=0.2, frac=0.05))
    refl_bin = int(round(8.0 / truth['tcspc_res_ns']))
    win = slice(refl_bin - 25, refl_bin + 25)
    local_peak = refl_bin - 25 + int(expected[win].argmax())
    assert abs(local_peak - refl_bin) <= 2
    assert truth['reflection']['frac'] == 0.05

def test_pileup_reduces_and_shortens():
    plain, _, _ = synth.build_decay(tau_ns=4.0, n_photons=1e6)
    piled, _, truth = synth.build_decay(tau_ns=4.0, n_photons=1e6, pileup_pp=0.1)
    assert piled.sum() < plain.sum()
    assert truth['pileup']['photons_per_pulse'] == 0.1

def test_write_ptu_roundtrips_photons_and_position(tmp_path):
    expected, _, truth = synth.build_decay(tau_ns=3.0, n_photons=8e4)
    cube = synth.sample_cube(expected, 8, 8, seed=0)
    path = synth.write_ptu(tmp_path / 's.ptu', cube,
                           truth['period_ns'], truth['tcspc_res_ns'])
    f = PTUFile(path, verbose=False)
    d = f.summed_decay()
    assert int(d.sum()) == int(cube.sum())
    assert f.n_bins == truth['n_bins']
    assert int(d.argmax()) == int(cube.sum(axis=(0, 1)).argmax())

def test_generate_writes_sample_irf_and_truth(tmp_path):
    r = synth.generate(tmp_path, name='g', ny=8, nx=8,
                       tau_ns=4.1, n_photons=5e4)
    assert Path(r['sample']).exists()
    t = r['truth']
    assert (tmp_path / t['irf_ptu']).exists()
    assert (tmp_path / t['sample_ptu']).exists()
    loaded = json.loads(Path(r['truth_json']).read_text())
    assert loaded['tau_ns'] == [4.1]
    assert loaded['n_photons_written'] > 0

def test_generate_series_shares_one_irf(tmp_path):
    res = synth.generate_series(tmp_path, [1e4, 5e4], name='ser',
                                tau_ns=3.5, with_reflection=True)
    assert len(res) == 2
    irfs = list(tmp_path.glob('*_IRF.ptu'))
    assert len(irfs) == 1
    for r in res:
        assert r['truth']['reflection'] is not None