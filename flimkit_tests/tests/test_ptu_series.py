import numpy as np
import pytest

from flimkit.formats.PTU.series import (
    parse_series_name,
    index_ptu_series,
    describe_series,
    register_tile_pair,
    recover_tile_positions,
    plane_tile_positions,
)

pytestmark = pytest.mark.unit

def _touch(directory, names):
    for n in names:
        (directory / n).write_bytes(b'')

def test_parse_full_series_name():
    p = parse_series_name('R 1 (4)_t100_s2_z3.ptu')
    assert p['base'] == 'R 1 (4)'
    assert (p['t'], p['s'], p['z']) == (100, 2, 3)
    assert p['has_t'] and p['has_z']

def test_parse_tile_only_name_defaults_to_one():
    p = parse_series_name('R 2_s7.ptu')
    assert p['base'] == 'R 2'
    assert (p['t'], p['s'], p['z']) == (1, 7, 1)
    assert not p['has_t'] and not p['has_z']

def test_parse_rejects_non_series_name():
    assert parse_series_name('calibration.ptu') is None

def test_parse_shares_grammar_with_batch_fit():
    from flimkit.utils.batch_fit import parse_timelapse_filename
    for name in ('R 1 (4)_t100_s2_z3.ptu', 'R 1_t3_z2.ptu', 'R 1_t7.ptu'):
        region, t, s, z = parse_timelapse_filename(name)
        p = parse_series_name(name)
        assert (p['base'], p['t'], p['s'], p['z']) == (region, t, s or 1, z or 1)

def test_index_rejects_single_position_series(tmp_path):
    _touch(tmp_path, [f'R 1_t{t}_z1.ptu' for t in (1, 2, 3)])
    with pytest.raises(RuntimeError, match='nothing to stitch'):
        index_ptu_series(tmp_path)

def test_index_groups_by_plane(tmp_path):
    _touch(tmp_path, [f'R 1_t{t}_s{s}_z{z}.ptu'
                      for t in (1, 2) for s in (1, 2) for z in (1, 2, 3)])
    idx = index_ptu_series(tmp_path)
    assert idx['base'] == 'R 1'
    assert idx['timepoints'] == [1, 2]
    assert idx['z_planes'] == [1, 2, 3]
    assert idx['tiles'] == [1, 2]
    assert idx['n_files'] == 12
    assert len(idx['planes']) == 6
    assert [e['s'] for e in idx['planes'][(2, 3)]] == [1, 2]
    assert not idx['is_ragged']

def test_index_handles_plain_tile_series(tmp_path):
    _touch(tmp_path, [f'R 2_s{s}.ptu' for s in range(1, 5)])
    idx = index_ptu_series(tmp_path)
    assert idx['timepoints'] == [1]
    assert idx['z_planes'] == [1]
    assert idx['tiles'] == [1, 2, 3, 4]
    assert not idx['has_t'] and not idx['has_z']

def test_index_flags_ragged_series(tmp_path):
    _touch(tmp_path, ['R 1_t1_s1_z1.ptu', 'R 1_t1_s2_z1.ptu', 'R 1_t2_s1_z1.ptu'])
    assert index_ptu_series(tmp_path)['is_ragged']

def test_index_rejects_mixed_basenames(tmp_path):
    _touch(tmp_path, ['R 1_s1.ptu', 'R 2_s1.ptu'])
    with pytest.raises(RuntimeError, match='Multiple series'):
        index_ptu_series(tmp_path)

def test_index_selects_requested_basename(tmp_path):
    _touch(tmp_path, ['R 1_s1.ptu', 'R 1_s2.ptu', 'R 2_s1.ptu'])
    idx = index_ptu_series(tmp_path, ptu_basename='R 1')
    assert idx['tiles'] == [1, 2]

def test_index_raises_when_empty(tmp_path):
    _touch(tmp_path, ['notes.txt', 'calibration.ptu'])
    with pytest.raises(RuntimeError, match='No PTU files'):
        index_ptu_series(tmp_path)

def test_describe_series_reports_counts(tmp_path):
    _touch(tmp_path, [f'R 1_t{t}_s{s}_z1.ptu' for t in (1, 2, 3) for s in (1, 2)])
    text = describe_series(index_ptu_series(tmp_path))
    assert '3 timepoint(s)' in text and '2 tile(s)' in text

def _mosaic(shape=(160, 140), seed=0):
    rng = np.random.default_rng(seed)
    scene = rng.gamma(2.0, 40.0, size=shape)
    for _ in range(25):
        y, x = rng.integers(8, shape[0] - 8), rng.integers(8, shape[1] - 8)
        scene[y - 6:y + 6, x - 6:x + 6] += rng.uniform(200, 900)
    return scene

def test_register_tile_pair_recovers_known_vertical_shift():
    scene = _mosaic()
    a = scene[0:100, 0:120]
    b = scene[60:160, 0:120]
    reg = register_tile_pair(a, b)
    assert (reg['dy'], reg['dx']) == (60, 0)
    assert reg['correlation'] > 0.9
    assert reg['overlap_px'] == 40 * 120

def test_register_tile_pair_recovers_diagonal_shift():
    scene = _mosaic(seed=3)
    a = scene[0:100, 0:100]
    b = scene[55:155, 18:118]
    reg = register_tile_pair(a, b)
    assert (reg['dy'], reg['dx']) == (55, 18)
    assert reg['correlation'] > 0.9

def test_register_tile_pair_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match='Tile shapes differ'):
        register_tile_pair(np.zeros((10, 10)), np.zeros((10, 12)))

def test_recover_tile_positions_chains_three_tiles():
    scene = _mosaic(shape=(260, 140), seed=5)
    tiles = [scene[0:100, 0:120], scene[70:170, 0:120], scene[140:240, 0:120]]
    positions, pairs = recover_tile_positions(tiles)
    assert [p['pixel_y'] for p in positions] == [0, 70, 140]
    assert [p['pixel_x'] for p in positions] == [0, 0, 0]
    assert len(pairs) == 2

def test_recover_tile_positions_normalises_to_origin():
    scene = _mosaic(shape=(260, 140), seed=7)
    tiles = [scene[140:240, 0:120], scene[70:170, 0:120]]
    positions, _ = recover_tile_positions(tiles)
    assert min(p['pixel_y'] for p in positions) == 0
    assert [p['pixel_y'] for p in positions] == [70, 0]

def test_recover_tile_positions_rejects_uncorrelated_tiles():
    rng = np.random.default_rng(11)
    tiles = [rng.gamma(2.0, 40.0, (100, 100)), rng.gamma(2.0, 40.0, (100, 100))]
    with pytest.raises(RuntimeError, match='registration too weak'):
        recover_tile_positions(tiles, min_correlation=0.9)

def test_recover_tile_positions_needs_two_tiles():
    with pytest.raises(ValueError, match='at least two tiles'):
        recover_tile_positions([np.zeros((10, 10))])

def test_plane_tile_positions_maps_reference_onto_plane():
    reference = [
        {'s': 1, 'pixel_y': 0, 'pixel_x': 0, 'file': 'R 1_t1_s1_z1.ptu'},
        {'s': 2, 'pixel_y': 54, 'pixel_x': 3, 'file': 'R 1_t1_s2_z1.ptu'},
    ]
    entries = [{'s': 2, 'file': 'R 1_t9_s2_z2.ptu'}, {'s': 1, 'file': 'R 1_t9_s1_z2.ptu'}]
    out = plane_tile_positions(reference, entries)
    assert [p['file'] for p in out] == ['R 1_t9_s1_z2.ptu', 'R 1_t9_s2_z2.ptu']
    assert [p['pixel_y'] for p in out] == [0, 54]

def test_plane_tile_positions_reports_missing_tile():
    reference = [{'s': 1, 'pixel_y': 0, 'pixel_x': 0, 'file': 'a.ptu'}]
    with pytest.raises(RuntimeError, match='no recovered position'):
        plane_tile_positions(reference, [{'s': 2, 'file': 'b.ptu'}])

def test_fit_flim_tiles_requires_positions_or_metadata(tmp_path):
    from types import SimpleNamespace
    from flimkit.formats.PTU.stitch import fit_flim_tiles
    with pytest.raises(ValueError, match='xlif_path or tile_positions'):
        fit_flim_tiles(None, tmp_path, tmp_path, SimpleNamespace(nexp=2))

def test_fit_flim_series_rejects_ragged_series(tmp_path):
    from types import SimpleNamespace
    from flimkit.formats.PTU.stitch import fit_flim_series
    _touch(tmp_path, ['R 1_t1_s1_z1.ptu', 'R 1_t1_s2_z1.ptu', 'R 1_t2_s1_z1.ptu'])
    with pytest.raises(RuntimeError, match='different tile count'):
        fit_flim_series(tmp_path, tmp_path / 'out', SimpleNamespace(nexp=2),
                        verbose=False)
