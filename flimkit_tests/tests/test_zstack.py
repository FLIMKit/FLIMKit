import argparse
from pathlib import Path

import pytest

from flimkit.FLIM.batch import fit_zstack
from flimkit.utils.batch_fit import (
    parse_zstack_filename,
    group_zstack_files,
    zstack_group_label,
    parse_timelapse_filename,
)

def test_parse_pure_zstack():
    assert parse_zstack_filename('Series008_z1.ptu') == ('Series008', 0, 0, 1)
    assert parse_zstack_filename('Series008_z8.ptu') == ('Series008', 0, 0, 8)

def test_parse_region_with_spaces_and_time():
    assert parse_zstack_filename('R 1 (4)_t100_s1_z1.ptu') == ('R 1 (4)', 100, 1, 1)
    assert parse_zstack_filename('R 1 (4)_t100_s2_z3.ptu') == ('R 1 (4)', 100, 2, 3)

def test_parse_rejects_non_zstack():
    assert parse_zstack_filename('Series008.ptu') is None
    assert parse_zstack_filename('WSLogfile.sptl') is None
    assert parse_zstack_filename('notes.txt') is None

def test_pure_zstack_rejected_by_timelapse_parser():
    assert parse_timelapse_filename('Series008_z1.ptu') is None
    assert parse_zstack_filename('Series008_z1.ptu') is not None

def test_group_single_stack(tmp_path):
    for z in range(1, 9):
        (tmp_path / f'Series008_z{z}.ptu').write_bytes(b'')
    groups = group_zstack_files(tmp_path)
    assert list(groups.keys()) == [('Series008', 0, 0)]
    assert sorted(groups[('Series008', 0, 0)]) == list(range(1, 9))

def test_group_sorts_z_numerically(tmp_path):
    for z in (1, 2, 10, 11):
        (tmp_path / f'Stack_z{z}.ptu').write_bytes(b'')
    zslices = group_zstack_files(tmp_path)[('Stack', 0, 0)]
    assert sorted(zslices) == [1, 2, 10, 11]

def test_group_splits_time_and_position(tmp_path):
    for t in (100, 101):
        for s in (1, 2):
            for z in (1, 2, 3):
                (tmp_path / f'R 1 (4)_t{t}_s{s}_z{z}.ptu').write_bytes(b'')
    groups = group_zstack_files(tmp_path)
    assert len(groups) == 4
    assert ('R 1 (4)', 100, 1) in groups
    assert sorted(groups[('R 1 (4)', 101, 2)]) == [1, 2, 3]

def test_group_ignores_non_ptu(tmp_path):
    (tmp_path / 'Series008_z1.ptu').write_bytes(b'')
    (tmp_path / 'WSLogfile.sptl').write_bytes(b'')
    (tmp_path / 'notes.txt').write_bytes(b'')
    groups = group_zstack_files(tmp_path)
    assert list(groups.keys()) == [('Series008', 0, 0)]

def test_label():
    assert zstack_group_label('Series008', 0, 0) == 'Series008'
    assert zstack_group_label('R 1 (4)', 100, 2) == 'R 1 (4)_t0100_s2'

def test_fit_zstack_empty_dir_raises(tmp_path):
    args = argparse.Namespace(nexp=2, tau_min=0.1, tau_max=5.0, optimizer='de')
    with pytest.raises(ValueError):
        fit_zstack(str(tmp_path), str(tmp_path / 'out'), args)
