from flimkit.project import ProjectFile

def test_project_collapses_zstack_group(tmp_path):
    for z in range(1, 6):
        (tmp_path / f'Series008_z{z}.ptu').write_bytes(b'')
    (tmp_path / 'CellA.ptu').write_bytes(b'')
    (tmp_path / 'CellB.ptu').write_bytes(b'')
    pf = ProjectFile.load_or_create(tmp_path)
    assert pf.scans['Series008'].scan_type == 'zstack'
    assert pf.scans['CellA'].scan_type == 'fov'
    assert pf.scans['CellB'].scan_type == 'fov'
    assert 'Series008_z1' not in pf.scans
    assert 'Series008_z4' not in pf.scans

def test_project_single_slice_stays_fov(tmp_path):
    (tmp_path / 'Foo_z1.ptu').write_bytes(b'')
    pf = ProjectFile.load_or_create(tmp_path)
    assert pf.scans['Foo_z1'].scan_type == 'fov'
