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

def test_project_json_is_not_a_scan(tmp_path):
    (tmp_path / 'CellA.ptu').write_bytes(b'')
    ProjectFile.load_or_create(tmp_path).save()
    pf = ProjectFile.load_or_create(tmp_path)
    assert 'project' not in pf.scans
    assert set(pf.scans) == {'CellA'}

def test_stale_project_scan_is_dropped(tmp_path):
    import json
    (tmp_path / 'CellA.ptu').write_bytes(b'')
    pf = ProjectFile.load_or_create(tmp_path)
    pf.save()
    data = json.loads((tmp_path / 'project.json').read_text())
    data['scans']['project'] = {'stem': 'project', 'scan_type': 'fov',
                                'source_path': str(tmp_path / 'project.json')}
    (tmp_path / 'project.json').write_text(json.dumps(data))
    pf = ProjectFile.load_or_create(tmp_path)
    assert 'project' not in pf.scans

def test_other_json_is_not_a_scan(tmp_path):
    (tmp_path / 'notes.json').write_text('{"a": 1}')
    pf = ProjectFile.load_or_create(tmp_path)
    assert 'notes' not in pf.scans
