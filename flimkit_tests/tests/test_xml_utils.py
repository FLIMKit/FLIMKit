import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mock_data import generate_mock_xlif

class TestXMLUtils:
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    @pytest.fixture
    def mock_xlif_2x2(self, temp_dir):
        xlif_path = generate_mock_xlif(
            temp_dir / "test.xlif",
            n_tiles=4,
            layout="2x2",
            pixel_size_m=3e-7
        )
        return xlif_path
    @pytest.fixture
    def mock_xlif_1x4(self, temp_dir):
        xlif_path = generate_mock_xlif(
            temp_dir / "test_1x4.xlif",
            n_tiles=4,
            layout="1x4",
            pixel_size_m=3e-7
        )
        return xlif_path
    def test_parse_xlif_tile_positions(self, mock_xlif_2x2):
        from flimkit.utils.xml_utils import parse_xlif_tile_positions
        tiles = parse_xlif_tile_positions(mock_xlif_2x2, ptu_basename="R 2")
        assert len(tiles) == 4
        for tile in tiles:
            assert 'file' in tile
            assert 'field_x' in tile
            assert 'pos_x' in tile
            assert 'pos_y' in tile
        filenames = [t['file'] for t in tiles]
        expected = ['R 2_s1.ptu', 'R 2_s2.ptu', 'R 2_s3.ptu', 'R 2_s4.ptu']
        assert filenames == expected
        for tile in tiles:
            assert isinstance(tile['pos_x'], float)
            assert isinstance(tile['pos_y'], float)
    def test_get_pixel_size_from_xlif(self, mock_xlif_2x2):
        from flimkit.utils.xml_utils import get_pixel_size_from_xlif
        pixel_size_m, n_pixels = get_pixel_size_from_xlif(mock_xlif_2x2)
        assert isinstance(pixel_size_m, float)
        assert isinstance(n_pixels, int)
        assert pixel_size_m > 0
        assert n_pixels > 0
        assert abs(pixel_size_m - 3e-7) < 1e-9
        assert n_pixels == 512
    def test_compute_tile_pixel_positions(self, mock_xlif_2x2):
        from flimkit.utils.xml_utils import (
            parse_xlif_tile_positions,
            get_pixel_size_from_xlif,
            compute_tile_pixel_positions
        )
        tiles = parse_xlif_tile_positions(mock_xlif_2x2, "R 2")
        pixel_size_m, _ = get_pixel_size_from_xlif(mock_xlif_2x2)
        tiles, canvas_width, canvas_height = compute_tile_pixel_positions(
            tiles, pixel_size_m, tile_size=512
        )
        assert canvas_width == 1024
        assert canvas_height == 1024
        for tile in tiles:
            assert 'pixel_x' in tile
            assert 'pixel_y' in tile
            assert isinstance(tile['pixel_x'], int)
            assert isinstance(tile['pixel_y'], int)
        for tile in tiles:
            assert 0 <= tile['pixel_x'] < canvas_width
            assert 0 <= tile['pixel_y'] < canvas_height
    def test_extract_roi_number(self):
        from flimkit.utils.xml_utils import extract_roi_number
        assert extract_roi_number("R 2_s1.ptu") == 2
        assert extract_roi_number("R 10_s3.ptu") == 10
        assert extract_roi_number("R123_tile5.ptu") == 123
        assert extract_roi_number("no_roi_here.ptu") is None
    def test_match_xml_ptu_sets(self, temp_dir):
        from flimkit.utils.xml_utils import match_xml_ptu_sets
        metadata_dir = temp_dir / "Metadata"
        metadata_dir.mkdir()
        generate_mock_xlif(metadata_dir / "R 2.xlif", n_tiles=4)
        generate_mock_xlif(metadata_dir / "R 3.xlif", n_tiles=4)
        (temp_dir / "R 2_s1.ptu").touch()
        (temp_dir / "R 2_s2.ptu").touch()
        (temp_dir / "R 3_s1.ptu").touch()
        matches = match_xml_ptu_sets(temp_dir)
        assert len(matches) == 2
        for match in matches:
            assert 'R_number' in match
            assert 'xml_files' in match
            assert 'ptu_files' in match
            assert 'status' in match
        r2_match = [m for m in matches if m['R_number'] == 'R2'][0]
        assert r2_match['status'] == 'MATCHED'
        assert r2_match['ptu_count'] == 2
        assert r2_match['xml_count'] == 1

def test_xlif_with_missing_tilescan():
    from flimkit.utils.xml_utils import parse_xlif_tile_positions
    import xml.etree.ElementTree as ET
    with tempfile.TemporaryDirectory() as temp_dir:
        root = ET.Element("LMSDataContainerHeader")
        tree = ET.ElementTree(root)
        xlif_path = Path(temp_dir) / "bad.xlif"
        tree.write(xlif_path)
        with pytest.raises(RuntimeError, match="No TileScanInfo"):
            parse_xlif_tile_positions(xlif_path, "R 2")

def test_xlif_different_layouts():
    from flimkit.utils.xml_utils import parse_xlif_tile_positions
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        xlif_3x3 = generate_mock_xlif(temp_path / "3x3.xlif", n_tiles=9, layout="3x3")
        tiles = parse_xlif_tile_positions(xlif_3x3, "R 2")
        assert len(tiles) == 9
        xlif_1x5 = generate_mock_xlif(temp_path / "1x5.xlif", n_tiles=5, layout="1x5")
        tiles = parse_xlif_tile_positions(xlif_1x5, "R 2")
        assert len(tiles) == 5

def _lif_tree(mosaics, length_m=2.9e-4, n_pixels=512):
    import xml.etree.ElementTree as ET
    root = ET.Element('LMSDataContainerHeader')
    top = ET.SubElement(root, 'Element', {'Name': 'project'})
    children = ET.SubElement(top, 'Children')
    for name, tiles, nest in mosaics:
        node = children
        for level in nest:
            node = ET.SubElement(node, 'Element', {'Name': level})
            node = ET.SubElement(node, 'Children')
        elem = ET.SubElement(node, 'Element', {'Name': name})
        data = ET.SubElement(elem, 'Data')
        image = ET.SubElement(data, 'Image')
        desc = ET.SubElement(image, 'ImageDescription')
        dims = ET.SubElement(desc, 'Dimensions')
        ET.SubElement(dims, 'DimensionDescription', {
            'DimID': '1', 'NumberOfElements': str(n_pixels),
            'Length': str(length_m), 'Unit': 'm',
        })
        tsi = ET.SubElement(image, 'Attachment', {'Name': 'TileScanInfo'})
        for fx, fy in tiles:
            ET.SubElement(tsi, 'Tile', {
                'FieldX': str(fx), 'FieldY': str(fy),
                'PosX': str(fx * 1e-4), 'PosY': str(fy * 1e-4), 'PosZ': '0',
            })
    return root

def test_get_pixel_size_from_lif(monkeypatch):
    import flimkit.utils.xml_utils as xu
    tree = _lif_tree([('R1', [(i, 0) for i in range(4)], [])],
                     length_m=2.9e-4, n_pixels=512)
    monkeypatch.setattr(xu, '_lif_root', lambda p: tree)
    pixel_size_m, n_pixels = xu.get_pixel_size_from_lif(Path('fake.lif'), 'R1')
    assert n_pixels == 512
    assert abs(pixel_size_m - 2.9e-4 / 512) < 1e-15

def test_dispatcher_routes_on_suffix(monkeypatch):
    import flimkit.utils.xml_utils as xu
    tree = _lif_tree([('R1', [(i, 0) for i in range(4)], [])])
    monkeypatch.setattr(xu, '_lif_root', lambda p: tree)
    tiles = xu.parse_tile_positions(Path('anything.lif'), 'R1')
    assert [t['file'] for t in tiles] == [f'R1_s{i + 1}.ptu' for i in range(4)]
    assert xu.get_pixel_size(Path('anything.lif'), 'R1')[1] == 512
    with tempfile.TemporaryDirectory() as temp_dir:
        xlif = generate_mock_xlif(Path(temp_dir) / 'R 2.xlif', n_tiles=4, layout='2x2')
        assert len(xu.parse_tile_positions(xlif, 'R 2')) == 4
        assert xu.get_pixel_size(xlif)[1] == 512

def test_tile_positions_include_field_y():
    from flimkit.utils.xml_utils import parse_xlif_tile_positions
    with tempfile.TemporaryDirectory() as temp_dir:
        xlif = generate_mock_xlif(Path(temp_dir) / 'x.xlif', n_tiles=4, layout='2x2')
        tiles = parse_xlif_tile_positions(xlif, 'R 2')
        assert all('field_y' in t for t in tiles)

def test_scan_index_drives_filenames_not_field_x(monkeypatch):
    import flimkit.utils.xml_utils as xu
    serpentine = []
    for y in range(4):
        xs = range(3) if y % 2 == 0 else reversed(range(3))
        serpentine += [(x, y) for x in xs]
    monkeypatch.setattr(xu, '_lif_root', lambda p: _lif_tree([('R1', serpentine, [])]))
    tiles = xu.parse_lif_tile_positions(Path('fake.lif'), 'R1')
    assert len(tiles) == 12
    assert len({t['file'] for t in tiles}) == 12
    assert [t['file'] for t in tiles[:4]] == [
        'R1_s1.ptu', 'R1_s2.ptu', 'R1_s3.ptu', 'R1_s4.ptu']
    assert (tiles[3]['field_x'], tiles[3]['field_y']) == (2, 1)
    assert [t['scan_index'] for t in tiles] == list(range(12))

def test_flat_grid_filenames_unchanged_by_scan_index(monkeypatch):
    import flimkit.utils.xml_utils as xu
    flat = [(i, 0) for i in range(6)]
    monkeypatch.setattr(xu, '_lif_root', lambda p: _lif_tree([('R1', flat, [])]))
    tiles = xu.parse_lif_tile_positions(Path('fake.lif'), 'R1')
    assert [t['file'] for t in tiles] == [f'R1_s{i + 1}.ptu' for i in range(6)]
    assert all(t['scan_index'] == t['field_x'] for t in tiles)

def test_lif_name_matching_prefers_parent_over_channel_children(monkeypatch):
    import flimkit.utils.xml_utils as xu
    parent = [(i, 0) for i in range(9)]
    child = [(i, 0) for i in range(9)]
    single = [(0, 0)]
    tree = _lif_tree([
        ('FOV_a', single, []),
        ('Intensity', child, ['R1', 'FLIM']),
        ('R1', parent, []),
    ])
    monkeypatch.setattr(xu, '_lif_root', lambda p: tree)
    mosaics = xu.list_lif_mosaics(Path('fake.lif'))
    names = [m['name'] for m in mosaics]
    assert 'FOV_a' not in names
    assert mosaics[0]['name'] == 'R1'
    tiles = xu.parse_lif_tile_positions(Path('fake.lif'), 'R1')
    assert len(tiles) == 9
    assert tiles[0]['file'] == 'R1_s1.ptu'

def test_lif_unknown_name_lists_alternatives(monkeypatch):
    import flimkit.utils.xml_utils as xu
    tree = _lif_tree([('R1', [(i, 0) for i in range(4)], [])])
    monkeypatch.setattr(xu, '_lif_root', lambda p: tree)
    with pytest.raises(RuntimeError, match="'R1' \\(4 tiles\\)"):
        xu.parse_lif_tile_positions(Path('fake.lif'), 'Nope')

def test_lif_without_any_mosaic_raises(monkeypatch):
    import flimkit.utils.xml_utils as xu
    tree = _lif_tree([('FOV_a', [(0, 0)], []), ('FOV_b', [(0, 0)], [])])
    monkeypatch.setattr(xu, '_lif_root', lambda p: tree)
    with pytest.raises(RuntimeError, match='No tilescan'):
        xu.parse_lif_tile_positions(Path('fake.lif'), 'FOV_a')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
