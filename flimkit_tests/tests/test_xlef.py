import os
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

import pytest

from flimkit.utils.xml_utils import (
    list_xlef_references,
    parse_tile_positions,
    resolve_xlef,
)

REAL = os.environ.get('FLIMKIT_TEST_XLEF', '')


def write_xlef(path, names):
    root = ET.Element('LMSDataContainerHeader')
    element = ET.SubElement(root, 'Element', Name=path.name)
    children = ET.SubElement(element, 'Children')
    for name in names:
        ET.SubElement(children, 'Reference',
                      File=quote('.\\metadata\\' + name + '.xlif', safe=''))
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return path


def lay_out(root, names, present=None):
    root.mkdir(parents=True, exist_ok=True)
    xlef = write_xlef(root / 'project.xlef', names)
    metadata = root / 'metadata'
    metadata.mkdir(exist_ok=True)
    for name in (names if present is None else present):
        (metadata / f'{name}.xlif').write_text('<root/>')
    return xlef


def test_references_are_decoded_from_windows_paths(tmp_path):
    xlef = lay_out(tmp_path, ['R 2', 'ctrl_1'])
    found = list_xlef_references(xlef)
    assert [entry['name'] for entry in found] == ['R 2', 'ctrl_1']
    assert all(entry['path'].exists() for entry in found)


def test_the_named_acquisition_is_chosen(tmp_path):
    xlef = lay_out(tmp_path, ['ctrl_1', 'R 2'])
    assert resolve_xlef(xlef, 'R 2').stem == 'R 2'


def test_the_name_match_ignores_spacing_and_case(tmp_path):
    xlef = lay_out(tmp_path, ['ctrl_1', 'R_2'])
    assert resolve_xlef(xlef, 'r 2').stem == 'R_2'


def test_a_single_acquisition_needs_no_name(tmp_path):
    xlef = lay_out(tmp_path, ['only_one'])
    assert resolve_xlef(xlef).stem == 'only_one'


def test_an_unknown_name_lists_what_is_there(tmp_path):
    xlef = lay_out(tmp_path, ['ctrl_1', 'R 2'])
    with pytest.raises(RuntimeError) as raised:
        resolve_xlef(xlef, 'missing')
    message = str(raised.value)
    assert 'ctrl_1' in message and 'R 2' in message


def test_references_that_are_not_on_disk_are_reported(tmp_path):
    xlef = lay_out(tmp_path, ['ctrl_1', 'R 2'], present=[])
    with pytest.raises(RuntimeError) as raised:
        resolve_xlef(xlef, 'R 2')
    assert 'none of the' in str(raised.value)


def test_tile_positions_go_through_the_referenced_xlif(tmp_path):
    xlef = lay_out(tmp_path, ['R 2'])
    with pytest.raises(RuntimeError) as raised:
        parse_tile_positions(xlef, 'R 2')
    assert 'TileScanInfo' in str(raised.value), (
        'the error should come from the .xlif it resolved to, not the .xlef')


@pytest.mark.skipif(not REAL or not Path(REAL).exists(),
                    reason='set FLIMKIT_TEST_XLEF to a Leica .xlef project')
def test_a_real_project_lists_its_acquisitions():
    found = list_xlef_references(REAL)
    assert found
    assert all(entry['path'].suffix == '.xlif' for entry in found)
