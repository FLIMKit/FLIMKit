import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

TILESCAN_ATTACHMENT = "./Data/Image/Attachment[@Name='TileScanInfo']"
DIMENSION_X = "./Data/Image/ImageDescription/Dimensions/DimensionDescription[@DimID='1']"

def _tiles_from_tilescan(tile_scan_info, ptu_basename: str) -> List[Dict[str, Any]]:
    tile_positions = []
    for scan_index, tile_elem in enumerate(tile_scan_info.findall('Tile')):
        field_x = int(tile_elem.attrib.get('FieldX', 0))
        field_y = int(tile_elem.attrib.get('FieldY', 0))
        pos_x = float(tile_elem.attrib.get('PosX', 0))
        pos_y = float(tile_elem.attrib.get('PosY', 0))
        filename = f'{ptu_basename}_s{scan_index + 1}.ptu'
        tile_positions.append({
            'file': filename,
            'scan_index': scan_index,
            'field_x': field_x,
            'field_y': field_y,
            'pos_x': pos_x,
            'pos_y': pos_y,
        })
    return tile_positions

def parse_xlif_tile_positions(xlif_path: Path, ptu_basename: str = 'R 2') -> List[Dict[str, Any]]:
    tree = ET.parse(xlif_path)
    root = tree.getroot()
    tile_scan_info = root.find(".//Attachment[@Name='TileScanInfo']")
    if tile_scan_info is None:
        raise RuntimeError(f"No TileScanInfo found in {xlif_path}")
    return _tiles_from_tilescan(tile_scan_info, ptu_basename)

def _lif_root(lif_path: Path):
    import liffile
    with liffile.LifFile(str(lif_path)) as lif:
        return lif.xml_element

def list_lif_mosaics(lif_path: Path) -> List[Dict[str, Any]]:
    root = _lif_root(lif_path)
    parents = {child: parent for parent in root.iter() for child in parent}
    mosaics = []
    for elem in root.iter('Element'):
        tile_scan_info = elem.find(TILESCAN_ATTACHMENT)
        if tile_scan_info is None:
            continue
        tiles = tile_scan_info.findall('Tile')
        if len(tiles) < 2:
            continue
        depth = 0
        node = elem
        while node in parents:
            node = parents[node]
            depth += 1
        mosaics.append({
            'name': elem.attrib.get('Name', ''),
            'n_tiles': len(tiles),
            'depth': depth,
            'element': elem,
        })
    mosaics.sort(key=lambda m: (m['depth'], m['name']))
    return mosaics

def _select_lif_mosaic(lif_path: Path, ptu_basename: str):
    mosaics = list_lif_mosaics(lif_path)
    if not mosaics:
        raise RuntimeError(f'No tilescan (multi-tile TileScanInfo) found in {lif_path}')
    named = [m for m in mosaics if m['name'] == ptu_basename]
    if not named:
        available = ', '.join(f"{m['name']!r} ({m['n_tiles']} tiles)" for m in mosaics)
        raise RuntimeError(
            f'No tilescan named {ptu_basename!r} in {lif_path}. Available: {available}')
    return named[0]['element']

def parse_lif_tile_positions(lif_path: Path, ptu_basename: str) -> List[Dict[str, Any]]:
    element = _select_lif_mosaic(lif_path, ptu_basename)
    return _tiles_from_tilescan(element.find(TILESCAN_ATTACHMENT), ptu_basename)

def get_pixel_size_from_lif(lif_path: Path, ptu_basename: str) -> Tuple[float, int]:
    element = _select_lif_mosaic(lif_path, ptu_basename)
    dim_desc = element.find(DIMENSION_X)
    if dim_desc is None:
        dim_desc = element.find(".//DimensionDescription[@DimID='1']")
    if dim_desc is not None:
        n_pixels = int(dim_desc.attrib.get('NumberOfElements', 512))
        length_m = float(dim_desc.attrib.get('Length', 1.5377e-4))
        return length_m / n_pixels, n_pixels
    return 1.5377e-4 / 512, 512

def list_xlef_references(xlef_path: Path) -> List[Dict[str, Any]]:
    from urllib.parse import unquote
    xlef_path = Path(xlef_path)
    root = ET.parse(xlef_path).getroot()
    found = []
    for reference in root.iter('Reference'):
        raw = reference.attrib.get('File')
        if not raw:
            continue
        relative = unquote(raw).replace('\\', '/')
        target = (xlef_path.parent / relative).resolve()
        if target.suffix.lower() != '.xlif':
            continue
        found.append({'name': target.stem, 'path': target})
    return found

def _matches_basename(name: str, ptu_basename: str) -> bool:
    def flatten(value):
        return value.lower().replace(' ', '').replace('_', '').replace('-', '')
    return flatten(name) == flatten(ptu_basename)

def resolve_xlef(xlef_path: Path, ptu_basename: str = None) -> Path:
    xlef_path = Path(xlef_path)
    references = list_xlef_references(xlef_path)
    present = [entry for entry in references if entry['path'].exists()]
    if not present:
        raise RuntimeError(
            f'{xlef_path.name} names {len(references)} acquisitions but none of the '
            f'.xlif files are in {xlef_path.parent}')
    if ptu_basename:
        for entry in present:
            if _matches_basename(entry['name'], ptu_basename):
                return entry['path']
    if len(present) == 1:
        return present[0]['path']
    names = ', '.join(entry['name'] for entry in present)
    raise RuntimeError(
        f'{xlef_path.name} holds {len(present)} acquisitions and '
        f'{ptu_basename!r} is not one of them; choose from {names}')

def parse_tile_positions(metadata_path: Path, ptu_basename: str) -> List[Dict[str, Any]]:
    metadata_path = Path(metadata_path)
    if metadata_path.suffix.lower() == '.lif':
        return parse_lif_tile_positions(metadata_path, ptu_basename)
    if metadata_path.suffix.lower() == '.xlef':
        return parse_xlif_tile_positions(
            resolve_xlef(metadata_path, ptu_basename), ptu_basename)
    return parse_xlif_tile_positions(metadata_path, ptu_basename)

def get_pixel_size(metadata_path: Path, ptu_basename: str = None) -> Tuple[float, int]:
    metadata_path = Path(metadata_path)
    if metadata_path.suffix.lower() == '.lif':
        return get_pixel_size_from_lif(metadata_path, ptu_basename)
    if metadata_path.suffix.lower() == '.xlef':
        return get_pixel_size_from_xlif(resolve_xlef(metadata_path, ptu_basename))
    return get_pixel_size_from_xlif(metadata_path)

def get_pixel_size_from_xlif(xlif_path: Path) -> Tuple[float, int]:
    tree = ET.parse(xlif_path)
    root = tree.getroot()
    dim_desc = root.find(".//DimensionDescription[@DimID='1']")
    if dim_desc is not None:
        n_pixels = int(dim_desc.attrib.get('NumberOfElements', 512))
        length_m = float(dim_desc.attrib.get('Length', 1.5377e-4))
        pixel_size_m = length_m / n_pixels
        return pixel_size_m, n_pixels
    return 1.5377e-4 / 512, 512

def compute_tile_pixel_positions(
    tile_positions: List[Dict[str, Any]],
    pixel_size_m: float,
    tile_size: int
) -> Tuple[List[Dict[str, Any]], int, int]:
    pos_x_list = [t['pos_x'] for t in tile_positions]
    pos_y_list = [t['pos_y'] for t in tile_positions]
    min_pos_x = min(pos_x_list)
    min_pos_y = min(pos_y_list)
    for t in tile_positions:
        t['pixel_x'] = int(round((t['pos_x'] - min_pos_x) / pixel_size_m))
        t['pixel_y'] = int(round((t['pos_y'] - min_pos_y) / pixel_size_m))
    canvas_width = max(t['pixel_x'] for t in tile_positions) + tile_size
    canvas_height = max(t['pixel_y'] for t in tile_positions) + tile_size
    return tile_positions, canvas_width, canvas_height

def match_xml_ptu_sets(ptu_dir: Path) -> List[Dict[str, Any]]:
    metadata_dir = ptu_dir / 'Metadata'    
    xml_files = []
    if metadata_dir.exists():
        xml_files = (list(metadata_dir.glob('*.xlif')) + 
                    list(metadata_dir.glob('*.xlof')) + 
                    list(metadata_dir.glob('*.xml')))
    ptu_files = list(ptu_dir.glob('*.ptu'))
    r_pattern = re.compile(r'R\s*\d+')
    xml_r_map = {}
    for xml in xml_files:
        m = r_pattern.search(xml.name)
        if m:
            r = m.group().replace(' ', '')
            xml_r_map.setdefault(r, []).append(str(xml))
    ptu_r_map = {}
    for ptu in ptu_files:
        m = r_pattern.search(ptu.name)
        if m:
            r = m.group().replace(' ', '')
            ptu_r_map.setdefault(r, []).append(str(ptu))
    results = []
    all_r_numbers = sorted(set(xml_r_map) | set(ptu_r_map))
    for r in all_r_numbers:
        xmls = xml_r_map.get(r, [])
        ptus = ptu_r_map.get(r, [])
        if xmls and ptus:
            status = 'MATCHED'
        elif ptus:
            status = 'MISSING_XML'
        else:
            status = 'MISSING_PTU'
        results.append({
            'R_number': r,
            'xml_files': xmls,
            'ptu_files': ptus,
            'xml_count': len(xmls),
            'ptu_count': len(ptus),
            'status': status
        })
    return results

def extract_roi_number(filename: str) -> Optional[int]:
    m = re.search(r'R\s*(\d+)', filename)
    return int(m.group(1)) if m else None
