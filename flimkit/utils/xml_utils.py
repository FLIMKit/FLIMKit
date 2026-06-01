import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


def parse_xlif_tile_positions(xlif_path: Path, ptu_basename: str = 'R 2') -> List[Dict[str, Any]]:
    tree = ET.parse(xlif_path)
    root = tree.getroot()
    
    tile_scan_info = root.find(".//Attachment[@Name='TileScanInfo']")
    if tile_scan_info is None:
        raise RuntimeError(f"No TileScanInfo found in {xlif_path}")
    
    tile_positions = []
    for tile_elem in tile_scan_info.findall('Tile'):
        field_x = int(tile_elem.attrib.get('FieldX', 0))
        pos_x = float(tile_elem.attrib.get('PosX', 0))   # meters (raw from XML)
        pos_y = float(tile_elem.attrib.get('PosY', 0))   # meters (raw from XML)
        filename = f"{ptu_basename}_s{field_x + 1}.ptu"
        tile_positions.append({
            'file': filename,
            'field_x': field_x,
            'pos_x': pos_x,
            'pos_y': pos_y,
        })
    
    return tile_positions


def get_pixel_size_from_xlif(xlif_path: Path) -> Tuple[float, int]:
    tree = ET.parse(xlif_path)
    root = tree.getroot()
    
    dim_desc = root.find(".//DimensionDescription[@DimID='1']")
    if dim_desc is not None:
        n_pixels = int(dim_desc.attrib.get('NumberOfElements', 512))
        length_m = float(dim_desc.attrib.get('Length', 1.5377e-4))
        pixel_size_m = length_m / n_pixels          # meters
        return pixel_size_m, n_pixels
    
    # Fallback defaults (meters)
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
    
    # Find all relevant files
    xml_files = []
    if metadata_dir.exists():
        xml_files = (list(metadata_dir.glob('*.xlif')) + 
                    list(metadata_dir.glob('*.xlof')) + 
                    list(metadata_dir.glob('*.xml')))
    
    ptu_files = list(ptu_dir.glob('*.ptu'))
    
    # Extract R numbers (e.g., "R 2" → "R2")
    r_pattern = re.compile(r'R\s*\d+')
    
    # Group XML files by R number
    xml_r_map = {}
    for xml in xml_files:
        m = r_pattern.search(xml.name)
        if m:
            r = m.group().replace(' ', '')
            xml_r_map.setdefault(r, []).append(str(xml))
    
    # Group PTU files by R number
    ptu_r_map = {}
    for ptu in ptu_files:
        m = r_pattern.search(ptu.name)
        if m:
            r = m.group().replace(' ', '')
            ptu_r_map.setdefault(r, []).append(str(ptu))
    
    # Match and create result list
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
