import copy
import json
from typing import Any, List, Dict, Optional, Tuple
import numpy as np

_COLORS = [
    '#FF6B6B',
    '#4ECDC4',
    '#FFE66D',
    '#95E1D3',
    '#C7CEEA',
    '#FF8C42',
]

class RoiManager:

    def __init__(self):
        self.regions: List[Dict] = []
        self._next_id = 0
        self._selected_id: Optional[int] = None

    def add_region(self, name: str, tool_type: str, coords: List[List[float]],
                   color_idx: Optional[int] = None) -> int:
        if tool_type not in ('rect', 'ellipse', 'polygon', 'freehand'):
            raise ValueError(f'Invalid tool_type: {tool_type}')
        if not coords or len(coords) == 0:
            raise ValueError('coords cannot be empty')
        if color_idx is None:
            color_idx = len(self.regions) % len(_COLORS)
        region = {
            'id': self._next_id,
            'name': name,
            'tool': tool_type,
            'coords': [[float(x), float(y)] for x, y in coords],
            'color_idx': int(color_idx),
        }
        self.regions.append(region)
        self._next_id += 1
        return region['id']

    def remove_region(self, region_id: int) -> bool:
        for i, r in enumerate(self.regions):
            if r['id'] == region_id:
                self.regions.pop(i)
                if self._selected_id == region_id:
                    self._selected_id = None
                return True
        return False

    def get_region(self, region_id: int) -> Optional[Dict]:
        for r in self.regions:
            if r['id'] == region_id:
                return r
        return None

    def update_region(self, region_id: int, **kwargs) -> bool:
        for r in self.regions:
            if r['id'] == region_id:
                if 'coords' in kwargs:
                    r['coords'] = [[float(x), float(y)] for x, y in kwargs['coords']]
                if 'name' in kwargs:
                    r['name'] = str(kwargs['name'])
                if 'color_idx' in kwargs:
                    r['color_idx'] = int(kwargs['color_idx'])
                if 'tool' in kwargs:
                    r['tool'] = str(kwargs['tool'])
                return True
        return False

    def select_region(self, region_id: Optional[int]) -> None:
        self._selected_id = region_id

    def get_selected_id(self) -> Optional[int]:
        return self._selected_id

    def get_all_regions(self) -> List[Dict]:
        return self.regions

    def clear_all(self) -> None:
        self.regions = []
        self._selected_id = None
        self._next_id = 0

    def to_json(self) -> str:
        data = {
            'regions': self.regions,
            'next_id': self._next_id,
        }
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'RoiManager':
        manager = cls()
        try:
            data = json.loads(json_str)
            manager.regions = data.get('regions', [])
            manager._next_id = data.get('next_id', len(manager.regions))
        except (json.JSONDecodeError, ValueError):
            pass
        return manager

    @staticmethod
    def _geojson_coordinates(values: Any, minimum: int) -> List[List[float]]:
        if not isinstance(values, list):
            raise ValueError('GeoJSON coordinates must be a list')
        coords = []
        for point in values:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise ValueError('GeoJSON coordinates must contain [x, y] points')
            x, y = float(point[0]), float(point[1])
            if not np.isfinite(x) or not np.isfinite(y):
                raise ValueError('GeoJSON coordinates must be finite')
            coords.append([x, y])
        if len(coords) < minimum:
            raise ValueError(f'GeoJSON geometry requires at least {minimum} points')
        return coords

    @staticmethod
    def _outer_boundary(ring: List[List[float]]) -> Optional[List[List[float]]]:
        from shapely.geometry import MultiPolygon, Polygon
        polygon = Polygon(ring)
        if polygon.is_valid:
            return None
        repaired = polygon.buffer(0)
        if isinstance(repaired, MultiPolygon):
            if not repaired.geoms:
                return None
            repaired = max(repaired.geoms, key=lambda part: part.area)
        if repaired.is_empty or repaired.geom_type != 'Polygon':
            return None
        out = [[float(x), float(y)] for x, y in repaired.exterior.coords]
        if len(out) < 4:
            return None
        if out[-1] != out[0]:
            out.append(out[0][:])
        return out

    @staticmethod
    def _region_feature(region: Dict) -> Dict:
        tool = region['tool']
        coords = [[float(x), float(y)] for x, y in region['coords']]
        properties = {
            'id': region.get('id'),
            'name': region.get('name', ''),
            'tool_type': tool,
            'color_idx': region.get('color_idx', 0),
        }
        statistic_keys = (
            'tau_median', 'tau_stdev', 'photon_count', 'photon_stdev',
        )
        statistics = {
            key: region.get('statistics', {}).get(key)
            for key in statistic_keys
            if region.get('statistics', {}).get(key) is not None
        }
        properties['statistics'] = statistics
        if tool == 'rect':
            if len(coords) != 2:
                raise ValueError('Rectangle regions require two corner points')
            (x1, y1), (x2, y2) = coords
            ring = [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1],
            ]
            properties['bounds'] = coords
        elif tool == 'ellipse':
            if len(coords) != 2:
                raise ValueError('Ellipse regions require two corner points')
            (x1, y1), (x2, y2) = coords
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = abs(x2 - x1) / 2, abs(y2 - y1) / 2
            angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
            ring = [
                [float(cx + rx * np.cos(angle)),
                 float(cy + ry * np.sin(angle))]
                for angle in angles
            ]
            ring.append(ring[0].copy())
            properties['bounds'] = coords
        elif tool in ('polygon', 'freehand'):
            if len(coords) < 3:
                raise ValueError(f'{tool.capitalize()} regions require at least three points')
            ring = coords.copy()
            if ring[-1] != ring[0]:
                ring.append(ring[0].copy())
            outer = RoiManager._outer_boundary(ring)
            if outer is not None:
                ring = outer
                properties['repaired'] = 'self-intersecting'
        else:
            raise ValueError(f'Unsupported ROI tool: {tool}')
        return {
            'type': 'Feature',
            'properties': properties,
            'geometry': {'type': 'Polygon', 'coordinates': [ring]},
        }

    def to_geojson(self, region_ids: Optional[List[int]] = None) -> Dict:
        if region_ids is None:
            regions = self.regions
        else:
            wanted = set(region_ids)
            regions = [region for region in self.regions if region['id'] in wanted]
            found = {region['id'] for region in regions}
            missing = wanted - found
            if missing:
                raise ValueError(f'Region IDs not found: {sorted(missing)}')
        return {
            'type': 'FeatureCollection',
            'features': [self._region_feature(region) for region in regions],
            'flimkit': {
                'coordinate_system': 'image-pixel',
                'axis_order': 'xy',
                'origin': 'top-left',
            },
        }

    @classmethod
    def _region_from_feature(cls, feature: Dict) -> Dict:
        if not isinstance(feature, dict) or feature.get('type') != 'Feature':
            raise ValueError('GeoJSON entries must be Features')
        properties = feature.get('properties') or {}
        geometry = feature.get('geometry') or {}
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            raise ValueError('GeoJSON Feature properties and geometry must be objects')
        geometry_type = geometry.get('type')
        raw_coordinates = geometry.get('coordinates')
        tool = properties.get('tool_type')
        if tool is None:
            if geometry_type == 'Polygon':
                tool = 'polygon'
            elif geometry_type == 'LineString':
                tool = 'freehand'
        if tool not in ('rect', 'ellipse', 'polygon', 'freehand'):
            raise ValueError(f'Unsupported GeoJSON geometry: {geometry_type}')
        if geometry_type == 'Polygon':
            if not isinstance(raw_coordinates, list) or not raw_coordinates:
                raise ValueError('GeoJSON Polygon must contain an outer ring')
            outer_ring = raw_coordinates[0]
            if isinstance(outer_ring, list) and len(outer_ring) < 4:
                raise ValueError(
                    'GeoJSON Polygon outer rings require at least four '
                    'positions, including a repeated closing position',
                )
            ring = cls._geojson_coordinates(outer_ring, 4)
            if ring[0] == ring[-1]:
                ring = ring[:-1]
            if tool in ('rect', 'ellipse'):
                bounds = properties.get('bounds')
                if bounds is not None:
                    coords = cls._geojson_coordinates(bounds, 2)
                    if len(coords) != 2:
                        raise ValueError('GeoJSON ROI bounds require two corner points')
                else:
                    xs = [point[0] for point in ring]
                    ys = [point[1] for point in ring]
                    coords = [[min(xs), min(ys)], [max(xs), max(ys)]]
            else:
                coords = cls._geojson_coordinates(ring, 3)
        elif geometry_type == 'LineString' and tool in ('polygon', 'freehand'):
            coords = cls._geojson_coordinates(raw_coordinates, 3)
        else:
            raise ValueError(f'Unsupported GeoJSON geometry: {geometry_type}')
        statistics = properties.get('statistics')
        if isinstance(statistics, dict):
            statistics = copy.deepcopy(statistics)
        else:
            statistics = {}
        for key in ('tau_median', 'tau_stdev', 'photon_count', 'photon_stdev'):
            if key not in statistics and properties.get(key) is not None:
                statistics[key] = properties[key]
        color_idx = properties.get('color_idx')
        return {
            'name': str(properties.get('name', 'imported-region')),
            'tool': tool,
            'coords': coords,
            'color_idx': int(color_idx) if color_idx is not None else None,
            'statistics': statistics,
        }

    def add_geojson(self, payload: Dict, mode: str = 'append') -> List[int]:
        if mode not in ('append', 'replace'):
            raise ValueError('mode must be append or replace')
        if not isinstance(payload, dict):
            raise ValueError('GeoJSON payload must be an object')
        if payload.get('type') == 'FeatureCollection':
            features = payload.get('features')
            if not isinstance(features, list):
                raise ValueError('GeoJSON FeatureCollection features must be a list')
        elif payload.get('type') == 'Feature':
            features = [payload]
        else:
            raise ValueError('GeoJSON must be a Feature or FeatureCollection')
        pending = [self._region_from_feature(feature) for feature in features]
        if mode == 'replace':
            self.clear_all()
        added = []
        for item in pending:
            region_id = self.add_region(
                item['name'], item['tool'], item['coords'], item['color_idx'],
            )
            region = self.get_region(region_id)
            if region is not None and item['statistics']:
                region['statistics'] = item['statistics']
            added.append(region_id)
        return added

    def compute_region_mask(self, region_id: int, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        from matplotlib.path import Path as MplPath
        region = self.get_region(region_id)
        if region is None:
            return None
        height, width = image_shape
        mask = np.zeros((height, width), dtype=bool)
        coords = np.array(region['coords'], dtype=float)
        if region['tool'] == 'rect':
            if len(coords) >= 2:
                x0, y0 = coords[0]
                x1, y1 = coords[1]
                x_min, x_max = int(min(x0, x1)), int(max(x0, x1))
                y_min, y_max = int(min(y0, y1)), int(max(y0, y1))
                mask[y_min:y_max+1, x_min:x_max+1] = True
        elif region['tool'] == 'ellipse':
            if len(coords) >= 2:
                x0, y0 = coords[0]
                x1, y1 = coords[1]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
                yy, xx = np.ogrid[:height, :width]
                mask = ((xx - cx)**2 / (rx**2 + 1e-6) +
                        (yy - cy)**2 / (ry**2 + 1e-6)) <= 1
        elif region['tool'] in ('polygon', 'freehand'):
            if len(coords) >= 3:
                path = MplPath(coords)
                yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                points = np.column_stack([xx.ravel(), yy.ravel()])
                mask = path.contains_points(points).reshape((height, width))
        return mask

    def get_color(self, region_id: int) -> str:
        region = self.get_region(region_id)
        if region is None:
            return '#999999'
        color_idx = region.get('color_idx', 0) % len(_COLORS)
        return _COLORS[color_idx]

    @staticmethod
    def get_color_palette() -> List[str]:
        return _COLORS.copy()

def get_rectangle_patch(coords, edgecolor, facecolor='none', linewidth=2):
    from matplotlib.patches import Rectangle
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    xy = (min(x0, x1), min(y0, y1))
    return Rectangle(xy, width, height, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)

def get_ellipse_patch(coords, edgecolor, facecolor='none', linewidth=2):
    from matplotlib.patches import Ellipse
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    return Ellipse((cx, cy), width, height, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)

def get_polygon_patch(coords, edgecolor, facecolor='none', linewidth=2):
    from matplotlib.patches import Polygon
    return Polygon(coords, edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, closed=True)
