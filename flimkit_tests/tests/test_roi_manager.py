import pytest
import numpy as np
from flimkit.UI.roi_tools import RoiManager


class TestRoiManager:
    @pytest.fixture
    def manager(self):
        return RoiManager()

    def test_add_rectangle(self, manager):
        coords = [[10, 20], [50, 60]]
        rid = manager.add_region("Test Rect", "rect", coords)
        assert manager.get_region(rid)['tool'] == "rect"

    def test_invalid_tool_type_raises(self, manager):
        with pytest.raises(ValueError):
            manager.add_region("Bad", "circle", [[0,0]])

    def test_empty_coords_raises(self, manager):
        with pytest.raises(ValueError):
            manager.add_region("Empty", "rect", [])

    def test_remove_region(self, manager):
        id1 = manager.add_region("R1", "rect", [[0,0],[10,10]])
        id2 = manager.add_region("R2", "rect", [[20,20],[30,30]])
        assert manager.remove_region(id1) is True
        assert len(manager.get_all_regions()) == 1

    def test_update_region(self, manager):
        rid = manager.add_region("Old", "rect", [[0,0],[10,10]])
        assert manager.update_region(rid, name="New")
        assert manager.get_region(rid)['name'] == "New"

    def test_serialize_deserialize(self, manager):
        manager.add_region("Rect1", "rect", [[10,20],[50,60]], color_idx=2)
        json_str = manager.to_json()
        manager2 = RoiManager.from_json(json_str)
        assert len(manager2.get_all_regions()) == 1

    def test_clear_all(self, manager):
        manager.add_region("R1", "rect", [[0,0],[10,10]])
        manager.clear_all()
        assert len(manager.get_all_regions()) == 0

    def test_compute_rectangle_mask(self, manager):
        manager.add_region("Rect", "rect", [[2,3],[5,7]])
        mask = manager.compute_region_mask(0, (10,10))
        assert mask[3,2]

    def test_compute_ellipse_mask(self, manager):
        manager.add_region("Ellipse", "ellipse", [[2,3],[6,7]])
        mask = manager.compute_region_mask(0, (10,10))
        assert mask[5,4]

    def test_get_color(self, manager):
        id1 = manager.add_region("R1", "rect", [[0,0],[1,1]], color_idx=0)
        id2 = manager.add_region("R2", "rect", [[0,0],[1,1]])
        colors = manager.get_color_palette()
        assert manager.get_color(id1) == colors[0]
        assert manager.get_color(id2) == colors[1]

    @pytest.mark.parametrize(('tool', 'coords'), [
        ('rect', [[2.25, 3.5], [8.75, 9.25]]),
        ('ellipse', [[3.0, 2.0], [11.0, 10.0]]),
        ('polygon', [[2.0, 2.0], [10.0, 3.0], [7.0, 12.0]]),
        ('freehand', [[3.0, 3.0], [9.0, 2.0], [11.0, 8.0], [4.0, 11.0]]),
    ])
    def test_geojson_round_trip_preserves_region_and_mask(self, tool, coords):
        manager = RoiManager()
        region_id = manager.add_region('Cell 1', tool, coords, color_idx=3)
        region = manager.get_region(region_id)
        assert region is not None
        region['statistics'] = {
            'tau_median': 2.75,
            'photon_count': 1234,
        }
        expected_mask = manager.compute_region_mask(region_id, (16, 16))

        payload = manager.to_geojson()
        properties = payload['features'][0]['properties']
        assert properties['statistics'] == {
            'tau_median': 2.75,
            'photon_count': 1234,
        }
        assert 'tau_median' not in properties
        assert 'photon_count' not in properties
        restored = RoiManager()
        restored_ids = restored.add_geojson(payload)

        assert len(restored_ids) == 1
        region = restored.get_region(restored_ids[0])
        assert region is not None
        assert region['name'] == 'Cell 1'
        assert region['tool'] == tool
        assert region['coords'] == coords
        assert region['color_idx'] == 3
        assert region['statistics'] == {
            'tau_median': 2.75,
            'photon_count': 1234,
        }
        np.testing.assert_array_equal(
            restored.compute_region_mask(restored_ids[0], (16, 16)),
            expected_mask,
        )

    def test_geojson_polygon_without_flimkit_metadata_imports_as_polygon(self, manager):
        payload = {
            'type': 'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'properties': {'name': 'Fiji ROI'},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[
                        [1.25, 2.5], [8.5, 2.5], [4.0, 9.75], [1.25, 2.5],
                    ]],
                },
            }],
        }

        restored_ids = manager.add_geojson(payload)

        region = manager.get_region(restored_ids[0])
        assert region is not None
        assert region['name'] == 'Fiji ROI'
        assert region['tool'] == 'polygon'
        assert region['coords'] == [[1.25, 2.5], [8.5, 2.5], [4.0, 9.75]]

    def test_geojson_import_accepts_legacy_flat_statistics(self, manager):
        payload = {
            'type': 'Feature',
            'properties': {
                'name': 'Legacy ROI',
                'tau_median': 2.5,
                'photon_count': 800,
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[
                    [1.0, 1.0], [5.0, 1.0], [3.0, 4.0], [1.0, 1.0],
                ]],
            },
        }

        region_ids = manager.add_geojson(payload)

        region = manager.get_region(region_ids[0])
        assert region is not None
        assert region['statistics'] == {
            'tau_median': 2.5,
            'photon_count': 800,
        }

    def test_geojson_replace_is_transactional(self, manager):
        manager.add_region('Existing', 'rect', [[0, 0], [2, 2]])
        invalid = {
            'type': 'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'properties': {'name': 'Broken'},
                'geometry': {'type': 'Point', 'coordinates': [1, 1]},
            }],
        }

        with pytest.raises(ValueError, match='Unsupported GeoJSON geometry'):
            manager.add_geojson(invalid, mode='replace')

        assert [region['name'] for region in manager.get_all_regions()] == ['Existing']
