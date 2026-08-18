import copy
import threading
from queue import Queue

import numpy as np
import pytest

import flimkit.plugins.bindings as plugin_bindings

from flimkit.plugins import (
    export_rois_geojson,
    get_current_images,
    import_rois_geojson,
)
from flimkit.utils.roi import RoiManager


class _FovPreview:
    def __init__(self, intensity=None, lifetime=None, roi_manager=None, roi_panel=None):
        self._intensity_map = intensity
        self._lifetime_map = lifetime
        self._roi_manager = roi_manager
        self._roi_analysis_panel = roi_panel
        self.redraw_count = 0
        self.save_count = 0

    def _redraw_region_overlays(self):
        self.redraw_count += 1

    def _save_regions_update(self):
        self.save_count += 1


class _Panel:
    def __init__(self):
        self.refresh_count = 0

    def _refresh_region_list(self):
        self.refresh_count += 1


class _QueuedRoot:
    def __init__(self):
        self.callbacks = Queue()

    def after(self, _delay, callback):
        self.callbacks.put(callback)


class _App:
    def __init__(self, fov_preview=None, root=None, roi_panel=None):
        self._fov_preview = fov_preview
        self.root = root
        self._roi_analysis_panel = roi_panel


def _polygon_payload():
    return {
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


def test_get_current_images_returns_named_copies_and_units():
    intensity = np.arange(12, dtype=np.float32).reshape(3, 4)
    lifetime = intensity / 10.0
    app = _App(_FovPreview(intensity, lifetime))

    result = get_current_images(app)
    images = result['images']

    assert set(result) == {'images', 'units'}
    assert set(images) == {'intensity', 'lifetime'}
    assert result['units'] == {
        'intensity': 'photons',
        'lifetime': 'ns',
    }
    np.testing.assert_array_equal(images['intensity'], intensity)
    np.testing.assert_array_equal(images['lifetime'], lifetime)
    assert images['intensity'] is not intensity
    assert images['lifetime'] is not lifetime


def test_get_current_images_omits_unavailable_images_and_units():
    intensity = np.ones((2, 3), dtype=np.float32)
    app = _App(_FovPreview(intensity=intensity))

    result = get_current_images(app)

    assert set(result['images']) == {'intensity'}
    assert result['units'] == {'intensity': 'photons'}


def test_get_current_images_sums_trailing_intensity_axes():
    intensity = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    lifetime = np.ones((2, 3), dtype=np.float32)
    app = _App(_FovPreview(intensity, lifetime))

    result = get_current_images(app)
    images = result['images']

    assert images['intensity'].shape == (2, 3)
    np.testing.assert_array_equal(images['intensity'], intensity.sum(axis=2))
    assert images['lifetime'].shape == (2, 3)


def test_get_current_images_rejects_non_2d_lifetime():
    intensity = np.ones((2, 3), dtype=np.float32)
    lifetime = np.ones((2, 3, 2), dtype=np.float32)
    app = _App(_FovPreview(intensity, lifetime))

    with pytest.raises(RuntimeError, match='lifetime image must be 2D'):
        get_current_images(app)


def test_get_current_images_requires_fov_preview():
    with pytest.raises(RuntimeError, match='FOV preview is not available'):
        get_current_images(_App())


def test_get_current_images_marshals_background_calls_to_ui_thread():
    root = _QueuedRoot()
    intensity = np.arange(6, dtype=np.float32).reshape(2, 3)
    app = _App(_FovPreview(intensity=intensity), root=root)
    result = {}

    worker = threading.Thread(
        target=lambda: result.update(get_current_images(app)),
    )
    worker.start()
    callback = root.callbacks.get(timeout=2)
    assert worker.is_alive()

    callback()
    worker.join(timeout=2)

    assert not worker.is_alive()
    np.testing.assert_array_equal(result['images']['intensity'], intensity)
    assert result['units'] == {'intensity': 'photons'}


def test_export_rois_geojson_uses_current_roi_manager():
    manager = RoiManager()
    manager.add_region('Cell 1', 'rect', [[2, 3], [8, 9]])
    app = _App(_FovPreview(roi_manager=manager))

    payload = export_rois_geojson(app)

    assert payload['type'] == 'FeatureCollection'
    assert payload['features'][0]['properties']['name'] == 'Cell 1'
    assert payload['features'][0]['properties']['tool_type'] == 'rect'


def test_import_rois_geojson_updates_manager_and_ui():
    manager = RoiManager()
    preview = _FovPreview(roi_manager=manager)
    panel = _Panel()
    app = _App(preview, roi_panel=panel)

    region_ids = import_rois_geojson(app, _polygon_payload())

    assert len(region_ids) == 1
    region = manager.get_region(region_ids[0])
    assert region is not None
    assert region['name'] == 'Fiji ROI'
    assert region['tool'] == 'polygon'
    assert preview.redraw_count == 1
    assert preview.save_count == 1
    assert panel.refresh_count == 1


def test_roi_bindings_require_roi_manager():
    app = _App(_FovPreview())

    with pytest.raises(RuntimeError, match='ROI manager is not available'):
        export_rois_geojson(app)
    with pytest.raises(RuntimeError, match='ROI manager is not available'):
        import_rois_geojson(app, _polygon_payload())


def test_background_import_waits_for_ui_callback(monkeypatch):
    manager = RoiManager()
    preview = _FovPreview(roi_manager=manager)
    root = _QueuedRoot()
    app = _App(preview, root=root)
    errors = []
    region_ids = []
    monkeypatch.setattr(
        plugin_bindings,
        '_UI_TIMEOUT_SECONDS',
        0.01,
        raising=False,
    )

    def import_in_background():
        try:
            region_ids.extend(import_rois_geojson(app, _polygon_payload()))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=import_in_background)
    worker.start()
    callback = root.callbacks.get(timeout=2)
    worker.join(timeout=0.05)

    assert worker.is_alive()
    assert errors == []
    assert manager.get_all_regions() == []

    callback()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == []
    assert len(region_ids) == 1
    assert manager.get_region(region_ids[0]) is not None
    assert preview.redraw_count == 1
    assert preview.save_count == 1


def test_import_refreshes_panel_attached_to_active_preview():
    manager = RoiManager()
    active_panel = _Panel()
    preview = _FovPreview(roi_manager=manager, roi_panel=active_panel)
    hidden_panel = _Panel()
    app = _App(preview, roi_panel=hidden_panel)

    import_rois_geojson(app, _polygon_payload())

    assert active_panel.refresh_count == 1
    assert hidden_panel.refresh_count == 0


def test_import_copies_statistics_without_mutating_payload():
    manager = RoiManager()
    app = _App(_FovPreview(roi_manager=manager))
    payload = _polygon_payload()
    properties = payload['features'][0]['properties']
    properties['statistics'] = {'nested': {'value': 1}}
    properties['tau_median'] = 2.5
    original = copy.deepcopy(payload)

    region_ids = import_rois_geojson(app, payload)

    assert payload == original
    region = manager.get_region(region_ids[0])
    assert region is not None
    assert region['statistics'] == {
        'nested': {'value': 1},
        'tau_median': 2.5,
    }

    properties['statistics']['nested']['value'] = 99
    properties['statistics']['new'] = 'changed later'

    assert region['statistics'] == {
        'nested': {'value': 1},
        'tau_median': 2.5,
    }
