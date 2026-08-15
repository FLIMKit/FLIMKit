import threading
from typing import Any, Callable, Dict, List, TypeVar, cast


_T = TypeVar('_T')


def _fov_preview(app):
    preview = getattr(app, '_fov_preview', None)
    if preview is None:
        raise RuntimeError('FOV preview is not available')
    return preview


def _roi_manager(preview):
    manager = getattr(preview, '_roi_manager', None)
    if manager is None:
        raise RuntimeError('ROI manager is not available')
    return manager


def _run_on_ui_thread(app, callback: Callable[[], _T]) -> _T:
    if threading.current_thread() is threading.main_thread():
        return callback()

    root = getattr(app, 'root', None)
    if root is None or not callable(getattr(root, 'after', None)):
        raise RuntimeError('FLIMKit UI is not available')

    done = threading.Event()
    outcome = {}

    def run():
        try:
            outcome['value'] = callback()
        except BaseException as error:
            outcome['error'] = error
        finally:
            done.set()

    root.after(0, run)
    done.wait()
    if 'error' in outcome:
        raise outcome['error']
    return cast(_T, outcome.get('value'))


def get_current_images(app) -> Dict[str, Any]:
    """Return copies of the intensity and lifetime images currently shown."""
    import numpy as np

    def snapshot():
        preview = _fov_preview(app)
        images = {}
        for name, attribute in (
            ('intensity', '_intensity_map'),
            ('lifetime', '_lifetime_map'),
        ):
            image = getattr(preview, attribute, None)
            if image is not None:
                images[name] = np.array(image, copy=True)
        return images

    return _run_on_ui_thread(app, snapshot)


def export_rois_geojson(app) -> Dict:
    """Return the current FLIMKit ROIs as a GeoJSON FeatureCollection."""
    def export():
        manager = _roi_manager(_fov_preview(app))
        return manager.to_geojson()

    return _run_on_ui_thread(app, export)


def import_rois_geojson(app, payload: Dict, mode: str = 'append') -> List[int]:
    """Import GeoJSON ROIs and refresh the FLIMKit ROI display."""
    def import_rois():
        preview = _fov_preview(app)
        manager = _roi_manager(preview)
        region_ids = manager.add_geojson(payload, mode=mode)
        if region_ids or mode == 'replace':
            redraw = getattr(preview, '_redraw_region_overlays', None)
            if callable(redraw):
                redraw()
            save = getattr(preview, '_save_regions_update', None)
            if callable(save):
                save()
            panel = getattr(preview, '_roi_analysis_panel', None)
            if panel is None:
                panel = getattr(app, '_roi_analysis_panel', None)
            refresh = getattr(panel, '_refresh_region_list', None)
            if callable(refresh):
                refresh()
        return region_ids

    return _run_on_ui_thread(app, import_rois)
