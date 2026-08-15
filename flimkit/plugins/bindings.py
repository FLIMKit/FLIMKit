from typing import Dict

import numpy as np


def _fov_preview(app):
    preview = getattr(app, '_fov_preview', None)
    if preview is None:
        raise RuntimeError('FOV preview is not available')
    return preview


def get_current_images(app) -> Dict[str, np.ndarray]:
    """Return copies of the intensity and lifetime images currently shown."""
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
