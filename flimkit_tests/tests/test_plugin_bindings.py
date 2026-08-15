import numpy as np
import pytest

from flimkit.plugins import get_current_images


class _FovPreview:
    def __init__(self, intensity=None, lifetime=None):
        self._intensity_map = intensity
        self._lifetime_map = lifetime


class _App:
    def __init__(self, fov_preview=None):
        self._fov_preview = fov_preview


def test_get_current_images_returns_named_copies():
    intensity = np.arange(12, dtype=np.float32).reshape(3, 4)
    lifetime = intensity / 10.0
    app = _App(_FovPreview(intensity, lifetime))

    images = get_current_images(app)

    assert set(images) == {'intensity', 'lifetime'}
    np.testing.assert_array_equal(images['intensity'], intensity)
    np.testing.assert_array_equal(images['lifetime'], lifetime)
    assert images['intensity'] is not intensity
    assert images['lifetime'] is not lifetime


def test_get_current_images_omits_unavailable_images():
    intensity = np.ones((2, 3), dtype=np.float32)
    app = _App(_FovPreview(intensity=intensity))

    images = get_current_images(app)

    assert set(images) == {'intensity'}


def test_get_current_images_requires_fov_preview():
    with pytest.raises(RuntimeError, match='FOV preview is not available'):
        get_current_images(_App())
