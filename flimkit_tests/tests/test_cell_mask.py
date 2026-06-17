import pytest
import numpy as np
from unittest.mock import patch, MagicMock

cellpose = pytest.importorskip('cellpose')

from flimkit.image.tools import make_cell_mask


def _blank_intensity(h=64, w=64):
    # two bright square blobs on a black background
    img = np.zeros((h, w), dtype=np.float32)
    img[10:24, 10:24] = 120.0
    img[40:54, 40:54] = 120.0
    return img


def _mock_model(label_map):
    m = MagicMock()
    m.eval.return_value = (label_map, None, None)
    return m


class TestMakeCellMask:
    @patch('cellpose.models.CellposeModel')
    def test_returns_bool_mask(self, mock_cls):
        img = _blank_intensity()
        labels = np.zeros(img.shape, dtype=np.int32)
        labels[10:24, 10:24] = 1
        labels[40:54, 40:54] = 2
        mock_cls.return_value = _mock_model(labels)

        mask = make_cell_mask(img, gpu=False)

        assert mask.dtype == bool
        assert mask.shape == img.shape

    @patch('cellpose.models.CellposeModel')
    def test_cell_pixels_true_background_false(self, mock_cls):
        img = _blank_intensity()
        labels = np.zeros(img.shape, dtype=np.int32)
        labels[10:24, 10:24] = 1
        labels[40:54, 40:54] = 2
        mock_cls.return_value = _mock_model(labels)

        mask = make_cell_mask(img, gpu=False)

        assert mask[16, 16]
        assert mask[47, 47]
        assert not mask[0, 0]

    @patch('cellpose.models.CellposeModel')
    def test_no_cells_returns_all_false(self, mock_cls):
        img = _blank_intensity()
        labels = np.zeros(img.shape, dtype=np.int32)
        mock_cls.return_value = _mock_model(labels)

        mask = make_cell_mask(img, gpu=False)

        assert not mask.any()

    @patch('cellpose.models.CellposeModel')
    def test_resize_scales_mask_back(self, mock_cls):
        # input 128x128, model runs at 224x224, output must be 128x128
        img = np.ones((128, 128), dtype=np.float32) * 50.0
        labels_224 = np.ones((224, 224), dtype=np.int32)
        mock_cls.return_value = _mock_model(labels_224)

        mask = make_cell_mask(img, resize_to=224, gpu=False)

        assert mask.shape == (128, 128)

    @patch('cellpose.models.CellposeModel')
    def test_accepts_uint8(self, mock_cls):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[20:44, 20:44] = 200
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[20:44, 20:44] = 1
        mock_cls.return_value = _mock_model(labels)

        mask = make_cell_mask(img, resize_to=None, gpu=False)

        assert mask.dtype == bool
        assert mask[30, 30]

    @patch('cellpose.models.CellposeModel')
    def test_save_mask_writes_png(self, mock_cls, tmp_path):
        img = _blank_intensity()
        labels = np.ones(img.shape, dtype=np.int32)
        mock_cls.return_value = _mock_model(labels)

        out = tmp_path / "sample.tif"
        make_cell_mask(img, save_mask=True, path=str(out), gpu=False)

        assert (tmp_path / "sample_cell_mask.png").exists()
