import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from mock_data import generate_mock_ptu_tiles, MockPTUFile
from ptu_writer import write_ptu
from flimkit.formats.PTU.reader import PTUFile


class TestDecode:
    """Test suite for decode module."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_ptu(self):
        """Create a mock PTU file."""
        return MockPTUFile(n_y=512, n_x=512, n_bins=256)

    def test_create_time_axis(self):
        """Test time axis creation."""
        from flimkit.formats.PTU.decode import create_time_axis

        n_bins = 256
        tcspc_res = 97e-12

        time_axis = create_time_axis(n_bins, tcspc_res)

        assert len(time_axis) == n_bins
        assert time_axis[0] == 0.0
        assert abs(time_axis[1] - 0.097) < 0.001
        assert abs(time_axis[-1] - 24.7) < 0.1
        assert np.all(np.diff(time_axis) > 0)

    def test_mock_ptu_structure(self, mock_ptu):
        """Test that mock PTU has correct structure."""
        assert hasattr(mock_ptu, 'n_y')
        assert hasattr(mock_ptu, 'n_x')
        assert hasattr(mock_ptu, 'n_bins')
        assert hasattr(mock_ptu, 'tcspc_res')
        assert hasattr(mock_ptu, 'frequency')
        assert hasattr(mock_ptu, 'summed_decay')
        assert hasattr(mock_ptu, 'pixel_stack')
        assert mock_ptu.n_y == 512
        assert mock_ptu.n_x == 512
        assert mock_ptu.n_bins == 256

    def test_summed_decay(self, mock_ptu):
        """Test summed decay extraction."""
        decay = mock_ptu.summed_decay()
        assert decay.shape == (mock_ptu.n_bins,)
        assert np.all(decay >= 0)
        assert np.sum(decay) > 0
        assert np.argmax(decay) > 0
        assert np.argmax(decay) < len(decay) - 1

    def test_pixel_stack(self, mock_ptu):
        """Test pixel stack extraction."""
        stack = mock_ptu.pixel_stack()
        assert stack.shape == (mock_ptu.n_y, mock_ptu.n_x, mock_ptu.n_bins)
        assert np.all(stack >= 0)
        assert np.sum(stack) > 0
        decay_from_stack = stack.sum(axis=(0, 1))
        decay_direct = mock_ptu.summed_decay()
        np.testing.assert_array_almost_equal(decay_from_stack, decay_direct)

    def test_pixel_stack_binning(self, mock_ptu):
        """Test pixel stack with binning."""
        binning = 2
        stack = mock_ptu.pixel_stack(binning=binning)
        expected_y = mock_ptu.n_y // binning
        expected_x = mock_ptu.n_x // binning
        assert stack.shape == (expected_y, expected_x, mock_ptu.n_bins)
        stack_full = mock_ptu.pixel_stack(binning=1)
        assert abs(stack.sum() - stack_full.sum()) < 1

    def test_histogram_properties(self, mock_ptu):
        """Test that histogram has realistic FLIM properties."""
        decay = mock_ptu.summed_decay()
        peak_idx = np.argmax(decay)
        peak_time_ns = peak_idx * mock_ptu.tcspc_res * 1e9
        assert 1.0 < peak_time_ns < 5.0
        tail = decay[peak_idx:]
        decreasing_ratio = np.sum(np.diff(tail) < 0) / len(np.diff(tail))
        assert decreasing_ratio > 0.7


class TestDecodeIntegration:
    """Integration tests with actual decode functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    def test_mock_ptu_save_load(self, temp_dir):
        """Test saving and loading mock PTU files."""
        ptu_files = generate_mock_ptu_tiles(
            temp_dir,
            ptu_basename="R 2",
            n_tiles=1,
            tile_shape=(256, 256),
            n_bins=128
        )

        assert len(ptu_files) == 1
        assert ptu_files[0].exists()

        mock_ptu = PTUFile(str(ptu_files[0]), verbose=False)

        assert mock_ptu.n_y == 256
        assert mock_ptu.n_x == 256
        assert mock_ptu.n_bins == 128

        decay = mock_ptu.summed_decay()
        assert len(decay) == 128
        assert decay.sum() > 0


def test_normalise_flim():
    """Test FLIM normalization function."""
    from flimkit.formats.PTU.decode import normalise_flim

    data_4d = np.random.rand(1, 512, 512, 256)
    result = normalise_flim(data_4d)
    assert result.shape == (512, 512, 256)

    data_3d = np.random.rand(512, 512, 256)
    result = normalise_flim(data_3d)
    assert result.shape == (512, 512, 256)

    data_2d = np.random.rand(512, 512)
    result = normalise_flim(data_2d)
    assert result is None

    result = normalise_flim(None)
    assert result is None


def test_estimate_bg_from_histogram():
    """Test background estimation."""
    from flimkit.FLIM.fit_tools import estimate_bg_from_histogram

    hist = np.random.poisson(5, size=(512, 512, 256))
    hist[:, :, 50:150] += np.random.poisson(100, size=(512, 512, 100))

    bg = estimate_bg_from_histogram(hist, pre_bins=20)

    assert 3 < bg < 7


class TestPTUWriteRead:
    """Test PTU file write and read roundtrip."""

    def test_write_read_roundtrip(self, tmp_path):
        """Write a synthetic histogram and read back, verify it round-trips."""
        ny, nx, nb = 8, 8, 64
        histogram = np.random.poisson(10, size=(ny, nx, nb)).astype(np.uint32)
        tcspc_res = 97e-12
        frequency = 1.0 / ((nb - 0.5) * tcspc_res)

        ptu_path = tmp_path / "test.ptu"
        n = write_ptu(ptu_path, histogram, tcspc_res, frequency, channel=1)
        assert n > 0

        ptu = PTUFile(str(ptu_path), verbose=False)
        stack = ptu.pixel_stack(channel=1, binning=1)

        assert stack.shape == (ny, nx, nb)
        assert stack.sum() == pytest.approx(histogram.sum(), rel=0.01)

        stack_raw = ptu.raw_pixel_stack(channel=1, binning=1)
        assert stack_raw.sum() == pytest.approx(histogram.sum(), rel=0.01)

    def test_write_invalid_shape_raises(self, tmp_path):
        histogram = np.ones((64, 64))
        with pytest.raises(ValueError, match="must be .*Y, X, H"):
            write_ptu(tmp_path / "bad.ptu", histogram, 97e-12, 20e6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
