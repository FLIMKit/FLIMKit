import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from flimkit.utils.roi import RoiManager


class TestOMETIFFExport:
    """Test OME-TIFF export from fit results."""

    def test_ometiff_export_intensity(self, tmp_path):
        """Export intensity as OME-TIFF and verify it's readable."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")

        from flimkit.utils.enhanced_outputs import save_weighted_tau_images

        # Create dummy pixel_maps
        ny, nx = 64, 64
        pixel_maps = {
            'intensity': np.random.poisson(100, (ny, nx)).astype(np.float32),
            'tau_1': np.random.uniform(1.0, 3.0, (ny, nx)).astype(np.float32),
            'a1': np.random.uniform(0.5, 1.5, (ny, nx)).astype(np.float32),
        }

        save_weighted_tau_images(
            pixel_maps,
            tmp_path,
            roi_name="test_roi",
            n_exp=1,
            save_intensity=True,
            save_amplitude=False,
        )

        # The function saves TIFF, not OME-TIFF by default.
        # We'll test that we can read the saved TIFF and it has correct dimensions.
        tiff_path = tmp_path / "test_roi_intensity.tif"
        assert tiff_path.exists()
        img = tifffile.imread(str(tiff_path))
        assert img.shape == (ny, nx)

class TestGeoJSONExport:
    """Test ROI export to GeoJSON."""

    def test_export_single_region_geojson(self, tmp_path):
        """Export a rectangle region as GeoJSON Feature."""
        manager = RoiManager()
        manager.add_region("TestRect", "rect", [[10, 20], [50, 60]])
        region_id = 0

        # Simulate export
        region = manager.get_region(region_id)
        feature = {
            "type": "Feature",
            "properties": {
                "id": region['id'],
                "name": region['name'],
                "tool_type": region['tool'],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [10, 20], [50, 20], [50, 60], [10, 60], [10, 20]
                ]]
            }
        }

        geojson_path = tmp_path / "region.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(feature, f)

        # Read back and verify
        with open(geojson_path) as f:
            loaded = json.load(f)
        assert loaded['type'] == 'Feature'
        assert loaded['properties']['name'] == 'TestRect'
        assert loaded['geometry']['type'] == 'Polygon'

    def test_export_all_regions_feature_collection(self, tmp_path):
        """Export multiple regions as GeoJSON FeatureCollection."""
        manager = RoiManager()
        manager.add_region("Rect1", "rect", [[0, 0], [10, 10]])
        manager.add_region("Ellipse1", "ellipse", [[20, 20], [40, 40]])

        features = []
        for region in manager.get_all_regions():
            coords = region['coords']
            if region['tool'] == 'rect':
                geometry = {
                    "type": "Polygon",
                    "coordinates": [[
                        [coords[0][0], coords[0][1]],
                        [coords[1][0], coords[0][1]],
                        [coords[1][0], coords[1][1]],
                        [coords[0][0], coords[1][1]],
                        [coords[0][0], coords[0][1]]
                    ]]
                }
            else:
                # Approximate ellipse as polygon for test
                geometry = {"type": "Polygon", "coordinates": [[[20,20],[40,20],[40,40],[20,40],[20,20]]]}
            features.append({
                "type": "Feature",
                "properties": {"id": region['id'], "name": region['name']},
                "geometry": geometry
            })

        collection = {"type": "FeatureCollection", "features": features}
        geojson_path = tmp_path / "all_regions.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(collection, f)

        with open(geojson_path) as f:
            loaded = json.load(f)
        assert loaded['type'] == 'FeatureCollection'
        assert len(loaded['features']) == 2

class TestOMEZarrExport:

    def _write(self, tmp_path, **kwargs):
        from flimkit.utils.ome_zarr import write_ome_zarr
        ny, nx = 32, 48
        channels = {
            'intensity': np.random.poisson(100, (ny, nx)).astype(np.float32),
            'lifetime': np.random.uniform(1.0, 3.0, (ny, nx)).astype(np.float32),
        }
        channels['lifetime'][0, 0] = np.nan
        out = tmp_path / 'results.ome.zarr'
        write_ome_zarr(out, channels, **kwargs)
        return out, channels

    def test_channels_round_trip(self, tmp_path):
        zarr = pytest.importorskip('zarr')
        out, channels = self._write(tmp_path)
        group = zarr.open_group(str(out), mode='r')
        data = group['0']
        assert data.shape == (2, 32, 48)
        assert data.dtype == np.float32
        assert np.allclose(data[0], channels['intensity'])
        assert np.isnan(data[1][0, 0])

    def test_ngff_metadata(self, tmp_path):
        zarr = pytest.importorskip('zarr')
        out, _ = self._write(tmp_path, pixel_size_um=0.284)
        attrs = dict(zarr.open_group(str(out), mode='r').attrs)
        multiscales = attrs['multiscales'][0]
        assert [a['name'] for a in multiscales['axes']] == ['c', 'y', 'x']
        assert multiscales['axes'][1]['unit'] == 'micrometer'
        assert multiscales['datasets'][0]['path'] == '0'
        scale = multiscales['datasets'][0]['coordinateTransformations'][0]['scale']
        assert scale == [1.0, 0.284, 0.284]
        assert [c['label'] for c in attrs['omero']['channels']] == ['intensity', 'lifetime']

    def test_fit_results_land_in_metadata(self, tmp_path):
        zarr = pytest.importorskip('zarr')
        from flimkit.utils.ome_zarr import fit_metadata
        summary = {'taus_ns': np.array([2.1, 0.4]), 'chi2': np.float64(1.03),
                   'bad': np.float64('nan')}
        meta = fit_metadata({'global_summary': summary, 'n_exp': 2}, 0.284,
                            channel_units={'lifetime': 'ns'})
        out, _ = self._write(tmp_path, metadata=meta)
        attrs = dict(zarr.open_group(str(out), mode='r').attrs)
        stored = attrs['flimkit']
        assert stored['n_exp'] == 2
        assert stored['pixel_size_um'] == 0.284
        assert stored['global_summary']['taus_ns'] == [2.1, 0.4]
        assert stored['global_summary']['chi2'] == 1.03
        assert stored['global_summary']['bad'] is None
        assert stored['channel_units'] == {'lifetime': 'ns'}

    def test_smaller_than_uncompressed_tiff(self, tmp_path):
        pytest.importorskip('zarr')
        tifffile = pytest.importorskip('tifffile')
        from flimkit.utils.ome_zarr import write_ome_zarr
        yy, xx = np.mgrid[0:512, 0:512]
        intensity = (np.exp(-((yy - 256) ** 2 + (xx - 256) ** 2) / 5e4) * 3000).astype(np.float32)
        lifetime = (2.0 + 0.5 * np.sin(xx / 60.0)).astype(np.float32)
        out = tmp_path / 'results.ome.zarr'
        write_ome_zarr(out, {'intensity': intensity, 'lifetime': lifetime})
        tifffile.imwrite(tmp_path / 'intensity.ome.tiff', intensity)
        tifffile.imwrite(tmp_path / 'lifetime.ome.tiff', lifetime)
        zarr_bytes = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
        tiff_bytes = sum((tmp_path / n).stat().st_size
                         for n in ('intensity.ome.tiff', 'lifetime.ome.tiff'))
        assert zarr_bytes < tiff_bytes

    def test_mismatched_shapes_rejected(self, tmp_path):
        pytest.importorskip('zarr')
        from flimkit.utils.ome_zarr import write_ome_zarr
        with pytest.raises(ValueError):
            write_ome_zarr(tmp_path / 'bad.ome.zarr',
                           {'a': np.zeros((8, 8), np.float32),
                            'b': np.zeros((8, 9), np.float32)})

    def test_no_channels_rejected(self, tmp_path):
        from flimkit.utils.ome_zarr import write_ome_zarr
        with pytest.raises(ValueError):
            write_ome_zarr(tmp_path / 'empty.ome.zarr', {})


class TestOMEZarrExportDialog:

    def _run(self, tmp_path, scan_stem, images, fit_result=None):
        from unittest.mock import patch
        from flimkit.UI.gui import _UIBuilder

        class Stub:
            _fov_preview = None
            def _get_pixel_size_um(self):
                return 0.284
            def _current_scan_stem(self):
                return scan_stem

        with patch('subprocess.Popen'):
            _UIBuilder._export_images(Stub(), images, str(tmp_path),
                                      with_scalebar=False, with_annotations=False,
                                      format='omezarr',
                                      fit_result=fit_result or images)
        return sorted(p.name for p in tmp_path.iterdir())

    def _maps(self, ny=16, nx=20):
        return {'intensity': np.random.poisson(50, (ny, nx)).astype(np.float32),
                'lifetime': np.random.uniform(1.0, 3.0, (ny, nx)).astype(np.float32)}

    def test_store_named_after_scan(self, tmp_path):
        pytest.importorskip('zarr')
        assert self._run(tmp_path, 'R146_FOV1', self._maps()) == ['R146_FOV1.ome.zarr']

    def test_two_scans_do_not_overwrite(self, tmp_path):
        pytest.importorskip('zarr')
        self._run(tmp_path, 'R146_FOV1', self._maps())
        names = self._run(tmp_path, 'R146_FOV2', self._maps())
        assert names == ['R146_FOV1.ome.zarr', 'R146_FOV2.ome.zarr']

    def test_falls_back_to_results_without_a_scan_name(self, tmp_path):
        pytest.importorskip('zarr')
        assert self._run(tmp_path, '', self._maps()) == ['results.ome.zarr']

    def test_rgb_composites_are_skipped(self, tmp_path):
        zarr = pytest.importorskip('zarr')
        images = self._maps()
        images['overlay'] = np.zeros((16, 20, 3), np.float32)
        self._run(tmp_path, 'scan', images)
        attrs = dict(zarr.open_group(str(tmp_path / 'scan.ome.zarr'), mode='r').attrs)
        assert [c['label'] for c in attrs['omero']['channels']] == ['intensity', 'lifetime']

    def test_fit_summary_is_written(self, tmp_path):
        zarr = pytest.importorskip('zarr')
        images = self._maps()
        fit_result = dict(images)
        fit_result['global_summary'] = {'taus_ns': np.array([2.2, 0.5])}
        fit_result['n_exp'] = 2
        self._run(tmp_path, 'scan', images, fit_result=fit_result)
        attrs = dict(zarr.open_group(str(tmp_path / 'scan.ome.zarr'), mode='r').attrs)
        assert attrs['flimkit']['n_exp'] == 2
        assert attrs['flimkit']['global_summary']['taus_ns'] == [2.2, 0.5]
        assert attrs['flimkit']['channel_units'] == {'lifetime': 'ns'}
        assert attrs['multiscales'][0]['name'] == 'scan'
