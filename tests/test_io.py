from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tanagerspec import io


def test_grid_from_positions_scatters_values():
    grid = io.grid_from_positions(
        values=np.array([1.0, 2.0, 3.0]),
        rows=np.array([0, 1, 2]),
        cols=np.array([0, 1, 2]),
        spatial_shape=(3, 3),
        fill=-1.0,
    )
    assert grid[0, 0] == 1.0
    assert grid[1, 1] == 2.0
    assert grid[2, 2] == 3.0
    assert grid[0, 1] == -1.0


def test_write_geotiff_roundtrip(tmp_path):
    data = np.arange(12, dtype="float32").reshape(3, 4)
    transform = from_origin(500000, 4000000, 30, 30)
    crs = "EPSG:32613"
    out = tmp_path / "scores.tif"
    io.write_geotiff(data, transform, crs, out, nodata=-1.0, tags={"detector": "test"})
    assert out.exists()
    with rasterio.open(out) as src:
        assert src.count == 1
        assert src.crs.to_epsg() == 32613
        assert src.transform == transform
        np.testing.assert_array_equal(src.read(1), data)
        assert src.tags()["detector"] == "test"


def test_positions_to_geotiff(tmp_path):
    out = tmp_path / "mask.tif"
    io.positions_to_geotiff(
        values=np.array([7.0, 8.0]),
        rows=np.array([0, 2]),
        cols=np.array([1, 3]),
        spatial_shape=(3, 4),
        transform=from_origin(0, 0, 1, 1),
        crs="EPSG:4326",
        path=out,
        nodata=np.nan,
    )
    with rasterio.open(out) as src:
        arr = src.read(1)
    assert arr[0, 1] == 7.0
    assert arr[2, 3] == 8.0
    assert np.isnan(arr[0, 0])


def test_load_reflectance_cube_roundtrip(tmp_path):
    n_band, ny, nx = 5, 6, 7
    data = np.random.default_rng(0).random((n_band, ny, nx)).astype("float32")
    transform = from_origin(500000, 4000000, 30, 30)
    src_path = tmp_path / "cube.tif"
    with rasterio.open(
        src_path, "w", driver="GTiff", height=ny, width=nx, count=n_band,
        dtype="float32", crs="EPSG:32613", transform=transform,
    ) as dst:
        dst.write(data)

    wl = np.linspace(400.0, 2400.0, n_band)
    cube, returned_wl = io.load_reflectance_cube(src_path, wavelengths=wl)
    assert cube.sizes["band"] == n_band
    assert cube.rio.crs.to_epsg() == 32613
    np.testing.assert_array_equal(returned_wl, wl)
    np.testing.assert_allclose(cube.values, data)


def test_load_reflectance_cube_rejects_non_increasing_wavelengths(tmp_path):
    data = np.zeros((3, 4, 4), dtype="float32")
    src_path = tmp_path / "c.tif"
    with rasterio.open(
        src_path, "w", driver="GTiff", height=4, width=4, count=3,
        dtype="float32", crs="EPSG:4326", transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)
    with pytest.raises(ValueError, match="strictly increasing"):
        io.load_reflectance_cube(src_path, wavelengths=np.array([600.0, 500.0, 700.0]))


def test_load_reflectance_cube_without_wavelengths_returns_none(tmp_path):
    data = np.zeros((3, 4, 4), dtype="float32")
    src_path = tmp_path / "c.tif"
    with rasterio.open(
        src_path, "w", driver="GTiff", height=4, width=4, count=3,
        dtype="float32", crs="EPSG:4326", transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)
    cube, wl = io.load_reflectance_cube(src_path)
    # Wavelengths are never inferred from metadata; caller must supply them.
    assert wl is None
    assert cube.sizes["band"] == 3
