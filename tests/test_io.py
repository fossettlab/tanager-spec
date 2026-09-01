from __future__ import annotations

import h5py
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tanager_spec import io


def _write_synthetic_sr_h5(
    path, nb=4, ny=5, nx=6, zone=13, fill=-9999.0, projection="HE5_GCTP_UTM"
):
    """Write a minimal HDF-EOS5 SR file mimicking Planet's Tanager layout."""
    rng = np.random.default_rng(0)
    wl = np.linspace(500.0, 2400.0, nb)
    data = rng.random((nb, ny, nx)).astype("float32")
    data[:, 0, 0] = fill  # one nodata pixel across all bands
    ulx, uly, px = 500000.0, 4000000.0, 30.0
    lrx, lry = ulx + nx * px, uly - ny * px
    struct = (
        "GROUP=GridStructure\n\tGROUP=GRID_1\n"
        '\t\tGridName="HYP"\n'
        f"\t\tXDim={nx}\n\t\tYDim={ny}\n"
        f"\t\tUpperLeftPointMtrs=({ulx:.2f},{uly:.2f})\n"
        f"\t\tLowerRightMtrs=({lrx:.2f},{lry:.2f})\n"
        f"\t\tProjection={projection}\n\t\tZoneCode={zone}\n"
        "\tEND_GROUP=GRID_1\nEND_GROUP=GridStructure\n"
    )
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance", data=data)
        ds.attrs["wavelengths"] = wl
        ds.attrs["fwhm"] = np.full(nb, 5.0)
        ds.attrs["_FillValue"] = fill
        f.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.bytes_(struct))
    return wl, (ny, nx)


def test_load_tanager_sr_hdf5(tmp_path):
    h5 = tmp_path / "sr.h5"
    wl, (ny, nx) = _write_synthetic_sr_h5(h5, nb=4, ny=5, nx=6, zone=13)
    cube, returned = io.load_tanager_sr_hdf5(h5)
    assert cube.dims == ("band", "y", "x")
    assert cube.shape == (4, ny, nx)
    assert cube.rio.crs.to_epsg() == 32613  # UTM 13N
    t = cube.rio.transform()
    assert (t.a, t.e, t.c, t.f) == (30.0, -30.0, 500000.0, 4000000.0)
    np.testing.assert_allclose(returned, wl)
    # nodata pixel -> NaN across all bands; others finite
    assert np.all(np.isnan(cube.values[:, 0, 0]))
    assert np.isfinite(cube.values[:, 1, 1]).all()


def test_load_tanager_sr_hdf5_band_subset(tmp_path):
    h5 = tmp_path / "sr.h5"
    wl, _ = _write_synthetic_sr_h5(h5, nb=6)
    cube, returned = io.load_tanager_sr_hdf5(h5, bands=slice(0, 3))
    assert cube.shape[0] == 3
    np.testing.assert_allclose(returned, wl[:3])


def test_load_tanager_sr_hdf5_southern_zone(tmp_path):
    h5 = tmp_path / "sr.h5"
    _write_synthetic_sr_h5(h5, zone=-23)  # GCTP negative zone = southern hemisphere
    cube, _ = io.load_tanager_sr_hdf5(h5)
    assert cube.rio.crs.to_epsg() == 32723  # UTM 23S


def test_load_tanager_sr_hdf5_non_utm_raises(tmp_path):
    h5 = tmp_path / "sr.h5"
    _write_synthetic_sr_h5(h5, projection="HE5_GCTP_GEO")
    with pytest.raises(NotImplementedError, match="UTM"):
        io.load_tanager_sr_hdf5(h5)


def test_load_tanager_sr_hdf5_missing_fillvalue_raises(tmp_path):
    h5 = tmp_path / "sr.h5"
    _write_synthetic_sr_h5(h5)
    with h5py.File(h5, "a") as f:
        del f["HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance"].attrs["_FillValue"]
    with pytest.raises(ValueError, match="_FillValue"):
        io.load_tanager_sr_hdf5(h5)


def test_load_tanager_sr_hdf5_explicit_fill_value(tmp_path):
    h5 = tmp_path / "sr.h5"
    _write_synthetic_sr_h5(h5)
    with h5py.File(h5, "a") as f:
        del f["HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance"].attrs["_FillValue"]
    # explicit fill_value is honored when the attr is absent
    cube, _ = io.load_tanager_sr_hdf5(h5, fill_value=-9999.0)
    assert np.all(np.isnan(cube.values[:, 0, 0]))


def test_parse_hdfeos_picks_matching_grid():
    # Two grids; the requested one (HYP) is second. Must read HYP's zone, not WRONG's.
    struct = (
        'GROUP=GRID_1\n\t\tGridName="WRONG"\n\t\tXDim=10\n\t\tYDim=10\n'
        "\t\tUpperLeftPointMtrs=(0.00,0.00)\n\t\tLowerRightMtrs=(300.00,-300.00)\n"
        "\t\tProjection=HE5_GCTP_UTM\n\t\tZoneCode=1\nEND_GROUP=GRID_1\n"
        'GROUP=GRID_2\n\t\tGridName="HYP"\n\t\tXDim=6\n\t\tYDim=5\n'
        "\t\tUpperLeftPointMtrs=(500000.00,4000000.00)\n"
        "\t\tLowerRightMtrs=(500180.00,3999850.00)\n"
        "\t\tProjection=HE5_GCTP_UTM\n\t\tZoneCode=13\nEND_GROUP=GRID_2\n"
    )
    crs, transform, shape = io._parse_hdfeos_utm_grid(struct, "HYP")
    assert crs == "EPSG:32613"  # HYP's zone, not WRONG's zone 1
    assert shape == (5, 6)
    assert (transform.a, transform.c, transform.f) == (30.0, 500000.0, 4000000.0)


def test_load_tanager_sr_hdf5_missing_wavelengths_raises(tmp_path):
    h5 = tmp_path / "sr.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset(
            "HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance",
            data=np.zeros((3, 4, 4), "float32"),
        )
        f.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.bytes_("Projection=UTM"))
    with pytest.raises(ValueError, match="wavelengths"):
        io.load_tanager_sr_hdf5(h5)


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
        src_path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=n_band,
        dtype="float32",
        crs="EPSG:32613",
        transform=transform,
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
        src_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)
    with pytest.raises(ValueError, match="strictly increasing"):
        io.load_reflectance_cube(src_path, wavelengths=np.array([600.0, 500.0, 700.0]))


def test_load_reflectance_cube_without_wavelengths_returns_none(tmp_path):
    data = np.zeros((3, 4, 4), dtype="float32")
    src_path = tmp_path / "c.tif"
    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)
    cube, wl = io.load_reflectance_cube(src_path)
    # Wavelengths are never inferred from metadata; caller must supply them.
    assert wl is None
    assert cube.sizes["band"] == 3
