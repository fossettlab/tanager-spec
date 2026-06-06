from __future__ import annotations

import math

import pandas as pd

from tanager_spec import config, stac


def test_tanager_catalog_url_configured():
    # The Tanager open data is a verified static catalog; the URL is set.
    assert config.TANAGER_STAC_URL.endswith("tanager-core-imagery/catalog.json")
    assert config.TANAGER_SR_ASSET == "ortho_sr_hdf5"


def test_bbox_overlaps():
    assert stac._bbox_overlaps([0, 0, 2, 2], [1, 1, 3, 3]) is True
    assert stac._bbox_overlaps([0, 0, 1, 1], [2, 2, 3, 3]) is False
    assert stac._bbox_overlaps([0, 0, 2, 2], [0.5, 0.5, 1.5, 1.5]) is True  # contained


def test_passes_filters_bbox():
    f = stac._passes_filters
    kw = dict(start=None, end=None, max_cloud_cover=None)
    # overlapping bbox passes; non-overlapping and missing-bbox under a filter fail
    assert f([0, 0, 2, 2], None, None, bbox=[1, 1, 3, 3], **kw)
    assert not f([10, 10, 11, 11], None, None, bbox=[0, 0, 1, 1], **kw)
    assert not f(None, None, None, bbox=[0, 0, 1, 1], **kw)


def test_passes_filters_datetime():
    f = stac._passes_filters
    assert f(None, "2025-06-01T00:00:00Z", None, bbox=None, start="2025-01-01", end="2025-12-31",
             max_cloud_cover=None)
    assert not f(None, "2024-01-01T00:00:00Z", None, bbox=None, start="2025-01-01", end=None,
                 max_cloud_cover=None)
    # missing datetime under a datetime filter fails
    assert not f(None, None, None, bbox=None, start="2025-01-01", end=None, max_cloud_cover=None)


def test_passes_filters_cloud():
    f = stac._passes_filters
    assert f(None, None, 5.0, bbox=None, start=None, end=None, max_cloud_cover=10.0)
    assert not f(None, None, 50.0, bbox=None, start=None, end=None, max_cloud_cover=10.0)
    # missing cloud under a cloud filter fails
    assert not f(None, None, None, bbox=None, start=None, end=None, max_cloud_cover=10.0)


def test_build_scene_inventory_from_items():
    items = [
        {
            "id": "scene_a",
            "bbox": [-90.5, 38.5, -90.0, 38.8],
            "geometry": None,
            "properties": {
                "datetime": "2025-01-01T00:00:00Z",
                "cloud_percent": 12.0,
                "category": "agriculture",
            },
            "assets": {"ortho_sr_hdf5": {"href": "https://x/scene_a.h5"}},
        },
        {
            "id": "scene_b",
            "bbox": [10.0, 20.0, 11.0, 21.0],
            "geometry": None,
            "properties": {"datetime": "2025-02-01T00:00:00Z"},  # no cloud, no category
            "assets": {},
        },
    ]
    gdf = stac.build_scene_inventory(items)
    assert list(gdf["scene_id"]) == ["scene_a", "scene_b"]
    assert gdf.crs.to_epsg() == 4326
    assert gdf.loc[0, "cloud_percent"] == 12.0
    assert gdf.loc[0, "category"] == "agriculture"
    # Absent cloud cover and category are missing (NaN), never imputed.
    assert math.isnan(gdf.loc[1, "cloud_percent"])
    assert pd.isna(gdf.loc[1, "category"])
    assert gdf.geometry.iloc[0] is not None  # bbox fallback geometry


def test_save_inventory_writes_csv(tmp_path):
    items = [
        {
            "id": "scene_a",
            "bbox": [-1.0, -1.0, 1.0, 1.0],
            "geometry": None,
            "properties": {"datetime": "2025-01-01T00:00:00Z", "eo:cloud_cover": 5.0},
            "assets": {"reflectance": {"href": "s3://b/a.tif"}},
        },
    ]
    gdf = stac.build_scene_inventory(items)
    out = tmp_path / "inv.csv"
    stac.save_inventory(gdf, out)
    assert out.exists()
    assert "scene_a" in out.read_text()
