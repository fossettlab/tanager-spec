from __future__ import annotations

import math

import pytest

from tanagerspec import config, stac


def test_query_tanager_raises_when_unconfigured():
    # Config ships with the endpoint/collection unset (TODO). The helper must
    # refuse rather than contact an invented URL.
    assert config.TANAGER_STAC_URL is None
    with pytest.raises(ValueError, match="not configured"):
        stac.query_tanager_scenes(bbox=[-1, -1, 1, 1])


def test_build_scene_inventory_from_items():
    items = [
        {
            "id": "scene_a",
            "bbox": [-90.5, 38.5, -90.0, 38.8],
            "geometry": None,
            "properties": {"datetime": "2025-01-01T00:00:00Z", "eo:cloud_cover": 12.0},
            "assets": {"reflectance": {"href": "s3://bucket/scene_a.tif"}},
        },
        {
            "id": "scene_b",
            "bbox": [10.0, 20.0, 11.0, 21.0],
            "geometry": None,
            "properties": {"datetime": "2025-02-01T00:00:00Z"},  # no cloud cover
            "assets": {},
        },
    ]
    gdf = stac.build_scene_inventory(items)
    assert list(gdf["scene_id"]) == ["scene_a", "scene_b"]
    assert gdf.crs.to_epsg() == 4326
    assert gdf.loc[0, "cloud_cover"] == 12.0
    # Absent cloud cover is NaN, never imputed.
    assert math.isnan(gdf.loc[1, "cloud_cover"])
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
