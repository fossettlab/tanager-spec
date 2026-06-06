"""STAC queries for Tanager and EMIT scenes.

The Planet Tanager open data is a **static** STAC catalog (no ``/search``
endpoint): it is traversed by walking child collections, which are the scene
categories. EMIT is a real STAC **API** (LP DAAC), queried with pystac-client.
Asset URLs are always resolved through the STAC item's ``assets`` dict, never
hand-assembled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, shape

from . import config

logger = logging.getLogger(__name__)


def _bbox_overlaps(a: list[float], b: list[float]) -> bool:
    """Return True if two ``[xmin, ymin, xmax, ymax]`` boxes overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _passes_filters(
    item_bbox: list[float] | None,
    item_dt: str | None,
    item_cloud: float | None,
    *,
    bbox: list[float] | None,
    start: str | None,
    end: str | None,
    max_cloud_cover: float | None,
) -> bool:
    """Return whether an item satisfies the active filters.

    When a filter is active and the item lacks the field it acts on, the item
    fails (an unfilterable item cannot be shown to satisfy the filter).
    """
    if bbox is not None and (item_bbox is None or not _bbox_overlaps(bbox, item_bbox)):
        return False
    if (start is not None or end is not None):
        if item_dt is None:
            return False
        if start is not None and item_dt < start:
            return False
        if end is not None and item_dt > end:
            return False
    if max_cloud_cover is not None and (item_cloud is None or item_cloud > max_cloud_cover):
        return False
    return True


def query_stac_items(
    catalog_url: str,
    collection: str,
    bbox: list[float] | None = None,
    datetime: str | None = None,
    query: dict | None = None,
    max_items: int | None = None,
) -> list[dict]:
    """Run a STAC item search and return items as dicts.

    Parameters
    ----------
    catalog_url : str
        STAC API root URL.
    collection : str
        Collection id to search.
    bbox : list of float, optional
        ``[xmin, ymin, xmax, ymax]`` in WGS84.
    datetime : str, optional
        RFC 3339 datetime or interval (e.g. ``"2024-01-01/2026-06-01"``).
    query : dict, optional
        Additional property filters passed to the STAC search ``query`` field.
    max_items : int, optional
        Cap on returned items.

    Returns
    -------
    list of dict
        STAC item dicts (``id``, ``bbox``, ``geometry``, ``properties``,
        ``assets``).
    """
    import pystac_client

    client = pystac_client.Client.open(catalog_url)
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime,
        query=query,
        max_items=max_items,
    )
    items = [item.to_dict() for item in search.items()]
    logger.info("STAC search on %s/%s returned %d items", catalog_url, collection, len(items))
    return items


def query_tanager_scenes(
    bbox: list[float] | None = None,
    datetime: str | None = None,
    categories: list[str] | None = None,
    max_cloud_cover: float | None = None,
    max_items: int | None = None,
    catalog_url: str | None = None,
) -> list[dict]:
    """Traverse the Planet Tanager static STAC catalog and return scene items.

    The catalog is static (no search API), so this walks the child collections
    (scene categories) and their items, filtering in Python. Each returned item
    dict has its parent collection id injected at ``properties['category']``.

    When a filter is active, items missing the field it acts on are excluded
    (an unfilterable item cannot be shown to satisfy the filter).

    Parameters
    ----------
    bbox : list of float, optional
        ``[xmin, ymin, xmax, ymax]`` in WGS84; keep only items whose bbox
        overlaps. Items without a bbox are excluded when this is set.
    datetime : str, optional
        RFC 3339 instant or ``"start/end"`` interval. Bounds are compared
        lexicographically against each item's ``datetime`` (use full
        timestamps for precise filtering — a date-only end excludes that day's
        timestamped items). Items without a datetime are excluded when this is
        set.
    categories : list of str, optional
        Restrict to these child-collection ids (see
        :data:`tanagerspec.config.TANAGER_CATEGORIES`). ``None`` = all.
    max_cloud_cover : float, optional
        Keep only items whose ``cloud_percent`` (Planet; falls back to
        ``eo:cloud_cover``) is <= this. Items without a cloud value are
        excluded when this is set.
    max_items : int, optional
        Stop after this many items.
    catalog_url : str, optional
        Override the configured catalog root (``config.TANAGER_STAC_URL``).

    Returns
    -------
    list of dict
        STAC item dicts (``id``, ``bbox``, ``geometry``, ``properties`` with an
        added ``category``, ``assets``).
    """
    import pystac

    catalog_url = catalog_url or config.TANAGER_STAC_URL
    start = end = None
    if datetime:
        parts = datetime.split("/")
        # Empty bound (e.g. "2025-01-01/" or "/2025-12-31") = open on that side.
        start = parts[0] or None
        end = (parts[1] if len(parts) > 1 else parts[0]) or None

    catalog = pystac.Catalog.from_file(catalog_url)
    out: list[dict] = []
    for child in catalog.get_children():
        if categories is not None and child.id not in categories:
            continue
        for item in child.get_items():
            cloud = item.properties.get("cloud_percent", item.properties.get("eo:cloud_cover"))
            if not _passes_filters(
                item.bbox,
                item.properties.get("datetime"),
                cloud,
                bbox=bbox,
                start=start,
                end=end,
                max_cloud_cover=max_cloud_cover,
            ):
                continue
            d = item.to_dict()
            d.setdefault("properties", {})["category"] = child.id
            out.append(d)
            if max_items is not None and len(out) >= max_items:
                logger.info("Tanager catalog traversal: %d items (capped)", len(out))
                return out
    logger.info("Tanager catalog traversal: %d items across child collections", len(out))
    return out


def query_emit_scenes(
    bbox: list[float],
    datetime: str | None = None,
    max_items: int | None = None,
    catalog_url: str | None = None,
    collection: str | None = None,
) -> list[dict]:
    """Query the NASA LP DAAC STAC for EMIT L2A scenes overlapping a bbox.

    Parameters
    ----------
    bbox : list of float
        ``[xmin, ymin, xmax, ymax]`` in WGS84.
    datetime : str, optional
        RFC 3339 datetime or interval.
    max_items : int, optional
        Cap on returned items.
    catalog_url, collection : str, optional
        Override the configured EMIT endpoint/collection.

    Returns
    -------
    list of dict
        STAC item dicts for overlapping EMIT granules.
    """
    return query_stac_items(
        catalog_url=catalog_url or config.EMIT_STAC_URL,
        collection=collection or config.EMIT_COLLECTION,
        bbox=bbox,
        datetime=datetime,
        max_items=max_items,
    )


def build_scene_inventory(items: list[dict]) -> gpd.GeoDataFrame:
    """Summarize STAC items into a scene-inventory GeoDataFrame.

    Parameters
    ----------
    items : list of dict
        STAC item dicts from a query function.

    Returns
    -------
    gpd.GeoDataFrame
        One row per scene. Columns: ``scene_id``, ``datetime``, ``category``
        (parent collection id, or ``None``), ``cloud_percent`` (``NaN`` if
        absent — never imputed), ``assets`` (the raw assets dict), and
        ``geometry`` (item geometry, or the bbox as a fallback). CRS is
        EPSG:4326.
    """
    records = []
    geometries = []
    for item in items:
        props = item.get("properties", {})
        if item.get("geometry"):
            geom = shape(item["geometry"])
        elif item.get("bbox"):
            geom = box(*item["bbox"])
        else:
            geom = None
        # Planet items report cloud cover as ``cloud_percent``; fall back to the
        # eo extension if present, else NaN (never imputed).
        cloud = props.get("cloud_percent", props.get("eo:cloud_cover", float("nan")))
        records.append(
            {
                "scene_id": item.get("id"),
                "datetime": props.get("datetime"),
                "category": props.get("category"),
                "cloud_percent": cloud,
                "assets": item.get("assets", {}),
            }
        )
        geometries.append(geom)

    gdf = gpd.GeoDataFrame(pd.DataFrame.from_records(records), geometry=geometries, crs="EPSG:4326")
    logger.info("built scene inventory with %d scenes", len(gdf))
    return gdf


def save_inventory(gdf: gpd.GeoDataFrame, path: str | Path) -> None:
    """Write a scene inventory to CSV (assets dict serialized to JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize to a plain DataFrame so geometry becomes WKT text and assets
    # become JSON; this also avoids GeoDataFrame geometry-accessor warnings.
    out = pd.DataFrame(gdf.drop(columns="geometry"))
    out["geometry"] = [g.wkt if g is not None else "" for g in gdf.geometry]
    out["assets"] = out["assets"].apply(lambda a: json.dumps(a) if a else "")
    out.to_csv(path, index=False)
    logger.info("wrote inventory to %s", path)
