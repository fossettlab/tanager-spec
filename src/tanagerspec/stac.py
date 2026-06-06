"""STAC queries for Tanager and EMIT scenes.

Asset URLs are always resolved through pystac-client; they are never
hand-assembled or cached across sessions. The Planet Tanager endpoint and
collection are not yet confirmed (see :mod:`tanagerspec.config`); the query
helpers raise a clear error rather than contacting an invented URL.
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
    max_cloud_cover: float | None = None,
    max_items: int | None = None,
    catalog_url: str | None = None,
    collection: str | None = None,
) -> list[dict]:
    """Query the Planet Open STAC catalog for Tanager scenes.

    Parameters
    ----------
    bbox : list of float, optional
        ``[xmin, ymin, xmax, ymax]`` in WGS84.
    datetime : str, optional
        RFC 3339 datetime or interval.
    max_cloud_cover : float, optional
        Maximum ``eo:cloud_cover`` (percent) to retain, if the property exists.
    max_items : int, optional
        Cap on returned items.
    catalog_url, collection : str, optional
        Override the configured endpoint/collection. If both these and the
        config defaults are ``None``, a ``ValueError`` is raised.

    Returns
    -------
    list of dict
        STAC item dicts.

    Raises
    ------
    ValueError
        If the catalog URL or collection is unset (see config TODOs).
    """
    catalog_url = catalog_url or config.TANAGER_STAC_URL
    collection = collection or config.TANAGER_COLLECTION
    if catalog_url is None or collection is None:
        raise ValueError(
            "Tanager STAC endpoint/collection is not configured. Set "
            "tanagerspec.config.TANAGER_STAC_URL and TANAGER_COLLECTION (or pass "
            "catalog_url= and collection=) once confirmed from Planet's "
            "Open Data Competition documentation."
        )
    query = None
    if max_cloud_cover is not None:
        query = {"eo:cloud_cover": {"lte": max_cloud_cover}}
    return query_stac_items(
        catalog_url=catalog_url,
        collection=collection,
        bbox=bbox,
        datetime=datetime,
        query=query,
        max_items=max_items,
    )


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
        One row per scene. Columns: ``scene_id``, ``datetime``,
        ``cloud_cover`` (``NaN`` if absent — never imputed), ``assets`` (the
        raw assets dict), and ``geometry`` (item geometry, or the bbox as a
        fallback). CRS is EPSG:4326.
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
        records.append(
            {
                "scene_id": item.get("id"),
                "datetime": props.get("datetime"),
                "cloud_cover": props.get("eo:cloud_cover", float("nan")),
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
