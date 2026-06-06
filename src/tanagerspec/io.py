"""Reflectance-cube loading and georeferenced raster export.

Loading goes through rioxarray so the CRS and affine transform travel with the
data. Export writes compressed, tiled GeoTIFFs and always takes the CRS and
transform from the source scene — it never invents georeferencing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.transform import Affine

from .bands import assert_strictly_increasing

logger = logging.getLogger(__name__)


def load_reflectance_cube(
    href: str | Path,
    wavelengths: np.ndarray | None = None,
    masked: bool = True,
) -> tuple[xr.DataArray, np.ndarray | None]:
    """Load a multiband reflectance cube from a raster asset.

    Parameters
    ----------
    href : str or Path
        Path or URL to the raster asset. With STAC, pass
        ``item["assets"][key]["href"]`` — do not hand-construct URLs.
    wavelengths : np.ndarray, optional
        Band-center wavelengths (nm). If given, attached as the ``band``
        coordinate and validated as strictly increasing. If ``None``, the
        wavelength vector returned is ``None`` and the caller must supply it;
        wavelengths are never inferred from band metadata (that would risk
        silently inventing the spectral axis).
    masked : bool
        Passed to ``rioxarray.open_rasterio``; converts the file's nodata to
        ``NaN`` (forces a float dtype).

    Returns
    -------
    cube : xr.DataArray
        Dims ``("band", "y", "x")`` with CRS/transform on the ``.rio`` accessor.
    wavelengths : np.ndarray or None
        The wavelength vector used, or ``None`` if it could not be determined.
    """
    cube = rioxarray.open_rasterio(href, masked=masked)
    if cube.ndim != 3:
        raise ValueError(f"expected a 3-D (band, y, x) raster, got dims {cube.dims}")

    if wavelengths is not None:
        wl = np.asarray(wavelengths, dtype=float)
        if wl.size != cube.sizes["band"]:
            raise ValueError(
                f"wavelengths length {wl.size} != band count {cube.sizes['band']}"
            )
        assert_strictly_increasing(wl)
        cube = cube.assign_coords(band=wl)
        wavelengths = wl
    else:
        # We deliberately do NOT parse wavelengths from band descriptions/tags:
        # the Tanager metadata field is unconfirmed, and guessing (e.g. parsing
        # the first token of a label) could silently invent a wavelength axis.
        # The caller must supply wavelengths explicitly. TODO: once the product
        # metadata field is confirmed, add an explicit, validated reader.
        logger.warning("no wavelengths supplied; caller must provide them explicitly")

    logger.info("loaded cube %s with %d bands", tuple(cube.sizes.values()), cube.sizes["band"])
    return cube, wavelengths


def grid_from_positions(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    spatial_shape: tuple[int, int],
    fill: float = np.nan,
) -> np.ndarray:
    """Scatter per-pixel values back onto a 2-D grid.

    Parameters
    ----------
    values : np.ndarray
        Per-pixel values, shape ``(n_pixels,)``.
    rows, cols : np.ndarray
        Pixel positions, shape ``(n_pixels,)``.
    spatial_shape : tuple of (int, int)
        ``(n_rows, n_cols)`` of the output grid.
    fill : float
        Value for positions not in the sample.

    Returns
    -------
    np.ndarray
        2-D grid, shape ``spatial_shape``.
    """
    grid = np.full(spatial_shape, fill, dtype=float)
    grid[rows, cols] = values
    return grid


def write_geotiff(
    array2d: np.ndarray,
    transform: Affine,
    crs,
    path: str | Path,
    nodata: float | None = None,
    dtype: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Write a 2-D array to a compressed, tiled GeoTIFF.

    Parameters
    ----------
    array2d : np.ndarray
        Raster data, shape ``(n_rows, n_cols)``.
    transform : affine.Affine
        Affine transform from the source scene.
    crs : Any
        CRS from the source scene (rasterio CRS, pyproj CRS, or WKT/EPSG).
    path : str or Path
        Output path.
    nodata : float, optional
        Nodata value to record.
    dtype : str, optional
        Output dtype; defaults to the array's dtype.
    tags : dict, optional
        Key/value tags written into the file metadata (e.g. provenance).
    """
    if array2d.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {array2d.shape}")
    dtype = dtype or str(array2d.dtype)
    is_float = np.issubdtype(np.dtype(dtype), np.floating)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array2d.shape[0],
        "width": array2d.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 3 if is_float else 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array2d.astype(dtype), 1)
        if tags:
            dst.update_tags(**tags)
    logger.info("wrote %s", path)


def positions_to_geotiff(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    spatial_shape: tuple[int, int],
    transform: Affine,
    crs,
    path: str | Path,
    nodata: float = np.nan,
    dtype: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Scatter per-pixel values onto a grid and write a GeoTIFF in one step.

    Convenience wrapper over :func:`grid_from_positions` and
    :func:`write_geotiff` for the common "sparse sample back to map" case.
    """
    grid = grid_from_positions(values, rows, cols, spatial_shape, fill=nodata)
    write_geotiff(grid, transform, crs, path, nodata=nodata, dtype=dtype, tags=tags)
