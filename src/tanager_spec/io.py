"""Reflectance-cube loading and georeferenced raster export.

Loading goes through rioxarray so the CRS and affine transform travel with the
data. Export writes compressed, tiled GeoTIFFs and always takes the CRS and
transform from the source scene — it never invents georeferencing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.transform import Affine

from . import config
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
            raise ValueError(f"wavelengths length {wl.size} != band count {cube.sizes['band']}")
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


def _parse_hdfeos_utm_grid(
    struct_metadata: str, grid_name: str
) -> tuple[str, Affine, tuple[int, int]]:
    """Parse CRS, affine transform, and shape from an HDF-EOS5 StructMetadata.

    Parameters
    ----------
    struct_metadata : str
        Contents of ``HDFEOS INFORMATION/StructMetadata.0``.
    grid_name : str
        Grid whose georeferencing to read (e.g. ``"HYP"``).

    Returns
    -------
    crs : str
        EPSG code string, e.g. ``"EPSG:32637"``.
    transform : affine.Affine
        Pixel->CRS affine transform (upper-left origin).
    shape : tuple of (int, int)
        ``(n_rows, n_cols)`` = ``(YDim, XDim)``.

    Raises
    ------
    NotImplementedError
        If the grid projection is not UTM (the only Tanager case seen).
    ValueError
        If required fields are missing.
    """

    # Scope parsing to the GROUP=GRID_N block whose GridName matches, so a
    # multi-grid file cannot silently attach another grid's georeferencing.
    block = None
    for chunk in re.split(r"GROUP=GRID_\d+", struct_metadata):
        m = re.search(r'GridName="([^"]+)"', chunk)
        if m is not None and m.group(1) == grid_name:
            block = chunk
            break
    if block is None:
        raise ValueError(f"grid {grid_name!r} not found in StructMetadata")

    def _find(key: str) -> str:
        m = re.search(rf"\b{key}=([^\n]+)", block)
        if m is None:
            raise ValueError(f"{key} not found in StructMetadata grid {grid_name!r}")
        return m.group(1).strip()

    projection = _find("Projection")
    if "UTM" not in projection.upper():
        raise NotImplementedError(
            f"only UTM grids are supported; StructMetadata says Projection={projection}"
        )
    xdim = int(_find("XDim"))
    ydim = int(_find("YDim"))
    ulx, uly = (float(v) for v in re.findall(r"[-\d.]+", _find("UpperLeftPointMtrs")))
    lrx, lry = (float(v) for v in re.findall(r"[-\d.]+", _find("LowerRightMtrs")))
    zone = int(_find("ZoneCode"))
    # GCTP encodes the southern hemisphere as a negative zone code.
    epsg = (32600 if zone > 0 else 32700) + abs(zone)
    px = (lrx - ulx) / xdim
    py = (uly - lry) / ydim
    transform = Affine(px, 0.0, ulx, 0.0, -py, uly)
    return f"EPSG:{epsg}", transform, (ydim, xdim)


def load_tanager_sr_hdf5(
    path: str | Path,
    field: str | None = None,
    grid: str | None = None,
    masked: bool = True,
    bands: slice | None = None,
    fill_value: float | None = None,
) -> tuple[xr.DataArray, np.ndarray]:
    """Load a Tanager surface-reflectance cube from an ortho/basic SR HDF5 file.

    Reads the HDF-EOS5 surface-reflectance grid written by Planet's Tanager
    products. Wavelengths and the fill value come from the dataset's own
    attributes (authoritative, not inferred); CRS and transform are parsed from
    the file's StructMetadata.

    Parameters
    ----------
    path : str or Path
        Local path to the ``*_ortho_sr_hdf5.h5`` (or basic SR) file. Download
        it first from the STAC asset href; HDF5 is not read over HTTP here.
    field : str, optional
        Reflectance field name. Defaults to ``config.TANAGER_SR_FIELD``.
    grid : str, optional
        HDF-EOS grid name. Defaults to ``config.TANAGER_HDF5_GRID``.
    masked : bool
        Replace the fill value with ``NaN``.
    bands : slice, optional
        Optional band subset (the full cube is ~1.1 GB in memory for a scene;
        pass a slice to limit what is read).
    fill_value : float, optional
        Override the fill value. If ``None``, it is read from the dataset's
        ``_FillValue`` attribute; if that attribute is absent, a ``ValueError``
        is raised rather than guessing one.

    Returns
    -------
    cube : xr.DataArray
        Dims ``("band", "y", "x")`` with ``band`` = wavelength (nm), and CRS /
        transform on the ``.rio`` accessor.
    wavelengths : np.ndarray
        Band-center wavelengths (nm), strictly increasing.
    """
    import h5py

    field = field or config.TANAGER_SR_FIELD
    grid = grid or config.TANAGER_HDF5_GRID
    with h5py.File(path, "r") as f:
        ds = f[f"HDFEOS/GRIDS/{grid}/Data Fields/{field}"]
        if "wavelengths" not in ds.attrs:
            raise ValueError(f"{field} has no 'wavelengths' attribute in {path}")
        wavelengths = np.asarray(ds.attrs["wavelengths"], dtype=float)
        if fill_value is not None:
            fill = float(fill_value)
        elif "_FillValue" in ds.attrs:
            fill = float(ds.attrs["_FillValue"])
        else:
            raise ValueError(
                f"{field} has no '_FillValue' attribute in {path}; pass fill_value= "
                "explicitly rather than assume one"
            )
        data = (ds[bands, :, :] if bands is not None else ds[...]).astype("float32")
        if bands is not None:
            wavelengths = wavelengths[bands]
        struct = f["HDFEOS INFORMATION/StructMetadata.0"][()]
    if isinstance(struct, bytes):
        struct = struct.decode(errors="replace")

    crs, transform, (ny, nx) = _parse_hdfeos_utm_grid(struct, grid)
    if data.shape[1:] != (ny, nx):
        raise ValueError(f"data spatial shape {data.shape[1:]} != StructMetadata grid {(ny, nx)}")
    if masked:
        data = np.where(data == fill, np.nan, data)
    assert_strictly_increasing(wavelengths)

    # Cell-center coordinates from the upper-left-origin transform.
    xs = transform.c + (np.arange(nx) + 0.5) * transform.a
    ys = transform.f + (np.arange(ny) + 0.5) * transform.e
    cube = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": wavelengths, "y": ys, "x": xs},
    )
    cube = cube.rio.write_crs(crs).rio.write_transform(transform)
    logger.info(
        "loaded Tanager SR cube %s (%d bands) %s from %s",
        (ny, nx),
        wavelengths.size,
        crs,
        path,
    )
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
