"""Invalid-pixel and absorption-band masking for reflectance cubes.

Convention: a *pixel* invalid-mask is a 2-D boolean array where ``True`` means
**invalid** (exclude this pixel). Band masking sets unreliable spectral
channels to ``NaN`` in place of the data.

All cubes are :class:`xarray.DataArray` with dims ``("band", "y", "x")``.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from .bands import indices_in_windows
from .config import ABSORPTION_MASKS_NM

logger = logging.getLogger(__name__)

_REQUIRED_DIMS = ("band", "y", "x")


def require_band_y_x(cube: xr.DataArray) -> xr.DataArray:
    """Validate a cube has dims ``("band", "y", "x")`` and return it in that order.

    Functions that drop to ``.values`` rely on positional axes, so a cube with
    missing or differently ordered dims would silently index the wrong pixels.
    This transposes by name (cheap, lazy) after checking the dims are present.

    Parameters
    ----------
    cube : xr.DataArray
        Reflectance cube.

    Returns
    -------
    xr.DataArray
        The same data with dims ordered ``("band", "y", "x")``.

    Raises
    ------
    ValueError
        If any of the required dims is missing.
    """
    missing = [d for d in _REQUIRED_DIMS if d not in cube.dims]
    if missing:
        raise ValueError(
            f"cube is missing required dims {missing}; has {tuple(cube.dims)}"
        )
    if tuple(cube.dims) != _REQUIRED_DIMS:
        cube = cube.transpose(*_REQUIRED_DIMS)
    return cube


def invalid_pixel_mask(
    cube: xr.DataArray,
    nodata: float | None = None,
    valid_range: tuple[float, float] | None = None,
    qa: xr.DataArray | None = None,
    qa_valid_values: list[int] | None = None,
) -> xr.DataArray:
    """Build a per-pixel invalid mask for a reflectance cube.

    A pixel is invalid if it is invalid in *any* band, so the returned mask is
    safe to apply uniformly across the spectral axis.

    Parameters
    ----------
    cube : xr.DataArray
        Reflectance cube, dims ``("band", "y", "x")``.
    nodata : float, optional
        Sentinel value flagged as invalid wherever it appears. ``NaN`` is
        always treated as invalid regardless of this argument.
    valid_range : tuple of (float, float), optional
        ``(low, high)`` inclusive bounds; any band value outside is flagged
        invalid. Useful to drop saturated or physically impossible reflectance.
    qa : xr.DataArray, optional
        Quality layer, dims ``("y", "x")``. Pixels whose QA value is not in
        ``qa_valid_values`` are flagged invalid.
    qa_valid_values : list of int, optional
        QA values considered valid. Required if ``qa`` is given.

    Returns
    -------
    xr.DataArray
        Boolean mask, dims ``("y", "x")``. ``True`` = invalid (exclude).
    """
    cube = require_band_y_x(cube)
    # Non-finite (NaN or +/-inf) in any band makes the pixel invalid.
    invalid = (~np.isfinite(cube)).any(dim="band")

    if nodata is not None:
        invalid = invalid | (cube == nodata).any(dim="band")

    if valid_range is not None:
        low, high = valid_range
        out_of_range = (cube < low) | (cube > high)
        invalid = invalid | out_of_range.any(dim="band")

    if qa is not None:
        if qa_valid_values is None:
            raise ValueError("qa_valid_values is required when qa is provided")
        if tuple(qa.dims) != ("y", "x"):
            raise ValueError(f"qa must have dims ('y', 'x'), got {tuple(qa.dims)}")
        if qa.sizes.get("y") != cube.sizes["y"] or qa.sizes.get("x") != cube.sizes["x"]:
            raise ValueError("qa spatial shape does not match cube")
        # Combine positionally on the validated grid. Using xarray's `|` here
        # would re-align qa by coordinate label, which silently empties or
        # shifts the mask if qa carries different y/x labels (e.g. loaded from a
        # separate raster). We have already checked dims and shape, so position
        # is the intended semantics.
        qa_invalid = ~np.isin(np.asarray(qa.values), qa_valid_values)
        invalid = invalid | xr.DataArray(qa_invalid, dims=("y", "x"), coords=invalid.coords)

    invalid = invalid.astype(bool)
    frac = float(invalid.mean())
    logger.info("invalid pixel fraction: %.3f", frac)
    return invalid


def mask_absorption_bands(
    cube: xr.DataArray,
    wavelengths: np.ndarray,
    windows: list[tuple[float, float]] | None = None,
) -> xr.DataArray:
    """Set atmospheric absorption-band channels to ``NaN``.

    Parameters
    ----------
    cube : xr.DataArray
        Reflectance cube, dims ``("band", "y", "x")``.
    wavelengths : np.ndarray
        Band-center wavelengths (nm), length matching the band dimension.
    windows : list of (float, float), optional
        Wavelength windows (nm) to mask. Defaults to
        :data:`tanagerspec.config.ABSORPTION_MASKS_NM`.

    Returns
    -------
    xr.DataArray
        A copy of the cube with masked channels set to ``NaN``.
    """
    if windows is None:
        windows = ABSORPTION_MASKS_NM
    cube = require_band_y_x(cube)
    wl = np.asarray(wavelengths, dtype=float)
    if wl.size != cube.sizes["band"]:
        raise ValueError(
            f"wavelengths length {wl.size} != band dimension {cube.sizes['band']}"
        )
    in_window = indices_in_windows(wl, windows)
    n_masked = int(in_window.sum())
    logger.info("masking %d absorption-band channels across %d windows", n_masked, len(windows))
    out = cube.astype(float).copy()
    out.values[in_window, :, :] = np.nan
    return out
