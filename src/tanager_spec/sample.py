"""Reproducible pixel sampling from reflectance cubes.

Sampling is seeded so a fresh clone reproduces the same pixel set. Optional
spatial-block stratification (``n_blocks > 1``) spreads the sample across the
scene to reduce the spatial-autocorrelation bias that inflates within-scene
metric estimates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .config import SEED
from .mask import require_band_y_x

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """A sampled set of valid pixel spectra and their grid positions.

    Attributes
    ----------
    spectra : np.ndarray
        Sampled spectra, shape ``(n_sampled, n_bands)``.
    rows : np.ndarray
        Row indices, shape ``(n_sampled,)``.
    cols : np.ndarray
        Column indices, shape ``(n_sampled,)``.
    flat_indices : np.ndarray
        Flat indices into the ``(n_rows * n_cols)`` grid, for rebuilding maps.
    n_valid : int
        Total valid pixels available before subsampling.
    spatial_shape : tuple of (int, int)
        ``(n_rows, n_cols)`` of the source grid.
    """

    spectra: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    flat_indices: np.ndarray
    n_valid: int
    spatial_shape: tuple[int, int]


def _valid_positions(
    cube: xr.DataArray, invalid: xr.DataArray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (row, col) indices of valid pixels (all bands finite, not invalid)."""
    bad = (~np.isfinite(cube)).any(dim="band").values
    if invalid is not None:
        if tuple(invalid.dims) != ("y", "x"):
            raise ValueError(f"invalid must have dims ('y', 'x'), got {tuple(invalid.dims)}")
        if invalid.sizes.get("y") != cube.sizes["y"] or invalid.sizes.get("x") != cube.sizes["x"]:
            raise ValueError(
                "invalid mask spatial shape "
                f"{(invalid.sizes.get('y'), invalid.sizes.get('x'))} does not match cube "
                f"{(cube.sizes['y'], cube.sizes['x'])}"
            )
        bad = bad | invalid.values.astype(bool)
    rows, cols = np.where(~bad)
    return rows, cols


def sample_pixels(
    cube: xr.DataArray,
    invalid: xr.DataArray | None = None,
    n_pixels: int | None = None,
    n_blocks: int = 1,
    seed: int = SEED,
) -> SampleResult:
    """Draw a reproducible sample of valid pixels from a cube.

    Parameters
    ----------
    cube : xr.DataArray
        Reflectance cube, dims ``("band", "y", "x")``.
    invalid : xr.DataArray, optional
        Per-pixel invalid mask, dims ``("y", "x")``, ``True`` = invalid.
        Pixels with any non-finite (``NaN`` or ``±inf``) band are also excluded
        regardless.
    n_pixels : int, optional
        Target sample size. If ``None`` or larger than the number of valid
        pixels, all valid pixels are returned (a warning is logged on
        shortfall).
    n_blocks : int
        Number of spatial blocks per axis for stratification. ``1`` means a
        plain random sample; ``k`` divides the scene into a ``k x k`` grid and
        samples proportionally from each block's valid pixels.
    seed : int
        Random seed.

    Returns
    -------
    SampleResult
    """
    if n_pixels is not None and n_pixels < 0:
        raise ValueError(f"n_pixels must be None or >= 0, got {n_pixels}")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    cube = require_band_y_x(cube)
    n_rows, n_cols = cube.sizes["y"], cube.sizes["x"]
    rows, cols = _valid_positions(cube, invalid)
    n_valid = rows.size
    rng = np.random.default_rng(seed)

    if n_valid == 0:
        logger.warning("no valid pixels available to sample")
        sel = np.empty(0, dtype=int)
    elif n_pixels is None or n_pixels >= n_valid:
        if n_pixels is not None and n_pixels > n_valid:
            logger.warning("requested %d pixels but only %d valid; using all", n_pixels, n_valid)
        sel = np.arange(n_valid)
    elif n_blocks <= 1:
        sel = rng.choice(n_valid, size=n_pixels, replace=False)
    else:
        sel = _stratified_select(rows, cols, n_rows, n_cols, n_pixels, n_blocks, rng)

    sel_rows = rows[sel]
    sel_cols = cols[sel]
    flat_indices = sel_rows * n_cols + sel_cols
    # cube.values is (band, y, x); gather spectra at sampled positions.
    spectra = cube.values[:, sel_rows, sel_cols].T  # (n_sampled, n_bands)

    logger.info("sampled %d of %d valid pixels (n_blocks=%d)", sel.size, n_valid, n_blocks)
    return SampleResult(
        spectra=spectra,
        rows=sel_rows,
        cols=sel_cols,
        flat_indices=flat_indices,
        n_valid=n_valid,
        spatial_shape=(n_rows, n_cols),
    )


def _stratified_select(
    rows: np.ndarray,
    cols: np.ndarray,
    n_rows: int,
    n_cols: int,
    n_pixels: int,
    n_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select sample indices proportionally across an ``n_blocks x n_blocks`` grid.

    Uses largest-remainder (Hamilton) apportionment: each block's quota is its
    proportional share floored, then the leftover is handed out one at a time to
    the blocks with the largest fractional remainders (skipping blocks already
    at capacity). No forced per-block minimum, so when ``n_pixels`` is smaller
    than the number of non-empty blocks the allocation stays proportional
    instead of overrepresenting sparse blocks. Caller guarantees
    ``0 <= n_pixels < n_valid``.
    """
    row_block = np.minimum((rows * n_blocks) // n_rows, n_blocks - 1)
    col_block = np.minimum((cols * n_blocks) // n_cols, n_blocks - 1)
    block_id = row_block * n_blocks + col_block

    block_ids = np.unique(block_id)
    members = {int(b): np.where(block_id == b)[0] for b in block_ids}
    sizes = np.array([members[int(b)].size for b in block_ids], dtype=float)
    n_valid = rows.size

    ideal = n_pixels * sizes / n_valid
    quota = np.floor(ideal).astype(int)
    quota = np.minimum(quota, sizes.astype(int))  # never exceed block membership

    # Distribute the remaining picks by largest fractional remainder, only to
    # blocks that still have spare members.
    leftover = n_pixels - int(quota.sum())
    if leftover > 0:
        remainder = ideal - np.floor(ideal)
        order = np.argsort(-remainder)  # largest remainder first
        for idx in order:
            if leftover == 0:
                break
            if quota[idx] < sizes[idx]:
                quota[idx] += 1
                leftover -= 1

    selected = [
        rng.choice(members[int(b)], size=quota[i], replace=False)
        for i, b in enumerate(block_ids)
        if quota[i] > 0
    ]
    if not selected:
        return np.empty(0, dtype=int)
    return np.concatenate(selected)
