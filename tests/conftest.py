"""Shared synthetic fixtures. No real data, no network."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


def make_cube(
    n_band: int = 12, ny: int = 8, nx: int = 8, seed: int = 0
) -> tuple[xr.DataArray, np.ndarray]:
    """Build a small synthetic reflectance cube and its wavelength vector."""
    rng = np.random.default_rng(seed)
    data = rng.random((n_band, ny, nx)).astype(float)
    wl = np.linspace(400.0, 2400.0, n_band)
    cube = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": wl, "y": np.arange(ny), "x": np.arange(nx)},
    )
    return cube, wl


@pytest.fixture
def cube_and_wavelengths() -> tuple[xr.DataArray, np.ndarray]:
    return make_cube()


@pytest.fixture
def source_wavelengths() -> np.ndarray:
    """A strictly increasing source grid spanning the VSWIR range."""
    return np.linspace(400.0, 2400.0, 200)
