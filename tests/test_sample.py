from __future__ import annotations

import numpy as np
import pytest
from conftest import make_cube

from tanager_spec import sample


def test_sample_is_deterministic_with_seed():
    cube, _ = make_cube(n_band=6, ny=20, nx=20)
    a = sample.sample_pixels(cube, n_pixels=50, seed=42)
    b = sample.sample_pixels(cube, n_pixels=50, seed=42)
    assert np.array_equal(a.flat_indices, b.flat_indices)


def test_sample_returns_requested_count():
    cube, _ = make_cube(n_band=6, ny=20, nx=20)
    result = sample.sample_pixels(cube, n_pixels=50, seed=1)
    assert result.spectra.shape == (50, 6)
    assert result.rows.size == 50


def test_stratified_returns_exact_count():
    cube, _ = make_cube(n_band=6, ny=40, nx=40)
    result = sample.sample_pixels(cube, n_pixels=137, n_blocks=4, seed=3)
    assert result.spectra.shape[0] == 137


def test_returns_all_when_requesting_more_than_valid():
    cube, _ = make_cube(n_band=4, ny=5, nx=5)  # 25 valid pixels
    result = sample.sample_pixels(cube, n_pixels=1000, seed=0)
    assert result.spectra.shape[0] == 25
    assert result.n_valid == 25


def test_excludes_invalid_and_nan_pixels():
    cube, _ = make_cube(n_band=4, ny=5, nx=5)
    cube.values[1, 0, 0] = np.nan  # NaN pixel excluded
    result = sample.sample_pixels(cube, n_pixels=None, seed=0)
    assert result.spectra.shape[0] == 24
    assert not np.any((result.rows == 0) & (result.cols == 0))


def test_spectra_match_cube_values():
    cube, _ = make_cube(n_band=4, ny=6, nx=6)
    result = sample.sample_pixels(cube, n_pixels=10, seed=7)
    for i in range(result.spectra.shape[0]):
        r, c = result.rows[i], result.cols[i]
        assert np.array_equal(result.spectra[i], cube.values[:, r, c])


def test_flat_indices_consistent_with_rowcol():
    cube, _ = make_cube(n_band=3, ny=8, nx=10)
    result = sample.sample_pixels(cube, n_pixels=20, seed=2)
    assert np.array_equal(result.flat_indices, result.rows * 10 + result.cols)


def test_rejects_bad_n_blocks():
    cube, _ = make_cube(n_band=3, ny=5, nx=5)
    with pytest.raises(ValueError, match="n_blocks"):
        sample.sample_pixels(cube, n_pixels=5, n_blocks=0)


def test_rejects_negative_n_pixels():
    cube, _ = make_cube(n_band=3, ny=5, nx=5)
    with pytest.raises(ValueError, match="n_pixels"):
        sample.sample_pixels(cube, n_pixels=-1)


def test_rejects_mismatched_invalid_shape():
    import xarray as xr

    cube, _ = make_cube(n_band=3, ny=6, nx=6)
    # dims are ("y", "x") but the y extent is wrong -> would broadcast silently.
    bad_invalid = xr.DataArray(
        np.zeros((1, 6), dtype=bool), dims=("y", "x")
    )
    with pytest.raises(ValueError, match="spatial shape"):
        sample.sample_pixels(cube, invalid=bad_invalid, n_pixels=4)


def test_stratified_proportional_when_fewer_pixels_than_blocks():
    # 64 blocks but only 3 pixels requested: largest-remainder gives exactly 3
    # (the old ">=1 per block" rule would have overshot then trimmed).
    cube, _ = make_cube(n_band=4, ny=40, nx=40, seed=5)
    result = sample.sample_pixels(cube, n_pixels=3, n_blocks=8, seed=5)
    assert result.spectra.shape[0] == 3
    again = sample.sample_pixels(cube, n_pixels=3, n_blocks=8, seed=5)
    assert np.array_equal(result.flat_indices, again.flat_indices)
