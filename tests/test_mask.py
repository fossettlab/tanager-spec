from __future__ import annotations

import numpy as np
import pytest
from conftest import make_cube

from tanager_spec import mask


def test_mask_absorption_bands_sets_nan_in_windows():
    cube, wl = make_cube(n_band=12)
    # Anchor a window tightly around an actual band center so the test is not
    # sensitive to the synthetic grid spacing.
    center = float(wl[6])
    windows = [(center - 5.0, center + 5.0)]
    out = mask.mask_absorption_bands(cube, wl, windows)
    from tanager_spec.bands import indices_in_windows

    in_win = indices_in_windows(wl, windows)
    assert in_win.any()
    assert np.isnan(out.values[in_win, :, :]).all()
    # Bands outside the window are untouched.
    assert np.isfinite(out.values[~in_win, :, :]).all()


def test_mask_absorption_does_not_mutate_input():
    cube, wl = make_cube()
    before = cube.values.copy()
    mask.mask_absorption_bands(cube, wl, [(1350.0, 1450.0)])
    assert np.array_equal(cube.values, before)


def test_invalid_pixel_mask_flags_nan():
    cube, _ = make_cube(n_band=5, ny=4, nx=4)
    cube.values[2, 1, 1] = np.nan  # one band NaN at (1,1)
    invalid = mask.invalid_pixel_mask(cube)
    assert bool(invalid.values[1, 1]) is True
    assert invalid.sum().item() == 1


def test_invalid_pixel_mask_flags_nodata_and_range():
    cube, _ = make_cube(n_band=5, ny=4, nx=4)
    cube.values[:, 0, 0] = -9999.0
    cube.values[0, 2, 2] = 5.0  # out of [0, 1] range
    invalid = mask.invalid_pixel_mask(cube, nodata=-9999.0, valid_range=(0.0, 1.0))
    assert bool(invalid.values[0, 0]) is True
    assert bool(invalid.values[2, 2]) is True


def test_invalid_pixel_mask_qa():
    cube, _ = make_cube(n_band=3, ny=3, nx=3)
    qa = cube.isel(band=0).copy()
    qa.values[:] = 1
    qa.values[0, 0] = 9  # not in valid set
    invalid = mask.invalid_pixel_mask(cube, qa=qa, qa_valid_values=[1])
    assert bool(invalid.values[0, 0]) is True
    assert invalid.sum().item() == 1


def test_invalid_pixel_mask_flags_inf():
    cube, _ = make_cube(n_band=4, ny=3, nx=3)
    cube.values[1, 2, 2] = np.inf
    invalid = mask.invalid_pixel_mask(cube)
    assert bool(invalid.values[2, 2]) is True


def test_require_band_y_x_transposes():
    cube, wl = make_cube(n_band=5, ny=4, nx=4)
    # Hand mask_absorption_bands a cube with a non-canonical dim order.
    reordered = cube.transpose("y", "x", "band")
    out = mask.mask_absorption_bands(reordered, wl, [(float(wl[2]) - 5, float(wl[2]) + 5)])
    assert out.dims == ("band", "y", "x")
    assert np.isnan(out.values[2, :, :]).all()


def test_require_band_y_x_missing_dim_raises():
    cube, _ = make_cube(n_band=3, ny=3, nx=3)
    flat = cube.isel(band=0)  # dims ("y", "x") only
    with pytest.raises(ValueError, match="missing required dims"):
        mask.require_band_y_x(flat)


def test_invalid_pixel_mask_rejects_bad_qa_dims():
    cube, _ = make_cube(n_band=3, ny=3, nx=3)
    bad_qa = cube.copy()  # dims ("band", "y", "x"), not ("y", "x")
    with pytest.raises(ValueError, match="qa must have dims"):
        mask.invalid_pixel_mask(cube, qa=bad_qa, qa_valid_values=[1])


def test_invalid_pixel_mask_qa_combined_positionally_despite_coord_mismatch():
    # qa has the same (y, x) shape but DIFFERENT coordinate labels than the
    # cube. xarray's `|` would align by coordinate and silently empty the mask;
    # the positional combine must instead flag the right pixel.
    cube, _ = make_cube(n_band=3, ny=4, nx=4)
    qa = cube.isel(band=0).copy()
    qa.values[:] = 1
    qa.values[0, 0] = 9  # invalid at (0, 0)
    qa = qa.assign_coords(y=qa["y"].values + 1000, x=qa["x"].values + 1000)
    invalid = mask.invalid_pixel_mask(cube, qa=qa, qa_valid_values=[1])
    assert invalid.shape == (4, 4)
    assert bool(invalid.values[0, 0]) is True
    assert invalid.sum().item() == 1
