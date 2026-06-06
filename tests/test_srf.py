from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tanagerspec import srf


def _toy_srf(source_wavelengths: np.ndarray) -> srf.SpectralResponse:
    """A few broad Gaussian bands well inside the source range (test only)."""
    return srf.gaussian_srf(
        band_names=["G1", "G2", "G3"],
        centers_nm=np.array([600.0, 1200.0, 2000.0]),
        fwhm_nm=np.array([100.0, 150.0, 200.0]),
        grid_nm=source_wavelengths,
        platform="TEST",
    )


def test_constant_reflectance_maps_to_constant(source_wavelengths):
    """The defining correctness check: a flat spectrum convolves to its value."""
    response = _toy_srf(source_wavelengths)
    n_src = source_wavelengths.size
    spectra = np.full((5, n_src), 0.3)
    out = srf.simulate(spectra, source_wavelengths, response)
    assert out.shape == (5, 3)
    assert np.allclose(out, 0.3, atol=1e-9)


def test_band_order_matches_names(source_wavelengths):
    response = _toy_srf(source_wavelengths)
    assert response.band_names == ["G1", "G2", "G3"]
    out = srf.simulate(np.full((1, source_wavelengths.size), 0.5), source_wavelengths, response)
    assert out.shape == (1, 3)


def test_all_nan_pixel_yields_nan(source_wavelengths):
    response = _toy_srf(source_wavelengths)
    spectra = np.full((2, source_wavelengths.size), 0.4)
    spectra[1, :] = np.nan
    out = srf.simulate(spectra, source_wavelengths, response)
    assert np.all(np.isfinite(out[0]))
    assert np.all(np.isnan(out[1]))


def test_partial_mask_keeps_broad_band_finite(source_wavelengths):
    """A few NaN channels in a broad band's support should not kill the band."""
    response = _toy_srf(source_wavelengths)
    spectra = np.full((1, source_wavelengths.size), 0.4)
    # NaN two channels near the G2 (1200 nm) center.
    near = np.argsort(np.abs(source_wavelengths - 1200.0))[:2]
    spectra[0, near] = np.nan
    out = srf.simulate(spectra, source_wavelengths, response, min_coverage=0.5)
    assert np.all(np.isfinite(out))


def test_low_coverage_band_yields_nan():
    """A narrow band whose support is entirely NaN drops to NaN."""
    grid = np.linspace(400.0, 2400.0, 200)
    response = srf.gaussian_srf(["narrow"], np.array([1200.0]), np.array([8.0]), grid)
    spectra = np.full((1, grid.size), 0.4)
    support = response.response[0] > 1e-3
    spectra[0, support] = np.nan
    out = srf.simulate(spectra, grid, response, min_coverage=0.5)
    assert np.isnan(out[0, 0])


def test_shape_mismatch_raises(source_wavelengths):
    response = _toy_srf(source_wavelengths)
    with pytest.raises(ValueError, match="bands"):
        srf.simulate(np.zeros((3, 5)), source_wavelengths, response)


def test_spectral_response_shape_validation():
    with pytest.raises(ValueError, match="response shape"):
        srf.SpectralResponse(
            band_names=["a", "b"],
            wavelength_nm=np.array([400.0, 500.0, 600.0]),
            response=np.zeros((2, 2)),  # wrong: n_grid should be 3
        )


def test_gaussian_length_mismatch_raises(source_wavelengths):
    with pytest.raises(ValueError, match="equal length"):
        srf.gaussian_srf(["a", "b"], np.array([600.0]), np.array([50.0]), source_wavelengths)


def test_load_srf_csv_roundtrip_and_simulate(tmp_path):
    """End-to-end: load a published-style CSV and simulate a constant spectrum."""
    grid = np.linspace(400.0, 2400.0, 60)
    df = pd.DataFrame(
        {
            "wavelength_nm": grid,
            "B1": np.exp(-0.5 * ((grid - 600.0) / 40.0) ** 2),
            "B2": np.exp(-0.5 * ((grid - 1500.0) / 60.0) ** 2),
        }
    )
    csv = tmp_path / "srf.csv"
    df.to_csv(csv, index=False)

    response = srf.load_srf_csv(csv, source="unit test")
    assert response.band_names == ["B1", "B2"]
    out = srf.simulate(np.full((4, grid.size), 0.25), grid, response)
    assert out.shape == (4, 2)
    assert np.allclose(out, 0.25, atol=1e-9)


def test_load_srf_csv_rejects_non_monotone_wavelengths(tmp_path):
    grid = np.linspace(400.0, 2400.0, 30)[::-1]  # descending -> invalid xp
    df = pd.DataFrame({"wavelength_nm": grid, "B1": np.ones_like(grid)})
    csv = tmp_path / "bad.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        srf.load_srf_csv(csv)


def test_load_srf_csv_missing_band_column_raises(tmp_path):
    grid = np.linspace(400.0, 2400.0, 10)
    df = pd.DataFrame({"wavelength_nm": grid, "B1": np.ones_like(grid)})
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="not found"):
        srf.load_srf_csv(csv, band_columns=["B1", "B99"])


def test_simulate_rejects_non_monotone_source_wavelengths(source_wavelengths):
    response = _toy_srf(source_wavelengths)
    bad = source_wavelengths[::-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        srf.simulate(np.full((1, bad.size), 0.3), bad, response)


def test_constant_invariance_on_non_uniform_grid():
    """Trapezoidal weighting must preserve constant->constant on uneven grids."""
    grid = np.concatenate([np.linspace(400.0, 1000.0, 20), np.linspace(1005.0, 2400.0, 80)])
    response = srf.gaussian_srf(["G"], np.array([1500.0]), np.array([200.0]), grid)
    out = srf.simulate(np.full((3, grid.size), 0.42), grid, response)
    assert np.allclose(out, 0.42, atol=1e-9)


def test_coverage_uses_full_srf_support():
    """A band only partly spanned by the source grid is gated by coverage."""
    srf_grid = np.linspace(400.0, 1400.0, 400)
    response = srf.gaussian_srf(["B"], np.array([980.0]), np.array([80.0]), srf_grid)
    source = np.linspace(400.0, 1000.0, 120)  # cuts off the upper half of the band
    spectra = np.full((1, source.size), 0.3)
    strict = srf.simulate(spectra, source, response, min_coverage=0.9)
    loose = srf.simulate(spectra, source, response, min_coverage=0.05)
    assert np.isnan(strict[0, 0])
    assert np.isfinite(loose[0, 0])
    assert np.allclose(loose, 0.3, atol=1e-9)  # constant over the covered part


def test_simulate_rejects_bad_min_coverage(source_wavelengths):
    response = _toy_srf(source_wavelengths)
    spectra = np.full((1, source_wavelengths.size), 0.3)
    with pytest.raises(ValueError, match="min_coverage"):
        srf.simulate(spectra, source_wavelengths, response, min_coverage=1.5)


def test_spectral_response_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        srf.SpectralResponse(["a"], np.array([400.0, 500.0, 600.0]), np.array([[0.1, -0.2, 0.3]]))


def test_spectral_response_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        srf.SpectralResponse(
            ["a"], np.array([400.0, 500.0, 600.0]), np.array([[0.1, np.inf, 0.3]])
        )


def test_spectral_response_rejects_zero_total_band():
    with pytest.raises(ValueError, match="zero total response"):
        srf.SpectralResponse(["a"], np.array([400.0, 500.0, 600.0]), np.zeros((1, 3)))


def test_gaussian_rejects_nonpositive_fwhm(source_wavelengths):
    with pytest.raises(ValueError, match="fwhm"):
        srf.gaussian_srf(["a"], np.array([600.0]), np.array([0.0]), source_wavelengths)


def test_trapezoid_weights_sum_to_interval():
    x = np.array([0.0, 0.5, 1.0])
    w = srf._trapezoid_weights(x)
    # Proper trapezoid weights: [0.25, 0.5, 0.25], summing to the interval (1.0),
    # unlike np.gradient which would give [0.5, 0.5, 0.5] (sum 1.5).
    np.testing.assert_allclose(w, [0.25, 0.5, 0.25])
    assert np.isclose(w.sum(), x[-1] - x[0])


def test_coverage_correct_for_partial_span_triangle():
    """Half-spanned triangular SRF must give coverage ~0.5, not 1.0.

    Regression for the np.gradient endpoint-weighting bug: with the source grid
    covering only the upper half of a triangular band, coverage must reflect the
    missing half.
    """
    # Triangular band on a fine native grid, peak at 1.0.
    native = np.linspace(0.0, 2.0, 2001)
    resp = np.clip(1.0 - np.abs(native - 1.0), 0.0, None)[np.newaxis, :]
    response = srf.SpectralResponse(["T"], native, resp)
    source = np.linspace(1.0, 2.0, 501)  # upper half only
    spectra = np.full((1, source.size), 0.3)
    # coverage ~0.5 -> masked at 0.6, kept at 0.4
    assert np.isnan(srf.simulate(spectra, source, response, min_coverage=0.6)[0, 0])
    assert np.isfinite(srf.simulate(spectra, source, response, min_coverage=0.4)[0, 0])


def test_inf_channel_does_not_contaminate(source_wavelengths):
    """A +inf or -inf source channel is dropped like NaN, not propagated."""
    response = _toy_srf(source_wavelengths)
    for bad in (np.inf, -np.inf):
        spectra = np.full((1, source_wavelengths.size), 0.3)
        spectra[0, 5] = bad
        out = srf.simulate(spectra, source_wavelengths, response)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.3, atol=1e-6)


def test_spectral_response_rejects_single_point():
    with pytest.raises(ValueError, match="at least 2 points"):
        srf.SpectralResponse(["a"], np.array([500.0]), np.array([[1.0]]))
