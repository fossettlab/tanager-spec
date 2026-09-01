"""Sentinel-2 band simulation by spectral-response-function (SRF) convolution.

This is the shared, scientifically load-bearing operation for all four
methodology projects: it produces "what Sentinel-2 would have measured" from a
full Tanager spectrum so the hyperspectral-vs-multispectral comparison is fair.

The simulated value for target band ``b`` is the response-weighted mean of the
source reflectance over the source wavelength grid, using trapezoidal
quadrature weights ``dλ_i`` (proper node widths, half-width at the endpoints):

    R_b = sum_i SRF_b(w_i) * dλ_i * r(w_i) / sum_i SRF_b(w_i) * dλ_i

where ``SRF_b`` is the target band's response interpolated onto the source
wavelengths ``w`` and zero outside its defined support. On a uniform grid the
``dλ_i`` cancel and this reduces to ``sum(SRF*r) / sum(SRF)``. This is a
convolution with the published response, **not** a simple average of nearby
bands.

Production use requires real, published SRF tables (e.g. ESA's Sentinel-2
spectral response functions). ``gaussian_srf`` builds an *approximate* response
from explicitly supplied band centers/widths and is intended for testing and
sanity checks only — never for final results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .bands import assert_strictly_increasing
from .config import DEFAULT_MIN_SRF_COVERAGE

logger = logging.getLogger(__name__)


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal integration node weights for a 1-D grid.

    For nodes ``x`` the weight of node ``i`` is half the span of its two
    neighbouring intervals (half-width at the endpoints), so the weights sum to
    ``x[-1] - x[0]`` — unlike ``np.gradient``, which double-counts the
    endpoints. Requires at least two nodes.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("a wavelength grid needs at least 2 points for quadrature")
    w = np.empty_like(x)
    w[1:-1] = (x[2:] - x[:-2]) / 2.0
    w[0] = (x[1] - x[0]) / 2.0
    w[-1] = (x[-1] - x[-2]) / 2.0
    return w


@dataclass
class SpectralResponse:
    """A set of band spectral-response functions on a common wavelength grid.

    Attributes
    ----------
    band_names : list of str
        Names of the target bands, in output order.
    wavelength_nm : np.ndarray
        1-D wavelength grid (nm) on which ``response`` is sampled.
    response : np.ndarray
        Response values, shape ``(n_bands, n_grid)``. Need not be normalized;
        ``simulate`` normalizes per band.
    platform : str or None
        Optional platform label (e.g. ``"S2A"``).
    source : str or None
        Optional provenance string (URL, document id, download date). For
        published SRFs this must be filled in and recorded in the consuming
        project's METHODS.md.
    """

    band_names: list[str]
    wavelength_nm: np.ndarray
    response: np.ndarray
    platform: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        self.wavelength_nm = np.asarray(self.wavelength_nm, dtype=float)
        self.response = np.asarray(self.response, dtype=float)
        if self.response.shape != (len(self.band_names), self.wavelength_nm.size):
            raise ValueError(
                "response shape "
                f"{self.response.shape} does not match "
                f"(n_bands={len(self.band_names)}, n_grid={self.wavelength_nm.size})"
            )
        if self.wavelength_nm.size < 2:
            raise ValueError("wavelength_nm needs at least 2 points")
        # interp_to uses wavelength_nm as the np.interp x-grid, which must be
        # strictly increasing or interpolation is silently wrong.
        assert_strictly_increasing(self.wavelength_nm)
        # Non-finite or negative responses corrupt the weighted mean and
        # coverage; reject them rather than propagate NaN/garbage downstream.
        if not np.all(np.isfinite(self.wavelength_nm)):
            raise ValueError("wavelength_nm contains non-finite values")
        if not np.all(np.isfinite(self.response)):
            raise ValueError("response contains non-finite values")
        if np.any(self.response < 0):
            raise ValueError("response contains negative values")
        if np.any(self.response.sum(axis=1) <= 0):
            dead = [self.band_names[i] for i in np.where(self.response.sum(axis=1) <= 0)[0]]
            raise ValueError(f"bands with zero total response on the grid: {dead}")

    def interp_to(self, target_wavelengths: np.ndarray) -> np.ndarray:
        """Interpolate each band's response onto a target wavelength grid.

        Parameters
        ----------
        target_wavelengths : np.ndarray
            1-D wavelengths (nm), typically the Tanager band centers.

        Returns
        -------
        np.ndarray
            Response on the target grid, shape ``(n_bands, n_target)``. Zero
            outside each band's defined support.
        """
        target = np.asarray(target_wavelengths, dtype=float)
        out = np.empty((len(self.band_names), target.size), dtype=float)
        for i in range(len(self.band_names)):
            out[i] = np.interp(target, self.wavelength_nm, self.response[i], left=0.0, right=0.0)
        return out


def gaussian_srf(
    band_names: list[str],
    centers_nm: np.ndarray,
    fwhm_nm: np.ndarray,
    grid_nm: np.ndarray,
    platform: str | None = None,
) -> SpectralResponse:
    """Build an approximate Gaussian SRF set from explicit band parameters.

    .. warning::
       This is an *approximation* for testing and sanity checks only. Final
       Sentinel-2 simulation must use real published response functions
       (see :func:`load_srf_csv`). The caller supplies all numbers; this
       function invents none.

    Parameters
    ----------
    band_names : list of str
        Target band names.
    centers_nm : np.ndarray
        Band-center wavelengths (nm), shape ``(n_bands,)``.
    fwhm_nm : np.ndarray
        Full-width-at-half-maximum per band (nm), shape ``(n_bands,)``.
    grid_nm : np.ndarray
        Wavelength grid (nm) on which to evaluate the responses.
    platform : str or None
        Optional platform label.

    Returns
    -------
    SpectralResponse
    """
    centers = np.asarray(centers_nm, dtype=float)
    fwhm = np.asarray(fwhm_nm, dtype=float)
    grid = np.asarray(grid_nm, dtype=float)
    if not (len(band_names) == centers.size == fwhm.size):
        raise ValueError("band_names, centers_nm, and fwhm_nm must have equal length")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers_nm contains non-finite values")
    if not np.all(np.isfinite(grid)):
        raise ValueError("grid_nm contains non-finite values")
    if not np.all(np.isfinite(fwhm)) or np.any(fwhm <= 0):
        raise ValueError("fwhm_nm must be finite and strictly positive")
    # FWHM -> Gaussian sigma: FWHM = 2*sqrt(2*ln2)*sigma
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    response = np.empty((centers.size, grid.size), dtype=float)
    for i in range(centers.size):
        response[i] = np.exp(-0.5 * ((grid - centers[i]) / sigma[i]) ** 2)
    return SpectralResponse(
        band_names=list(band_names),
        wavelength_nm=grid,
        response=response,
        platform=platform,
        source="gaussian approximation (NOT a published SRF)",
    )


def load_srf_csv(
    path: Path,
    wavelength_column: str = "wavelength_nm",
    band_columns: list[str] | None = None,
    platform: str | None = None,
    source: str | None = None,
) -> SpectralResponse:
    """Load published spectral response functions from a CSV file.

    The expected layout is one row per wavelength, one column for wavelength
    and one column per band of response values:

    ===============  ====  ====  ===
    wavelength_nm    B01   B02   ...
    ===============  ====  ====  ===
    412.0            0.01  0.00  ...
    ...              ...   ...   ...
    ===============  ====  ====  ===

    Parameters
    ----------
    path : Path
        CSV path. The real ESA Sentinel-2 SRF table should be reshaped into
        this layout by the consuming project and committed there with a
        provenance note.
    wavelength_column : str
        Name of the wavelength column.
    band_columns : list of str, optional
        Band columns to load, in order. If ``None``, all non-wavelength
        columns are used in file order.
    platform : str, optional
        Platform label to attach (e.g. ``"S2A"``).
    source : str, optional
        Provenance string (URL, document id, download date).

    Returns
    -------
    SpectralResponse
    """
    df = pd.read_csv(path)
    if wavelength_column not in df.columns:
        raise ValueError(
            f"wavelength column {wavelength_column!r} not found in {path} "
            f"(columns: {list(df.columns)})"
        )
    if band_columns is None:
        band_columns = [c for c in df.columns if c != wavelength_column]
    missing = [c for c in band_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"band_columns {missing!r} not found in {path} "
            f"(available: {[c for c in df.columns if c != wavelength_column]})"
        )
    wavelength = df[wavelength_column].to_numpy(dtype=float)
    response = df[band_columns].to_numpy(dtype=float).T  # (n_bands, n_grid)
    logger.info("loaded %d SRF bands from %s", len(band_columns), path)
    return SpectralResponse(
        band_names=list(band_columns),
        wavelength_nm=wavelength,
        response=response,
        platform=platform,
        source=source,
    )


def load_s2_srf(platform: str = "S2A") -> SpectralResponse:
    """Load the bundled ESA Sentinel-2 spectral response functions.

    Convenience wrapper over :func:`load_srf_csv` for the SRF tables shipped as
    package data (see ``tanager_spec/data/SOURCE.md`` for provenance).

    Parameters
    ----------
    platform : str
        ``"S2A"`` or ``"S2B"``.

    Returns
    -------
    SpectralResponse
    """
    from importlib.resources import files

    platform = platform.upper()
    if platform not in ("S2A", "S2B"):
        raise ValueError(f"platform must be 'S2A' or 'S2B', got {platform!r}")
    path = files("tanager_spec.data") / f"{platform}_SRF.csv"
    with path.open("rb") as fh:  # works whether installed as files or in a zip
        df = pd.read_csv(fh)
    band_columns = [c for c in df.columns if c != "wavelength_nm"]
    return SpectralResponse(
        band_names=band_columns,
        wavelength_nm=df["wavelength_nm"].to_numpy(dtype=float),
        response=df[band_columns].to_numpy(dtype=float).T,
        platform=platform,
        source="ESA S2-SRF COPE-GSEG-EOPG-TN-15-0007 (USGS mirror); see data/SOURCE.md",
    )


def simulate(
    spectra: np.ndarray,
    source_wavelengths: np.ndarray,
    srf: SpectralResponse,
    min_coverage: float = DEFAULT_MIN_SRF_COVERAGE,
) -> np.ndarray:
    """Simulate target bands from source spectra by SRF convolution.

    Parameters
    ----------
    spectra : np.ndarray
        Source reflectance, shape ``(n_pixels, n_source_bands)``. Non-finite
        channels (``NaN`` or ``±inf``, e.g. masked absorption bands) are
        excluded from each band's weighted mean and reduce its coverage.
    source_wavelengths : np.ndarray
        Source band centers (nm), shape ``(n_source_bands,)``.
    srf : SpectralResponse
        Target-band spectral responses.
    min_coverage : float
        In ``[0, 1]``. If a target band's *coverage* falls below this, the
        output for that band and pixel is set to ``NaN``. Coverage is the ratio
        of the valid in-grid response weight to the band's full published SRF
        integral; it is ~1 when the band is fully spanned and may marginally
        exceed 1 when the source grid is coarser than the SRF's native grid
        (quadrature approximation). The gate catches both NaN gaps within the
        band and a source grid that spans only part of the band's support.

    Returns
    -------
    np.ndarray
        Simulated reflectance, shape ``(n_pixels, n_target_bands)``, band order
        matching ``srf.band_names``.

    Notes
    -----
    The weighted mean uses trapezoidal quadrature weights (proper node widths
    via :func:`_trapezoid_weights`, half-width at the endpoints), so it is
    correct on non-uniform source grids and reduces to ``sum(SRF*r)/sum(SRF)``
    when the grid is uniform. A constant reflectance spectrum maps to that same
    constant in every target band — the basic correctness check.
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
    spectra = np.atleast_2d(np.asarray(spectra, dtype=float))
    source_wavelengths = np.asarray(source_wavelengths, dtype=float)
    if spectra.shape[1] != source_wavelengths.size:
        raise ValueError(
            f"spectra has {spectra.shape[1]} bands but source_wavelengths has "
            f"{source_wavelengths.size}"
        )
    # Contract check: column i of spectra is the reflectance at
    # source_wavelengths[i]; the workspace requires this axis strictly
    # increasing, and a violation usually signals a band/wavelength misalignment.
    assert_strictly_increasing(source_wavelengths)

    # Trapezoidal quadrature weights: response x proper node width, so the
    # weighted mean approximates the SRF-weighted wavelength integral even when
    # the source grid is non-uniform.
    src_widths = _trapezoid_weights(source_wavelengths)  # (n_source,)
    weights = srf.interp_to(source_wavelengths) * src_widths[np.newaxis, :]  # (n_target, n_source)

    # Full published response integral on the SRF's own native grid; used as the
    # coverage denominator so a band only partly spanned by the source grid is
    # flagged even when its in-grid weight is internally complete.
    srf_widths = _trapezoid_weights(srf.wavelength_nm)  # (n_grid,)
    srf_full = (srf.response * srf_widths[np.newaxis, :]).sum(axis=1)  # (n_target,)

    total_weight = weights.sum(axis=1)  # (n_target,)
    if np.any(~(total_weight > 0)):  # catches <= 0 and NaN total weight
        empty = [srf.band_names[i] for i in np.where(~(total_weight > 0))[0]]
        logger.warning("SRF bands with no usable overlap on the source wavelength grid: %s", empty)

    # Treat only finite channels as valid so a stray +/-inf does not contaminate
    # other target bands through the matrix product.
    valid = np.isfinite(spectra)  # (n_pixels, n_source)
    filled = np.where(valid, spectra, 0.0)

    # Weighted sums over valid channels, vectorized over pixels and bands.
    num = filled @ weights.T  # (n_pixels, n_target)
    valid_weight = valid.astype(float) @ weights.T  # (n_pixels, n_target)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / valid_weight
        coverage = valid_weight / srf_full[np.newaxis, :]

    out[(coverage < min_coverage) | ~np.isfinite(out)] = np.nan
    return out
