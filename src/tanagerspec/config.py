"""Shared configuration constants for tanagerspec.

Values that have a scientific or operational meaning are named here so the
four consuming projects (tanager-spectralsep, tanager-anomaly,
tanager-featureimp, tanager-infotheory) share one source of truth.

Constants marked ``TODO`` are NOT yet confirmed against real products or
documentation. They must be set before use; functions that depend on an
unset value raise rather than guessing. Do not replace a ``None`` placeholder
with a plausible-looking value without a source.
"""

from __future__ import annotations

# --- Reproducibility -------------------------------------------------------
# Single named seed passed through all stochastic operations.
SEED: int = 42

# --- Atmospheric absorption band masking -----------------------------------
# Wavelength windows (nm) dropped before any spectral analysis. These are
# strong atmospheric absorption regions where surface reflectance retrieval is
# unreliable. The window edges follow the convention already used elsewhere in
# this workspace (tanager-rocks); they are physically motivated (O2 A-band,
# two H2O bands) but the exact edges should still be checked against the actual
# Tanager wavelength grid before final results. TODO: verify edges vs. grid.
ABSORPTION_MASKS_NM: list[tuple[float, float]] = [
    (755.0, 770.0),    # O2 A-band
    (1350.0, 1450.0),  # H2O
    (1800.0, 1950.0),  # H2O
]

# --- Sentinel-2 simulation -------------------------------------------------
# Engineering guard for SRF convolution: if, for a given target band, the
# fraction of the band's spectral-response weight that falls on *valid*
# (non-NaN) source channels drops below this, the simulated band is set to NaN
# rather than reported from partial coverage. This is a numerical-stability
# threshold, not a scientific parameter.
DEFAULT_MIN_SRF_COVERAGE: float = 0.5

# --- STAC endpoints --------------------------------------------------------
# Planet Open STAC catalog serving Tanager scenes. NOT confirmed yet — left
# as None on purpose so query functions raise a clear error instead of hitting
# an invented URL. TODO: set from Planet's Open Data Competition documentation.
TANAGER_STAC_URL: str | None = None
TANAGER_COLLECTION: str | None = None

# EMIT L2A reflectance via the NASA LP DAAC STAC (used for the cross-sensor
# tie-breaker comparison). The endpoint is public; the collection version
# string should still be confirmed. TODO: confirm collection id/version.
EMIT_STAC_URL: str = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
EMIT_COLLECTION: str = "EMITL2ARFL_001"
