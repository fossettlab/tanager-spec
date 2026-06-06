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

# --- Planet Tanager open data ----------------------------------------------
# The Tanager open data is a STATIC STAC catalog (not a STAC API): it is
# traversed by walking child collections, not via a /search endpoint. The
# child collections are the scene categories (agriculture, urban, fire, ...).
# Public, CC BY 4.0. Verified against the live catalog 2026-06-06.
TANAGER_STAC_URL: str = "https://www.planet.com/data/stac/tanager-core-imagery/catalog.json"

# Child-collection ids in the catalog, i.e. the scene categories.
TANAGER_CATEGORIES: tuple[str, ...] = (
    "agriculture",
    "urban",
    "fire",
    "snow-ice",
    "natural-lands",
    "coastal-water-bodies",
    "energy-mining",
    "GHG-plumes",
    "ROCX2025",
)

# Asset keys on each item. Surface reflectance and radiance are each offered as
# "basic" (georeferenced, unprojected) and "ortho" (orthorectified, projected)
# HDF5 (HDF-EOS5) products. The orthorectified surface-reflectance product is
# the default for spatial analysis (projected; ready for georeferenced output).
TANAGER_SR_ASSET: str = "ortho_sr_hdf5"
TANAGER_RADIANCE_ASSET: str = "ortho_radiance_hdf5"

# HDF-EOS grid + reflectance field inside the SR HDF5, and its fill value.
TANAGER_HDF5_GRID: str = "HYP"
TANAGER_SR_FIELD: str = "surface_reflectance"
TANAGER_SR_FILL_VALUE: float = -9999.0

# EMIT L2A reflectance via the NASA LP DAAC STAC (a real STAC API; used for the
# cross-sensor tie-breaker comparison). TODO: confirm collection id/version.
EMIT_STAC_URL: str = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
EMIT_COLLECTION: str = "EMITL2ARFL_001"
