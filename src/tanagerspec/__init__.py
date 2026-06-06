"""tanagerspec: shared data layer for the Tanager methodology projects.

Provides the primitives common to tanager-spectralsep, tanager-anomaly,
tanager-featureimp, and tanager-infotheory:

- ``stac``    : query Tanager/EMIT scenes and build a scene inventory
- ``io``      : load georeferenced reflectance cubes; export GeoTIFFs
- ``bands``   : wavelength-axis validation and window selection
- ``mask``    : invalid-pixel and atmospheric absorption-band masking
- ``srf``     : Sentinel-2 band simulation by SRF convolution
- ``sample``  : reproducible (optionally stratified) pixel sampling
- ``config``  : shared constants (SEED, absorption windows, endpoints)

Analysis (clustering, anomaly detectors, classifiers, dimensionality
estimators, metrics, plotting) deliberately lives in the individual project
repositories, not here.
"""

from __future__ import annotations

from . import bands, config, io, mask, sample, srf, stac
from .sample import SampleResult, sample_pixels
from .srf import SpectralResponse, gaussian_srf, load_srf_csv, simulate

__version__ = "0.1.0"

__all__ = [
    "bands",
    "config",
    "io",
    "mask",
    "sample",
    "srf",
    "stac",
    "SampleResult",
    "sample_pixels",
    "SpectralResponse",
    "gaussian_srf",
    "load_srf_csv",
    "simulate",
    "__version__",
]
