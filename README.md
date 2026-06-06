# tanagerspec

Shared data layer for the four Tanager hyperspectral-vs-multispectral
methodology projects in this workspace:
[`tanager-spectralsep`](../tanager-spectralsep),
[`tanager-anomaly`](../tanager-anomaly),
[`tanager-featureimp`](../tanager-featureimp), and
[`tanager-infotheory`](../tanager-infotheory).

All four answer variants of *"what does full Tanager VSWIR give you that a
simulated Sentinel-2 does not?"* and so repeat the same upstream steps. This
package holds those steps once; the analysis (clustering, anomaly detectors,
classifiers, dimensionality estimators, metrics, plotting) stays in each
project repo.

## What's in scope

| Module | Responsibility |
|---|---|
| `stac` | Query Tanager / EMIT scenes; build a scene inventory GeoDataFrame |
| `io` | Load georeferenced reflectance cubes (rioxarray); export compressed GeoTIFFs |
| `bands` | Validate the wavelength axis; map wavelength windows to band indices |
| `mask` | Per-pixel invalid masks; set atmospheric absorption bands to NaN |
| `srf` | Simulate Sentinel-2 bands by spectral-response-function convolution |
| `sample` | Reproducible, optionally spatial-block-stratified pixel sampling |
| `config` | Shared constants: `SEED`, absorption windows, STAC endpoints |

## Install

```bash
uv sync --extra dev      # create .venv and install with dev tools
uv run pytest            # run the test suite
uv run ruff check src tests
```

Consuming projects add it as an editable dependency, e.g.

```bash
uv add --editable ../tanagerspec
```

## The Sentinel-2 simulation

Simulated band `b` for source spectrum `r(λ)` (Tanager) is the
response-weighted mean over the source wavelength grid `w`, using trapezoidal
quadrature weights `Δλ_i` (per-node cell widths) so it is correct on
non-uniform grids:

```
R_b = Σ_i SRF_b(w_i) · Δλ_i · r(w_i)  /  Σ_i SRF_b(w_i) · Δλ_i
```

with `SRF_b` interpolated onto `w` and zero outside its support; on a uniform
grid the `Δλ_i` cancel and this reduces to `Σ SRF·r / Σ SRF`. Non-finite
channels (`NaN` or `±inf`, e.g. masked absorption bands) are dropped from each
band's weighted mean. Coverage
is measured as the valid in-grid weight divided by the band's **full published
SRF integral**, so a band that is only partly spanned by the source grid — or
riddled with NaN gaps — is returned as NaN once coverage falls below
`config.DEFAULT_MIN_SRF_COVERAGE`. A constant reflectance spectrum maps to that
same constant in every band — the correctness check exercised in the tests.

This is convolution with the published response, **not** averaging nearby
bands.

## Values you must supply (not invented here)

This package deliberately ships **no fabricated reference values**. Before the
consuming projects can run end to end, confirm and set:

- **Planet Open STAC endpoint + Tanager collection id** —
  `config.TANAGER_STAC_URL` and `config.TANAGER_COLLECTION` are `None`. The
  query helpers raise until these are set (from Planet's Open Data Competition
  documentation). The EMIT endpoint is the public LP DAAC STAC; confirm its
  collection version.
- **Real Sentinel-2 spectral response functions** — `load_srf_csv()` reads a
  CSV (wavelength column + one column per band) that the consuming project
  commits with a provenance note (source URL, ESA document id, download date).
  `gaussian_srf()` builds an *approximation* from band centers/widths you pass
  explicitly; it is for tests and sanity checks only and **must not** be used
  for final results.
- **Tanager wavelength vector** — read from the product at load time;
  `load_reflectance_cube()` validates it is strictly increasing. The
  band-metadata reader is best-effort (TODO: confirm where the product stores
  wavelengths); otherwise pass `wavelengths=` explicitly.
- **Absorption windows** — `config.ABSORPTION_MASKS_NM` follows the workspace
  convention (O2 A-band + two H2O bands); verify the edges against the actual
  Tanager grid.

## Conventions

- Python ≥ 3.11, `uv` for environments (`uv.lock` committed), `ruff` for
  lint/format.
- Reproducibility: the default sampling seed is `config.SEED` (42); pass
  `seed=` explicitly to override per call.
- Invalid-pixel masks use the convention **`True` = invalid (exclude)**.
- Band order is strictly increasing in wavelength everywhere.
- `logging`, not `print`. NumPy-style docstrings.
