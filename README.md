# tanager-spec

Shared data layer for Tanager hyperspectral-versus-multispectral research,
including the public mineral-mapping workflow
[`tanager-rocks`](https://github.com/fossettlab/tanager-rocks).

All four answer variants of *"what does full Tanager VSWIR give you that a
simulated Sentinel-2 does not?"* and so repeat the same upstream steps. This
package holds those steps once; the analysis (clustering, anomaly detectors,
classifiers, dimensionality estimators, metrics, plotting) stays in each
project repo.

## What's in scope

| Module | Responsibility |
|---|---|
| `stac` | Traverse the Tanager **static** STAC catalog (category = child collection); query EMIT (STAC API); build a scene inventory GeoDataFrame |
| `io` | Load the Tanager SR cube from HDF-EOS5 (`load_tanager_sr_hdf5`); load generic COG/EMIT cubes (`load_reflectance_cube`); export compressed GeoTIFFs |
| `bands` | Validate the wavelength axis; map wavelength windows to band indices |
| `mask` | Per-pixel invalid masks; set atmospheric absorption bands to NaN |
| `srf` | Simulate Sentinel-2 bands by SRF convolution; bundled ESA S2A/S2B SRFs (`load_s2_srf`) |
| `sample` | Reproducible, optionally spatial-block-stratified pixel sampling |
| `config` | Shared constants: `SEED`, absorption windows, Tanager catalog + asset keys, EMIT endpoint |

## The Tanager open data

The open data is a **static** STAC catalog at
`config.TANAGER_STAC_URL`, whose child collections are the scene categories
(agriculture, urban, fire, …). `stac.query_tanager_scenes()` walks it (there is
no search API) and tags each item with its category. Each item offers surface
reflectance and radiance as `basic_*`/`ortho_*` **HDF-EOS5 HDF5** assets;
the default is `ortho_sr_hdf5` (orthorectified surface reflectance, EPSG UTM,
30 m, fill −9999). `io.load_tanager_sr_hdf5()` reads that cube and takes the
426 band-center wavelengths from the file's own dataset attributes (authoritative,
not inferred). Download the `.h5` from the asset href first; it is not read over
HTTP.

## Install

```bash
uv add "tanager-spec==0.1.0"
```

For an editable development checkout, clone this repository beside the
consuming project and run:

```bash
uv add --editable ../tanager-spec
```

Contributor setup:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

The package currently bundles ESA Sentinel-2A/Sentinel-2B spectral-response
tables. Public artifact publication remains blocked until their redistribution
terms are explicitly resolved; source provenance is recorded in
`src/tanager_spec/data/SOURCE.md`.

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

## Reference values — resolved vs. still to confirm

This package ships **no fabricated reference values**. Status:

- **Tanager STAC + product** — resolved (verified against the live catalog
  2026-06-06): `config.TANAGER_STAC_URL` points at the static catalog; the SR
  product is `ortho_sr_hdf5`. EMIT uses the public LP DAAC STAC API
  (`config.EMIT_*`); the collection version string is still worth confirming.
- **Sentinel-2 SRFs** — resolved: the real ESA S2A/S2B response functions are
  bundled as package data (`tanager_spec/data/`, see its `SOURCE.md`) and loaded
  via `load_s2_srf("S2A")`. `gaussian_srf()` remains an *approximation* for
  tests/sanity checks only and **must not** be used for final results;
  `load_srf_csv()` stays available for a user-supplied table.
- **Tanager wavelengths** — resolved: `load_tanager_sr_hdf5()` reads the 426
  band-center wavelengths from the HDF5 dataset's own `wavelengths` attribute
  (authoritative). For non-Tanager rasters, `load_reflectance_cube()` never
  infers wavelengths from metadata — pass `wavelengths=` explicitly.
- **Absorption windows** — still to confirm: `config.ABSORPTION_MASKS_NM`
  follows the workspace convention (O2 A-band + two H2O bands); verify the edges
  against the actual Tanager grid before final results.

## Conventions

- Python ≥ 3.11, `uv` for environments (`uv.lock` committed), `ruff` for
  lint/format.
- Reproducibility: the default sampling seed is `config.SEED` (42); pass
  `seed=` explicitly to override per call.
- Invalid-pixel masks use the convention **`True` = invalid (exclude)**.
- Band order is strictly increasing in wavelength everywhere.
- `logging`, not `print`. NumPy-style docstrings.
