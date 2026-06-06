# Bundled Sentinel-2 spectral response functions

`S2A_SRF.csv`, `S2B_SRF.csv` — per-band relative spectral response for
Sentinel-2A and Sentinel-2B MSI.

- **Source:** ESA Sentinel-2 Spectral Response Functions (S2-SRF),
  document `COPE-GSEG-EOPG-TN-15-0007`, obtained from the USGS-hosted mirror
  `Sentinel-2A MSI Spectral Responses.xlsx`
  (https://landsat.usgs.gov/landsat/spectral_viewer/bands/, which redistributes
  the ESA spreadsheet, sheets "Spectral Responses (S2A)" / "(S2B)").
- **Downloaded:** 2026-06-06.
- **Transform applied:** the sheet's `SR_WL` column was renamed to
  `wavelength_nm` and the `S2x_SR_AV_B*` columns to `B1…B12` (incl. `B8A`).
  No values were altered; blanks outside a band's support are the ESA table's
  explicit zeros (none required imputation).
- **Grid:** 300–2600 nm at 1 nm spacing (2301 rows), responses in [0, 1].

Load with `tanagerspec.srf.load_s2_srf("S2A")` (or `"S2B"`).

To update to a newer ESA release (e.g. v4.0, which adds S2C), re-export the
same columns and bump this note.
