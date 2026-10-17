# Original Figure 4d zoom restoration

Verified 2026-09-04. This change restores the original spatial windows, zoom
rectangles, linked magnifications, and horizontal per-coordinate colorbars.
It does not change the released production-v2 checkpoint or prediction tables.

The exact bounds and physical pixel size occur both in
`Figure4/imap_if/config_all_cells_production_v2_if_sft.yaml` and the frozen
`Figure4/imap_if/outputs/all_cells_production_v2_if_sft/imap_if_run_metadata.json`.
The renderer hash-checks the configuration before loading any bounds and
independently checks the pixel-to-micrometre conversion in both prediction
tables.

| Condition | Level-0 x | Level-0 y | All cells | Zoom cells |
| --- | --- | --- | ---: | ---: |
| DMSO | 45000–51000 | 35000–39500 | 163,992 | 3,776 |
| RARi | 17000–23000 | 48000–52500 | 136,437 | 3,982 |

Preview QA used `/tmp/figure4d_locked_zooms_20260904/`. The combined PDF was
rendered back to PNG at 300 dpi and visually inspected, along with an individual
whole-tissue/zoom component. The boxes select the intended original windows;
the two maps for each condition show the same cells and geometry, without
reference/prediction reversal. No whole-map display downsampling is performed.
The PDF has one page, is unencrypted, and embeds Arial and Arial Bold TrueType
fonts. `pdfimages -list` confirms 300-ppi dense scatter rasters. Labels and
colorbars are not clipped. A second command without `--overwrite` correctly
refused all pre-existing declared outputs. `py_compile` passed.

This record concerns the regenerated component preview, not the manually
assembled `Figures/Figure4.pdf` or `Figures/Figure4.png`, which were not edited.
