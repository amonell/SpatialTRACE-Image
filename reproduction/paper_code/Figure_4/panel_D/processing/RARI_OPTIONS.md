# Ten new RARi fields — 16 October 2026

The latest review is [options 31–50](RARI_OPTIONS_31_50.md).
The home-drive [options 11–30](RARI_OPTIONS_11_30.md) were moved to Trash at
the author's request. The ten options below remain available.

Review only. The original DMSO field and both published Figure 4 PDFs remain
unchanged. None of these ten fields repeats a field in the earlier gallery.
All are from the original RARi specimen and have the same 6000 × 4500-pixel
size. Numbered options show microscopy, saved coordinate predictions and
existing gate calls with the same contrast and color scales.

Prepared native DAPI/Cy5 crops are under SATA
`Figure_4/stored_outputs/panel_D/rari_options_20261016/native_v1/`.
The manifest SHA256 is
`6b2c7190ec0438360bf5fa9028ecc13e52b61a9918dd6c2d691a575a25a7bd43`.
It hashes the original VSI/ETS sources, confirmed Cy5 channel key, prediction
release and checkpoint. Exact bounds are in `rari_options_20261016_plan.json`.
Required cell columns are `centroid_x_fullres_px`, `centroid_y_fullres_px`,
`predicted_axis_coordinate`, `predicted_epithelial_distance_clipped_1p0`,
`coordinate_gate` and `class`. Every source column is retained in the exported
displayed-cell tables; no cell inside a field is omitted.

Review output:
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_20261016_v3_large_dots/`.
The main PDF is `Figure4D_RARi_10_options.pdf`: two contact sheets, one locator
and ten detailed pages. Each detailed page has a separate PDF in
`individual_PDFs/`; PNGs are in `previews/`. `candidate_fields.csv` lists bounds
and counts. Provenance and `data_identity_audit.json` are alongside the PDF.

The current review uses dots 2.5 times the earlier diameter on the contact
sheets and twice the diameter on detailed pages and the locator. P14 retains
its relative marker size. Field bounds, every cell value, colors, microscopy,
fonts and layout are unchanged. The preceding smaller-dot review remains
intact in `figure4D_RARi_options_20261016_v2/`.

Run from the home project root. Use new destinations when rerunning; these
commands refuse to overwrite a review. Preparation is separate from rendering.

```bash
uv run --no-project --with bioformats-jar --with numpy \
  --python /home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python \
  python -B Figure_4/panel_D/processing/prepare_panel_D_field_candidates.py \
  --source-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json \
  --windows Figure_4/panel_D/processing/rari_options_20261016_plan.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_20261016/native_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/figure_script/make_panel_D_rari_options.py \
  --candidate-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_20261016/native_v1/manifest.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_20261016_v3_large_dots

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/processing/audit_panel_D_rari_options.py \
  --candidate-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_20261016/native_v1/manifest.json \
  --render-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_20261016_v3_large_dots
```

These are qualitative alternatives for author review, not independent
anatomical-accuracy estimates. Native microscopy and predictions are visible
together so remaining disagreements can be assessed. No inference, training,
gate adjustment or cohort analysis is performed. The review scripts are not
part of the final-figure runner. A subsequent author choice is required before
updating the two final D panels together.
