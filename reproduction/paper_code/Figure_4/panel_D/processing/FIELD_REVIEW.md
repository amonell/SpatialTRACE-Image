# Figure 4D field choices — 16 October 2026

This separate review offers five DMSO and five RARi fields. Each shows the same
6000 × 4500 native-pixel window in microscopy, crypt–villus prediction,
epithelial-distance prediction and gate overlay. The original main-panel fields
are included for comparison. No field is promoted into either final figure.

The author requested intact morphology and less conspicuous assignment of
muscularis to a top gate. Candidate selection therefore uses visible predictions
as well as microscopy. These examples are qualitative illustrations, not
unbiased accuracy estimates. There are no independent muscularis labels for
this review. Some residual disagreement can remain. Select one DMSO and one
RARi field before changing the main and extended panels together.

## Sources

The original two specimens, released predictions and gates come from
SATA `Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json`,
SHA256 `f3a67a9481d9969126e2df03d2e7e598158bcd286e92295d717779c7c8817d5b`.
Its nested records lock the coordinate checkpoint, prediction tables, source
VSI/ETS, Cy5 antibody key and original channel arrays. The preparer checks these
hashes. Neither model inference nor gate reassignment is performed.

Required cell columns are `centroid_x_fullres_px`, `centroid_y_fullres_px`,
`predicted_axis_coordinate`, `predicted_epithelial_distance_clipped_1p0`,
`coordinate_gate`, and `class`. Every cell inside each window is plotted.
P14 cells retain larger filled markers without outlines. E-cadherin is Cy5.
Both conditions retain the original shared channel contrast settings.

All main plotting entry points remain unchanged; the candidate renderer is
deliberately excluded from the final-figure runners.

## Regeneration

Run from the home project root. Each command requires a new output directory;
existing review artifacts are preserved. `uv` supplies only the microscopy
reader dependencies in an isolated environment.

```bash
uv run --no-project --with bioformats-jar --with numpy \
  --python /home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python \
  python Figure_4/panel_D/processing/prepare_panel_D_field_candidates.py \
  --source-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/field_review_20261016/survey

uv run --no-project --with bioformats-jar --with numpy \
  --python /home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python \
  python Figure_4/panel_D/processing/prepare_panel_D_field_candidates.py \
  --source-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json \
  --windows Figure_4/panel_D/processing/field_review_plan_20261016_v2.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/field_review_20261016/native_candidates_v2

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/figure_script/make_panel_D_field_review.py \
  --overview-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/field_review_20261016/survey/manifest.json \
  --candidate-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/field_review_20261016/native_candidates_v2/manifest.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_field_candidates_20261016_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/processing/audit_panel_D_field_review.py \
  --candidate-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/field_review_20261016/native_candidates_v2/manifest.json \
  --render-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_field_candidates_20261016_v1
```

The review output contains `Figure4D_field_candidates.pdf`, individual PDFs in
`individual_PDFs/`, 300-dpi field PNGs in `previews/`, `candidate_fields.csv`,
the exact displayed cell tables, and provenance. The separate audit verifies
all cell columns against the frozen predictions, native microscopy where it
overlaps old fields, common bounds, and unchanged hashes of all eight current
figure PDFs. Fonts and color scales match the figure components.
