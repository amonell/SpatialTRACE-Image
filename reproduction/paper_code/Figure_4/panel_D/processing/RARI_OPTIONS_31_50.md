# RARi options 31–50

Twenty additional fields from the original RARi specimen, using the same
6000 × 4500-pixel size, frozen predictions, gate assignments, channel display,
color scales and larger dots as options 11–30. The explicit bounds and spatial
coverage rule are in `rari_options_31_50_plan.json`. Selection uses cell
positions and counts, not prediction values. Fields can overlap, but none
repeats an earlier window; overlap with any single previous or new field is
at most 37.1% of field area.

Each detail page shows native DAPI/Cy5 E-cadherin, gate-colored cells, and both
coordinate predictions. P14 cells retain larger filled dots. All cells in
each half-open crop are displayed. There is no inference, gate adjustment,
model selection or change to published figures. These are qualitative review
options, not independent anatomical-accuracy measurements.

## Locations

- Home-drive PDF copies: `/home/amonell/Desktop/RARi_field_options_31-50/`.
- Native crops: SATA `Figure_4/stored_outputs/panel_D/rari_options_31_50/native_v1/`.
- Rendered review, tables and provenance: SATA `Figure_Run_Archive/figure4D_RARi_options_31_50/`.

The home folder contains only `Figure4D_RARi_20_options.pdf` and
`Option_31.pdf` through `Option_50.pdf`. The combined PDF contains four
comparison sheets, a location map and twenty detail pages.

The original backprojection manifest is
`Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json`
under SATA, SHA256
`f3a67a9481d9969126e2df03d2e7e598158bcd286e92295d717779c7c8817d5b`.
It pins the microscopy, Cy5 channel identity, checkpoint, predictions and
gate assignments. The prepared manifest hashes each extracted crop. Required
cell columns are `centroid_x_fullres_px`, `centroid_y_fullres_px`,
`predicted_axis_coordinate`, `predicted_epithelial_distance_clipped_1p0`,
`coordinate_gate` and `class`; displayed tables retain every source column.

Prepared manifest SHA256:
`3ba85d7e516b75f75e1c93637a63687cd70ff46b7b2b8f378fae9898e5f884f8`.

The author requested removal of the previous home-drive options 11–30.
That folder, including its combined PDF, was moved to the home-drive Trash
on 16 October 2026. Scientific source data and technical records were retained.

## Regeneration

Run from the home code project. All output directories must be new.

```bash
uv run --no-project --with bioformats-jar --with numpy \
  --python /home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python \
  python -B Figure_4/panel_D/processing/prepare_panel_D_field_candidates.py \
  --source-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json \
  --windows Figure_4/panel_D/processing/rari_options_31_50_plan.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_31_50/native_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/figure_script/make_panel_D_rari_options.py \
  --candidate-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_31_50/native_v1/manifest.json \
  --expected-count 20 --first-option 31 \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_31_50

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/processing/audit_panel_D_rari_options.py \
  --candidate-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_31_50/native_v1/manifest.json \
  --expected-count 20 --first-option 31 \
  --exclude-plan Figure_4/panel_D/processing/field_review_plan_20261016.json \
  --exclude-plan Figure_4/panel_D/processing/rari_options_20261016_plan.json \
  --exclude-plan Figure_4/panel_D/processing/rari_options_11_30_plan.json \
  --render-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_31_50

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/processing/export_panel_D_review_pdfs.py \
  --review-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_31_50 \
  --destination /home/amonell/Desktop/RARi_field_options_31-50
```

The audit checks each displayed value against its frozen source, native
channel dimensions, contiguous numbering, no repeated earlier bounds,
embedded Arial, all four views, 25 combined pages, and unchanged current
figures and field-selection pointer. The export verifies every PDF copy.
