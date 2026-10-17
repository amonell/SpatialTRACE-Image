# RARi options 11–30

The author requested removal of these home-drive PDFs on 16 October 2026.
The folder was moved to Trash; technical records and native crops remain.
For the subsequent review, see [options 31–50](RARI_OPTIONS_31_50.md).

Twenty additional review fields from the original RARi specimen. None repeats
the exact bounds of an earlier option. Nearby fields can overlap. All use
6000 × 4500 native pixels and the approved larger dots: 2.5× the earlier
diameter on contact sheets and 2× on detail pages. Predictions, colors, gates,
channel display limits are unchanged. This review does not publish figures.

The user-facing PDFs are copied to the home drive after validation:
`/home/amonell/Desktop/RARi_field_options_11-30/`.
That folder contains `Figure4D_RARi_20_options.pdf` and `Option_11.pdf` through
`Option_30.pdf`. The combined PDF has four contact sheets, a locator and
twenty detailed pages. Each option shows DAPI/Cy5 E-cadherin, both coordinate
predictions and the existing cell-gate assignments. P14 cells have larger
filled dots without outlines.

Code and the explicit bounds are in the home project. Prepared native crops
and technical render outputs remain under SATA:

- `Figure_4/stored_outputs/panel_D/rari_options_11_30/native_v1/`
- `Figure_Run_Archive/figure4D_RARi_options_11_30_v2/`

The first render is preserved in `figure4D_RARi_options_11_30/`. Its final
check stopped because a separate CCR9-panel publication changed Extended
Figure 4 during rendering. The second render starts from that updated
publication and does not modify it.

The prepared native-field manifest SHA256 is
`33433deed678ad68b7a70472bd722b1dd6e39bc41ecdd84e5103a4c8e02bf606`.

The existing source manifest remains
`Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json`,
SHA256 `f3a67a9481d9969126e2df03d2e7e598158bcd286e92295d717779c7c8817d5b`.
It pins the checkpoint, predictions, native microscopy and confirmed Cy5
channel. Required source fields are the native centroid x/y coordinates,
`predicted_axis_coordinate`, `predicted_epithelial_distance_clipped_1p0`,
`coordinate_gate`, and `class`. Displayed-cell tables preserve all source
columns. The audit compares every value against the frozen source table.

Run from the home project root. Existing destinations are never overwritten.

```bash
uv run --no-project --with bioformats-jar --with numpy \
  --python /home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python \
  python -B Figure_4/panel_D/processing/prepare_panel_D_field_candidates.py \
  --source-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json \
  --windows Figure_4/panel_D/processing/rari_options_11_30_plan.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_11_30/native_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/figure_script/make_panel_D_rari_options.py \
  --candidate-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_11_30/native_v1/manifest.json \
  --expected-count 20 --first-option 11 \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_11_30_v2

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/processing/audit_panel_D_rari_options.py \
  --candidate-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/rari_options_11_30/native_v1/manifest.json \
  --expected-count 20 --first-option 11 \
  --exclude-plan Figure_4/panel_D/processing/rari_options_20261016_plan.json \
  --render-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_11_30_v2

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/panel_D/processing/export_panel_D_review_pdfs.py \
  --review-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/figure4D_RARi_options_11_30_v2 \
  --destination /home/amonell/Desktop/RARi_field_options_11-30
```

These are qualitative choices, not an independent test of anatomical accuracy.
No model fitting, inference or gate adjustment occurs. Final figures are not
changed until the author chooses a field.
