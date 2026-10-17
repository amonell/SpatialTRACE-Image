# Matched branch retraining

User-requested panel H is included in the approved A–H generated page. Its
separately fitted models complement panel E's after-training crop shuffles.

The production-v2 no-local, no-context and no-fine models were separately
trained for 20 epochs on the same 81,656 training and 5,099 validation cells as
the full model. `frozen_preprocessing/model_design_comparisons_v1/audit_image_branches.py`
checks checkpoint metadata, validation-loss selection, the original training
contract and exact reconstructed retained-encoder initialization. It reevaluates
all four checkpoints on the same 5,107 prepared development cells.

Frozen release and per-input SHA256 records:
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/model_design_comparisons_v1/image_branches_evaluated_v1/release.json`.
Required metrics fields: condition, axis, n, pearson_r, mae. Display all four
conditions for both coordinates. This single matched-seed analysis has no
seed-replicate confidence intervals and is not a new independent test.

Regenerate without training or inference:

```bash
Figure2/.venv/bin/python Figure_3_Extended/panel_F/figure_script/make_panel_F_matched_branch_retraining.py \
  --release /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/model_design_comparisons_v1/image_branches_evaluated_v1/release.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3_Extended/panel_F
```

Outputs: panel_F_matched_branch_retraining.pdf, .png and panel_F_provenance.json.
`--overwrite` archives this script's existing exports before replacement.
