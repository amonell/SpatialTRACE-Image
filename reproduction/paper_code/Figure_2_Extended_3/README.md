# Graph component and annotation-budget comparisons

The completed component comparison (A) is also integrated as Extended Figure
2E through its thin rendering wrapper. This package retains the common plot
implementation. Its completed annotation-budget comparison (B) is integrated
as Extended Figure 2F through a hash-locked wrapper; the spatial illustration
(C) remains a separate component. No historical artwork is replaced. Their training completion and release
status are recorded by the frozen pipeline, independently of figure rendering.

| Panel | Generated component | Source |
| --- | --- | --- |
| A | Four-model paired section comparison; Pearson r and MAE, both coordinates | 48 new fits plus the original 16 full-graph fits |
| B | Full/cell-only/uniform matched annotation-budget curves | 320 new fits plus the original 160 full-graph fits |
| C | One held-out annotated field: manual reference, full graph, validation-selected simpler model, both absolute-error maps | Locked five-villus budget and geometry-selected field |

All three panels are generated from numeric tables, not manually assembled.
Fitting, graph randomization, checkpoint selection, metric creation and field
preparation belong to `frozen_preprocessing/model_design_comparisons_v1/`.
The locked graph plan is on SATA at
`frozen_preprocessing/model_design_comparisons_v1/graph/plan.json`. It resolves
and hashes the original eight section-disjoint splits, 30-dimensional scVI
features, annotations, exact-degree-preserving randomization and full-model
reference outputs. No per-image or per-model annotation selection occurs.

Required fields: variant, axis, fold_id, test_section, n_train_villi, n,
pearson_r and mae. Spatial tables also contain x, y and paired reference,
prediction and absolute-error columns for each coordinate. Frozen release
manifests hash every table. Graph figure scripts reject incomplete comparisons.

Aggregation unit: held-out section (eight). Budget ribbons are section mean ±
SEM. Joint learning of the upstream scVI representation means these are
section-disjoint supervised evaluations, not fully inductive feature-learning
tests. The sparse spatial example has manual coordinate references only at
annotated cells. Its baseline is chosen by validation MSE, not test errors.

After preprocessing completes, regenerate:

```bash
Figure2/.venv/bin/python frozen_preprocessing/model_design_comparisons_v1/aggregate_graph.py
Figure2/.venv/bin/python Figure_2_Extended_3/run_all_panels.py \
  --release /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/model_design_comparisons_v1/graph/release_all/manifest.json \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2_Extended_3
```

Output directory: SATA `Figure_2_Extended_3/panel_A/`, `panel_B/`, `panel_C/`.
Expected stems: `panel_A_graph_component_comparison`,
`panel_B_matched_annotation_budgets`, `panel_C_heldout_spatial_errors`.
Each has PDF, 300-dpi PNG and provenance JSON. All text is Arial and dense
layers, if any, are separated from vector text/axes. `--overwrite` archives
only the script's own previous exports. `--dry-run` prints exact commands.
