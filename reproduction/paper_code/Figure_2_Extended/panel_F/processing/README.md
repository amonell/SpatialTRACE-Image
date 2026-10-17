# Frozen matched annotation-budget comparison

Panel F follows the graph component comparison (E). It compares the full graph,
cell-only, and uniform-neighbor models at ten matched training-villus counts,
using the unchanged eight section-disjoint folds and validation/test sections.
The 320 new baseline fits and 160 original full-graph fits are complete.

Authoritative release:
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/model_design_comparisons_v1/graph/release_all/manifest.json`
SHA256: `ffaa667a34ef1046f64925f03bc629bfd0b12f79b09f50080e904680305206e5`.
Its `budget_metrics.tsv` SHA256 is
`1df2c27b73f7d76f8ce964ba35f1d7bb413dcdd7f9578a878f431f071e821c0d`.
The active wrapper verifies the release hash; the shared renderer verifies the
table hash and all 480 unique model/coordinate/section/budget combinations.

Required columns: `variant`, `axis`, `test_section`, `n_train_villi`,
`pearson_r`, and `mae`. Curves summarize section mean and SEM at 1, 5, 10, 15,
20, 25, 30, 35, 40, and 45 annotated training villi. No cells are reselected,
models fitted, or per-section metric tables created during rendering.
The original analysis is frozen by
`frozen_preprocessing/model_design_comparisons_v1/aggregate_graph.py`.
The panel shares its plotting implementation with the standalone budget figure
in `Figure_2_Extended_3/panel_B/figure_script/`.

Supervised splits are section-disjoint. The upstream 30-dimensional scVI
representation was learned jointly across sections. Validation annotations are
additional to the plotted training budget; the curves describe held-out-section
means, not significance tests or independent training-seed repeats.

```bash
python -B Figure_2_Extended/run_all_panels.py --panels F \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2_Extended
python -B update_final_figure.py Figure_2_Extended --overwrite
```

Outputs under SATA `Figure_2_Extended/panel_F/`:
`panel_F_matched_annotation_budgets.pdf`, its 300-dpi PNG and provenance JSON,
plus the assembled `panel_F_final` PDF/PNG/provenance layer. The final page
arranges all four axes horizontally, grouped by coordinate, and retains the
three-model legend. All labels are Arial; panel letters have no periods.
