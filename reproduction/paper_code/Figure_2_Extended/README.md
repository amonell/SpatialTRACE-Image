# Extended Figure 2 reproducible component package

For the **finished panel layout**, run from the home project:

```bash
python -B update_final_figure.py Figure_2_Extended --overwrite
```

This regenerates the page and publishes `panel_X_final.pdf`,
`panel_X_final.png`, and `panel_X_final_provenance.json` to
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2_Extended/panel_X/`.
These PDFs contain only their own panel on the registered full-page canvas;
import at the page origin without rescaling. `final_panel_manifest.json` in
the SATA figure directory identifies each final renderer and its source hashes.
Use `--dry-run` to inspect commands without writes. Previous final exports
are archived; original assembled PDFs remain untouched. See
[the final-figure editing guide](../FINAL_FIGURE_EDITING.md).
The component commands below provide smaller individual assets, not a second
authoritative final-page layout.

This directory regenerates six data panels. The user-approved layout adds
graph-component and matched annotation-budget comparisons to the four original
panels and reflows the tissue grid. The supplied `Figures/Figure2_Extended.pdf` remains
an unchanged source-artwork reference.

| Panel | Generated component | Frozen source |
|---|---|---|
| A | Manual sparse labels and frozen dense graph predictions for Day 6, Day 8, and Day 30 SI; matched zooms in all four columns | Original sparse annotations, all-label graph-coordinate visualization lineage and frozen spatial zoom subsets |
| B | Measured fitting runtime versus training villi | Completed matching 160-fit benchmark; `runtime_benchmark_v1/completed.json` |
| C | Held-out-section MAE versus training villi | Same eight section-disjoint folds as main Figure 2e |
| D | Held-out-section R² versus training villi | Same eight section-disjoint folds, including negative values |
| E | Full graph, cell-only, uniform-neighbor, and randomized-connection models; Pearson r and MAE | Frozen separately fitted comparators with matched section splits |
| F | Full graph, cell-only, and uniform-neighbor models across 1–45 annotated training villi; Pearson r and MAE for both coordinates | Completed 320 baseline fits and 160 original full-graph fits; matched annotations and unchanged validation/test sections |

Panels C/D were regenerated from the completed section-disjoint experiment.
The aggregation unit is a complete held-out section; each count has eight points,
with mean ± SEM. Counts are 1 and multiples of 5 through 45 training villi.
The same six training, one validation and one test sections apply to every count
within a fold. Validation annotations are additional to the x-axis training budget.
Upstream 30-dimensional scVI features were jointly learned across sections.
Panel B uses independently timed fits under that same design, not historical
approximate runtimes. No rendering script runs training. Arial is embedded.

## Locked sources

The metric table is SATA `Figure2/section_disjoint_villus_scaling_v1/per_fold_metrics.tsv`
(SHA256 `462e31be3c8c6cfe0ec7aff09f282bfe5bb685f16f151bd3484f3478e75dcf94`).
The split is SATA `splits/graph_model_section_disjoint_villus_scaling_v1.json`
(SHA256 `e1779c3caab90164eea05ba38d2dc93c3432ad845711fd7428f281bd52eaa0f8`).
Required fields are fold, held-out section, axis, training-villus count, MAE and R²;
the shared `scaling.py` verifies the complete matched 160-row design.
Runtime inputs resolve through SATA
`Figure_2_Extended/stored_outputs/runtime_benchmark_v1/completed.json`, hashing
`runtime_records.tsv` with measured `fit_seconds`, fold, axis, seed and count.
The endpoint includes validation and checkpoint writing inside fitting, but excludes
data preparation and test inference. Benchmark checkpoints cannot replace scientific models.

Outputs under SATA `Figure_2_Extended/panel_X/` are PDF/PNG pairs and provenance:
`panel_B_training_runtime`, `panel_C_heldout_villus_mae`, and
`panel_D_heldout_villus_r2` (legacy filename stems retained for Illustrator links).
Each renderer also exports its plotted source summary. Rendering refuses missing
or mismatched final inputs. B–D have all been regenerated from the matched design.

The completed timing manifest SHA256 is
`ff9c26154f9422e7063a008e32940c074aafdf36631e7ac7aed8599d0db91de8`.

```bash
/home/amonell/Desktop/spatial_axes_manuscript/Figure2/.venv/bin/python \
  /home/amonell/Desktop/spatial_axes_manuscript/Figure_2_Extended/run_all_panels.py \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2_Extended
```

`panel_order.json` records final component filenames and the visible order:
panel A rows are Day 6, Day 8, and Day 30; its columns are crypt-villus
reference, crypt-villus prediction, epithelial-distance reference and prediction.
Panels B–D run left-to-right beneath those maps; E compares graph components
and F follows with matched annotation requirements. E/F each use four axes
grouped by coordinate. F retains a shared three-model legend.
The smaller standalone A components retain their original import ordering.
Panel A includes twelve framed reference/prediction zooms with vector source
boxes, connectors and external 200 µm bars. All four views of each section
use the same prespecified 800 µm field. Whole maps, zooms and horizontal
colorbars have matched sizes across all four columns. Reference zooms retain
gray unannotated cells and color only the original sparse labels. All views
fit within the existing rows; headings, row centers and panels B–F do not move.
Colorbars are horizontal below the last row, with unchanged ranges
(crypt-villus 0–1; epithelial distance 0–0.7). Zoom inputs and their hash locks
are documented in `panel_A/processing/README.md`.
Its four PDF/PNG stems remain `panel_A_crypt_villus_manual_labels`,
`panel_A_crypt_villus_dense_predictions`,
`panel_A_epithelial_distance_manual_labels`, and
`panel_A_epithelial_distance_dense_predictions`; the combined final layer is
`panel_A_final`. All are written beneath SATA `Figure_2_Extended/panel_A/`.
Use `--panels A D` to select
panels, `--dry-run` to print commands, and `--prepared-only` to explicitly
request frozen inputs. This runner has no preprocessing; `--prepare-only`
therefore renders nothing. Pre-change code is recoverable in the home
`lineage_archive/final_figure_organization_2026-09-04/Figure_2_Extended/`.

E reads the hash-locked component metric table resolved by SATA
`frozen_preprocessing/model_design_comparisons_v1/graph/release_components/manifest.json`.
Required fields are `axis`, `variant`, `test_section`, `pearson_r`, and `mae`.
Gray lines pair the eight held-out sections; colored horizontal segments show
section means. The renderer performs no fitting or metric-table creation.
Its standalone stem is `panel_E_graph_component_comparison`; the final-page
layer is `panel_E_final`. See `panel_E/processing/README.md` for the input lock.

F reads the complete frozen release under SATA
`frozen_preprocessing/model_design_comparisons_v1/graph/release_all/manifest.json`
(SHA256 `ffaa667a34ef1046f64925f03bc629bfd0b12f79b09f50080e904680305206e5`).
Its `budget_metrics.tsv` SHA256 is
`1df2c27b73f7d76f8ce964ba35f1d7bb413dcdd7f9578a878f431f071e821c0d`.
Required fields are `variant`, `axis`, `test_section`, `n_train_villi`,
`pearson_r`, and `mae`. All 480 combinations must be present exactly once.
Lines and ribbons show section mean ± SEM. The scVI feature-lineage limitation
and additional validation annotations described above apply to this comparison.
The output stem is `panel_F_matched_annotation_budgets`; its final-page layer
is `panel_F_final`, both under SATA `Figure_2_Extended/panel_F/`.
See `panel_F/processing/README.md` for the standalone command and frozen lineage.
