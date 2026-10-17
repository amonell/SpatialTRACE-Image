# Extended Figure 3 reproducible component package

For the **finished panel layout**, run from the home project:

```bash
python -B update_final_figure.py Figure_3_Extended --overwrite
```

This regenerates the page and publishes `panel_X_final.pdf`,
`panel_X_final.png`, and `panel_X_final_provenance.json` to
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3_Extended/panel_X/`.
These PDFs contain only their own panel on the registered full-page canvas;
import at the page origin without rescaling. `final_panel_manifest.json` in
the SATA figure directory identifies each final renderer and its source hashes.
Use `--dry-run` to inspect commands without writes. Previous final exports
are archived; original assembled PDFs remain untouched. See
[the final-figure editing guide](../FINAL_FIGURE_EDITING.md).
The component commands below provide smaller individual assets, not a second
authoritative final-page layout.

This directory regenerates the data and microscopy components in the active
SATA `Figures/Figure3_Extended.pdf` (reviewed 2026-09-05; SHA256
`c124a3765ac89e71439d705e7af72b422074e0d157153f7e76e66fb09f0a3e6f`).
The original PDF locks the source artwork. The generated page includes the
September 6 branch-retraining addition and September 7 reorder: F is matched
branch retraining, G is the Peyer window, and H is the classifier ROC comparison.
Rendering never assembles or overwrites the user's final artwork.

Classifier architecture update (7 September 2026): G/H require the representation-
pretrained EMA teacher and the exact shared, scale-aware axis backbone. Full-model
SFT with manual targets is prespecified for deployment. Four models compare
manual/GAT-soft targets and frozen-encoder/full-model SFT. Labels, evaluation
cells, display windows and the 0.5 threshold remain fixed. A–F retain their
existing scientific inputs. The release must pass its checks before rendering.

| Panel | Generated component | Frozen source |
|---|---|---|
| A | One row: masked student local, EMA-teacher local, masked student context, EMA-teacher context | Exact Figure 3a Cont1 region and paired-representation pretraining settings |
| B | Paired-representation pretraining history | 35 completed epochs; validation-selected epoch 33 |
| C | Released supervised fine-tuning history | Production-v2 seed 44; 20 completed epochs; selected epoch 9 |
| D | Pretraining initialization comparison | Complete upstream recipes differ; matched 5% SFT; seeds 42–44 |
| E | Branch-input shuffling, crypt-villus above epithelial distance; local/context/fine left to right | Frozen raw-crop batch-shuffle outputs; five repeats; formerly panel F |
| F | Matched branch retraining: full model and each omitted branch | Frozen, separately trained comparison; same 5,107 evaluation cells |
| G | 2×2 held-out Peyer window: DAPI / manual label above probability / predicted class | Representation-pretrained shared-scale binary classifier; fixed original window |
| H | Peyer's patch classifier strategy | Complete 2×2 ROC comparison of manual/GAT-soft targets and frozen-encoder/full-model SFT |

A's authoritative prepared views are SATA
`Figure_3_Extended/stored_outputs/paired_view_repair_v1/paired_views.npz`, hash
`e5982d9a7f2fc4e6a8320a1d2515f9c48a84d203305315c78b50fb7b4787ba2d`.
Its manifest records the actual frozen dataset pair, shared geometry/intensity,
seed and teacher images. Gray (`#808080`) blocks illustrate learned-token
masking; only the displayed fill changes, with frozen images and patch indices
preserved. G reads SATA
`frozen_preprocessing/peyer_representation_v1/image_windows/` and requires its manifest,
including `hard_class_fixed_0p5`. Probabilities and manual labels are repaired;
the original window and its 3,373 cells are unchanged.

Panels D–E read saved outputs produced by frozen preprocessing; plotting never
trains a model, runs an ablation, or executes inference. The pretraining
comparison and branch analyses use the 5,107-cell prepared development test,
not the locked SI2 confirmatory section. This limitation is stored in
provenance rather than printed across the artwork. Panels G–H use the same
encoder initialization and backbone architecture as the axis ViT, with a binary
output. The new fine CNN and classifier layers train in every condition. Panel H
renders four saved 20-epoch runs and does not select or retrain a model. All
checkpoint selection finishes before evaluation of the unchanged 819-cell test
cohort. ROC coordinates and display metrics are frozen under SATA
`frozen_preprocessing/peyer_representation_v1/roc/` and read directly during plotting. Its renderer contains no ROC or metric
calculation; the explicit one-time conversion and hashes are documented in
`panel_H/processing/README.md`.

Panel F displays the approved matched branch-retraining comparison. Historical
alternative code and pre-relabel A/F/H code are preserved read-only under
`/home/amonell/Desktop/spatial_axes_manuscript/lineage_archive/final_figure_organization_2026-09-04/Figure_3_Extended/`.
Historical frozen outputs and checkpoints remain untouched; the current F reads
the separately verified matched-branch evaluation.

All generated typography is Arial with embedded TrueType fonts. Outputs are
written beneath `/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3_Extended/panel_X/`.
Each script hash-checks its frozen inputs, writes an adjacent provenance JSON,
and refuses existing declared outputs unless `--overwrite` is supplied.

## Sources and required fields

All relative source paths below are rooted at the home project except the
explicit SATA region inputs. Exact full commands are emitted by `--dry-run`.

| Panel | Primary source | SHA256 | Required fields |
| --- | --- | --- | --- |
| A | SATA `Figure_3/stored_outputs/panel_A/panel_A_exact_historical_region.npz` | `1b58cdfe13909c7da88c225e9f98250b6bbf4bb485e05c7fdaeaa3900e77bbaa` | `local_model_input`, `context_model_input`; companion metadata locks nucleus 352082 |
| B | `Figure3/train_model/outputs/paired_representation_pretraining/pretraining/pretrain_history.csv` | `d3227f0c39ee235cb04335b8da0f55dad87a8f46398fae243a56e014208fbb59` | completed epoch, training/validation composite loss, alignment loss, feature standard deviation |
| C | `Figure3/train_model/outputs/production_representation_model_v2/history.csv` | `72dde255d67cf441e39d54042b354f06ae2b410551d8ddf262518a60312e3516` | epoch, training/validation Smooth L1 and coordinate MAEs |
| D | `Figure3/train_model/outputs/paired_representation_pretraining/metrics/performance_by_seed.csv` | `6fc901ad0041a81dea630c9b9a00ca413da3bb1cb6406cb98d35ae787e890bae` | objective variant, seed, coordinate Pearson r |
| E | `Figure3/train_model/outputs/production_representation_model_v2/scale_importance/scale_importance_deltas.csv` | `23608adef77b6a27417187df298ad466c960fedf5710d8dc348e91e8ef061247` | `scale`, `axis`, `repeat`, `pearson_r_drop` |
| F | SATA `frozen_preprocessing/model_design_comparisons_v1/image_branches_evaluated_v1/release.json` | Release hashes lock the matching audit and metrics | `condition`, `axis`, `n`, `pearson_r`, `mae` |
| G | SATA `frozen_preprocessing/peyer_representation_v1/image_windows/display_window_fixed_threshold.csv` | Pinned by `peyer_representation_v1/active_release.json` and its window manifest | `batch`, `x_spatial`, `y_spatial`, `peyer_manual_label`, `peyer_probability`, `hard_class_fixed_0p5` |
| H | SATA `frozen_preprocessing/peyer_representation_v1/vit/strategy_metrics.tsv` | Pinned by `peyer_representation_v1/active_release.json` and its comparison manifest | variant plus each variant's saved labels/probabilities; release manifest checks all four prediction hashes |

All companion summaries, checkpoints, and source image hashes are enforced in
the corresponding rendering script. The final coordinate checkpoint is
production-v2 (`d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137`);
the selected Peyer image classifier is separate
(`d16035223171d2017a44c64db2ebaed6ead3e45fc05d74bd10c94d6e27806e4a`).

Expected PDF/PNG stems, in display order:

1. `panel_A_paired_pretraining_views`
2. `panel_B_paired_representation_pretraining_history`
3. `panel_C_released_supervised_training_history`
4. `panel_D_pretraining_objective_ablation`
5. `panel_E_branch_input_shuffling`
6. `panel_F_matched_branch_retraining`
7. `panel_G_heldout_peyer_prediction`
8. `panel_H_peyer_classifier_strategy`

## Regeneration

```bash
/home/amonell/Desktop/spatial_axes_manuscript/Figure2/.venv/bin/python \
  /home/amonell/Desktop/spatial_axes_manuscript/Figure_3_Extended/run_all_panels.py \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3_Extended
```

To inspect only the changed components without opening data or writing files:

```bash
python Figure_3_Extended/run_all_panels.py \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3_Extended \
  --panels A E F --prepared-only --dry-run
```

Remove `--dry-run` to render only A/E/F. Archive active components before using
`--overwrite`. `--prepared-only` is accepted for uniform orchestration; every
panel already consumes frozen inputs. Selection occurs before input-file I/O.

For an A-only preview in a fresh directory, pass `--panels A --output-root
/tmp/Extended3_A_preview`. Panel A's prepared inputs default to the canonical
SATA `Figure_3_Extended/stored_outputs/paired_view_repair_v1/`, independently
of the output location. `--prepared-root` explicitly changes that input
directory while retaining the frozen manifest and array hash checks.

The exact-layout adapter invokes these same panel scripts and owns only final
page geometry, shared labels, and typography in `assembly/layout.json`. Every
returned element is tagged with its panel letter for independent final-page
panel exports. No assembled PDF is used as a scientific plotting input.
# Matched branch retraining (added September 6; reordered September 7)

Panel F contains the separately retrained full/no-local/no-context/no-fine
production-v2 comparison. It is generated by
`panel_F/figure_script/make_panel_F_matched_branch_retraining.py`; its matching
audit and frozen reevaluation are documented in `panel_F/processing/README.md`.
Use `run_all_panels.py --panels F --output-root
/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3_Extended`.
Expected outputs are `panel_F/panel_F_matched_branch_retraining.pdf`, `.png`
and `panel_F_provenance.json`. All four models use the same 5,107 development
cells. The default renderer and generated complete page now include A–H.
All existing panels remain; E is arranged horizontally and the histories
are more compact. Original artwork and its geometry measurements are preserved.
F is a single row of four coordinate/metric plots above G/H in the complete page, while
its standalone component is 2×2. Neither version contains the removed footer.
