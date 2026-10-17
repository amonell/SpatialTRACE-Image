# Extended Figure 2 page 2: corrected-label classifier components

For the **finished panel layout**, run from the home project:

```bash
python -B update_final_figure.py Figure_2_Extended_2 --overwrite
```

This regenerates the page and publishes `panel_X_final.pdf`,
`panel_X_final.png`, and `panel_X_final_provenance.json` to
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2_Extended_2/panel_X/`.
These PDFs contain only their own panel on the registered full-page canvas;
import at the page origin without rescaling. `final_panel_manifest.json` in
the SATA figure directory identifies each final renderer and its source hashes.
Use `--dry-run` to inspect commands without writes. Previous final exports
are archived; original assembled PDFs remain untouched. See
[the final-figure editing guide](../FINAL_FIGURE_EDITING.md).
The component commands below provide smaller individual assets, not a second
authoritative final-page layout.

This package generates the components in SATA `Figures/Figure2_Extended_2.pdf`
without replacing the artist's assembled page. The corrected-label refits were
explicitly authorized on 5 September 2026.

| Panel | Generated component | Frozen source |
| --- | --- | --- |
| A | Exact positive and sampled negative training seeds | Four training sections; seed 42; 2,509 positives and 62,725 negatives |
| B | Loss, average precision, AUROC and F1 history | 22 completed epochs; validation-AP-selected epoch 10 |
| C | DAPI, corrected manual labels, probabilities and hard classes | Unchanged 3,457-cell Day 90 replicate-2 window; threshold 0.95 |
| D | Two test sections and test-only pool | 500,050 test cells; no validation cells in the pooled bar |

## Authoritative inputs

The model is the separate BinaryGAT, not the coordinate GAT or image classifier.
It uses 30-dimensional X_scVI features, 20 spatial neighbors within each section,
two attention layers (128 hidden features; four first-layer heads), and a binary
linear output. Training sections are day6_SI, day8_SI_Ctrl, day30_SI and day90_SI;
validation sections are day6_SI_r2 and day8_SI_r2; test sections are day30_SI_r2
and day90_SI_r2.

- Checkpoint: SATA `frozen_preprocessing/peyer_label_repair_v1/gat/gat_peyer_classifier.pt`
- Checkpoint SHA256: `c4f9c94b81f47ff01fe25a3f5587a0e9258ec0f2799c4106abd986a9853e82cc`
- Classifier release: SATA `frozen_preprocessing/peyer_label_repair_v1/release_manifest.json`
- Release SHA256: `e378b349ec48db306bed6ffd255190c04e377078b86ff8bf637660f509da1699`
- Prepared inputs: SATA `Figure_2_Extended_2/stored_outputs/corrected_labels_v1/prepared_inputs_manifest.json`
- Prepared manifest SHA256: `ef18bf9315c658c5b2fa8bfea1e5bc471dbde39d31af37eb4b5c978d2d6bebe6`

Required fields include global_index, batch, split, x_spatial, y_spatial,
peyer_manual_label, gat_peyer_probability and gat_peyer_class. History needs
completed epoch, train/validation objective, AP, AUROC and F1. Prepared manifests
hash every compact input, source checkpoint and unchanged microscopy crop.

## Regeneration

Run from the home code root:

```bash
Figure2/.venv/bin/python Figure_2_Extended_2/run_all_panels.py --prepared-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2_Extended_2/stored_outputs/corrected_labels_v1 --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2_Extended_2
```

Use `--panels A C` for selection and `--dry-run` to inspect all explicit input
arguments. Archive active exports before `--overwrite`. The runner never trains,
resplits or infers. The processing entry point forwards to
`frozen_preprocessing/peyer_label_repair_v1/prepare_graph_panels.py`; its exact
completed command is in the prepared manifest. Use a new output directory if
rebuilding compact tables. The old threshold-0.88 preparer remains in
`frozen_preprocessing/scientific_repair_v1/original_sources/`.

Final components are under SATA `Figure_2_Extended_2/panel_X/`.
Expected stems, with PDF/300-dpi PNG and adjacent provenance, are listed in
`panel_order.json`. C import order is morphology, manual labels, probability,
then hard class; B is loss/AP above AUROC/F1-at-0.5. All text is embedded Arial.

## Evaluation limitations

The supervised split is section-disjoint, but scVI and historical feature scaling
were jointly learned. Negatives mean outside annotated positive polygons, not
exhaustively validated negative anatomy. The original empty Day 6 polygon is
preserved. The validation-F1 threshold 0.95 is the upper edge of the unchanged
grid; no test-driven grid extension occurred. The repaired evaluation uses
previously evaluated sections, not a new untouched confirmatory cohort.
