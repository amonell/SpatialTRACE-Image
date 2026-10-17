# Figure 2 reproducible component package

For the **finished panel layout**, run from the home project:

```bash
python -B update_final_figure.py Figure_2 --overwrite
```

This regenerates the page and publishes `panel_X_final.pdf`,
`panel_X_final.png`, and `panel_X_final_provenance.json` to
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2/panel_X/`.
These PDFs contain only their own panel on the registered full-page canvas;
import at the page origin without rescaling. `final_panel_manifest.json` in
the SATA figure directory identifies each final renderer and its source hashes.
Use `--dry-run` to inspect commands without writes. Previous final exports
are archived; original assembled PDFs remain untouched. See
[the final-figure editing guide](../FINAL_FIGURE_EDITING.md).
The component commands below provide smaller individual assets, not a second
authoritative final-page layout.

Every Figure 2a-e data component is rendered from numeric inputs by the code in
this directory. The scripts do not copy legacy PDF/PNG files and do not assemble
the manually edited page in `Figures/Figure2.pdf`.

## Panel map

| Panel | Freshly generated components | Evaluation scope |
|---|---|---|
| A | Four section class maps in a 2×2 grid, scVI latent projection, 20-neighbor spatial graph, sparse-coordinate, dense-coordinate, and separate Peyer-classifier insets | Descriptive inputs and examples from frozen models |
| B | Day 90-only whole-section sparse labels and zooms, crypt-villus above epithelial-distance | Annotation display |
| C | Day 90-only whole-section and zoom coordinate maps, crypt-villus above epithelial-distance | Frozen all-label graph model; visualization, not a test metric |
| D | Coordinate scatterplots | Final eight-fold, section-disjoint OOF evaluation; 25,436 cells |
| E | Performance versus annotated training villi | Eight whole-section test folds with separate validation sections |

`dataset_processing/build_graph_component_tables.py` makes one deterministic
compact table set from the 16-GB H5AD and the hash-locked graph predictions.
Panels A-C read those compact tables; A's separate classifier inset also reads
the corrected-label BinaryGAT's frozen test-window table. Panel D reads the frozen OOF table;
panel E reads the completed 160-run section-disjoint scaling output. No figure script trains or
selects a model.

Panel D uses manual LabelMe-derived coordinate annotations as its reference,
so its x-axis is `Manual reference`. Each scatter visibly reports only Pearson
`r`, MAE, and R²; the complete metric record remains in provenance. All
generated Figure 2 typography uses embedded Arial.

## Frozen inputs

- Registry: `/home/amonell/Desktop/spatial_axes_manuscript/frozen_preprocessing/frozen_model_registry.json`
- Graph visualization predictions: SHA256 `89f3f115...c5e0` and `cc6cdc76...b5aa`
- Sparse labels: SHA256 `6eabc5f2...be2c`
- Section-disjoint OOF predictions: SHA256 `56b4647a...a660`
- Section-disjoint split: SHA256 `746c72b9...10f2`
- Section-disjoint scaling split: SHA256 `e1779c3c...0f8`
- Scaling outputs: `/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure2/section_disjoint_villus_scaling_v1/`

Panel E reuses the eight prespecified panel-D section folds. For every training-villus
count, selected villi are drawn from a six-section training pool, a seventh complete section
selects the checkpoint, and the eighth complete section is evaluated once. The
curve therefore ends at 45 villi, the largest count available in every fold. The
mean and SEM are across held-out sections. The upstream 30-dimensional scVI
features were learned jointly across the dataset; that feature-lineage limitation
is retained in the machine-readable provenance.

## Regenerate Figure 2a-e

```bash
/home/amonell/Desktop/spatial_axes_manuscript/Figure2/.venv/bin/python \
  /home/amonell/Desktop/spatial_axes_manuscript/Figure_2/run_all_panels.py \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2 \
  --prepared-only
```

Outputs are protected unless `--overwrite` is given after archiving. The prior
copied component bundle is preserved in
`polished_panel_archive/pre_true_regeneration_2026-09-02/`. Final hashes and
PDF preflight appear in `component_index.tsv` and `QA.md`.

`panel_order.json` locks the displayed order and final component filenames from
the current root `Figures/Figure2.pdf`. Main panels B/C contain only Day 90;
Day 6, Day 8, and Day 30 label/prediction maps remain in `Figure_2_Extended`.
The workflow's three numerical insets have their own real panel-A renderer and
export stems in the order manifest. The architecture boxes and arrows remain
manual artwork, separated by panel in the fixed-layout adapter.

Use `--panels A B C` to select panels and `--dry-run` to print commands.
The default uses existing prepared inputs; `--prepared-only` explicitly selects
that same behavior. `--prepare-only` performs shared input preparation without
plotting. Model fitting is never run by this runner.
Pre-change code is archived under
`lineage_archive/final_figure_organization_2026-09-04/Figure_2/`.

## Independent panel updates

Frozen prepared inputs and new outputs use separate roots. For an A-only
preview in a new directory, with no preparation or fitting:

```bash
python Figure_2/run_all_panels.py --panels A --prepared-only \
  --prepared-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2/stored_outputs/shared_graph_components \
  --output-root /tmp/Figure2_A_preview
```

The prepared root defaults to the canonical path above; it never defaults to a
fresh output directory during rendering. All three original A components and
all three schematic insets are exported. Each renderer validates its selected
tables against the hash-locked manifest. Main E now pins the same metrics bytes
as the extended section-disjoint plots.

Standalone components use their own importable canvas. The complete-page
adapter calls those same scripts and applies `assembly/layout.json` for exact
page placement, typography, shared labels, and viewports. Change data styling
in the panel renderer; change final-page labels/positions in the explicitly
owned layout elements. Assembly does not import previous component PDFs.
See `assembly/README.md` for exact-layout regeneration and QA.
