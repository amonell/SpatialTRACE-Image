# Figure 3 reproducible component package

For the **finished panel layout**, run from the home project:

```bash
python -B update_final_figure.py Figure_3 --overwrite
```

This regenerates the page and publishes `panel_X_final.pdf`,
`panel_X_final.png`, and `panel_X_final_provenance.json` to
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3/panel_X/`.
These PDFs contain only their own panel on the registered full-page canvas;
import at the page origin without rescaling. `final_panel_manifest.json` in
the SATA figure directory identifies each final renderer and its source hashes.
Use `--dry-run` to inspect commands without writes. Previous final exports
are archived; original assembled PDFs remain untouched. See
[the final-figure editing guide](../FINAL_FIGURE_EDITING.md).
The component commands below provide smaller individual assets, not a second
authoritative final-page layout.

This directory regenerates the data and image components visible in the newly
supplied `Figures/Figure3.pdf` using the released
`paired-representation-shared-scale-v2` checkpoint. The approved `assembly/`
adapter composes the complete page using preserved original manual vectors;
component scripts never redraw those vectors.

| Panel | Generated component | Frozen lineage |
|---|---|---|
| A | Original Cont1 IF crop stack and paired pretraining examples; Xenium supervised inputs, two head predictions and two regional prediction maps; architecture data TSV | Cont1 pretraining rows 2352/5035; Day90 Xenium prepared row 4540 and 2,744 saved current-model predictions |
| B | 30,000-cell deterministic embedding UMAP | Released production-v2 checkpoint and locked `sample_008` subset |
| C | Coordinate scatterplots using all 50,000 locked `sample_008` cells | Released production-v2 checkpoint |
| D | Full and zoom crypt-villus reference/prediction maps | Same 50,000-cell production-v2 predictions |
| E | Full and zoom epithelial-distance reference/prediction maps | Same 50,000-cell production-v2 predictions |

The Figure 3c-e references are dense outputs of the frozen graph-coordinate
model, not direct manual labels. The graph model itself was fit from sparse
manual annotations. Figure 3a's imported microscopy region is the exact
historical Cont1 IF pretraining example centered at nucleus label 352082
(`x=15709.439790575916`, `y=22534.952879581157` full-resolution pixels). It is
present in multimodal pretraining and was not a Xenium supervised-coordinate
row. The prior provisional day30 Xenium example is archived and is not active.

## Current training-matched preprocessing release

Use SATA `Figure_3/stored_outputs/training_matched_inputs_v1/manifest.json` for
B–E. `assembly/input_lock.json` pins its SHA256 and all consumed output hashes.
This authorized refresh replaces the earlier full-resolution floating-point
inference route with the exact supervised image-pyramid preparation and uint8
roundtrip. All 1,002 branch tensors from 334 overlapping supervised-test cells
must be byte-identical to their saved inputs, and all 334 predictions must match
their frozen evaluation before release.

The checkpoint, current graph-reference values, 50,000 cell identities/order,
30,000 embedding identities/order, physical fields and spatial zooms are fixed.
Predictions and fused embeddings are newly computed; one new UMAP uses the
original subset and settings. `reference_repair_v1/` and the original UMAP
remain preserved historical inputs, not the active rendering source.

Authoritative outputs, each hash-locked by the manifest, are
`predictions_current_references.parquet`,
`embedding_metadata_current_references.csv`, `figure3f_embeddings.npz`,
`figure3f_umap.csv`, `metrics.json` and `comparison.json`.
C–E require finite reference/prediction coordinates and locked spatial centroids
for all 50,000 rows; B requires the original embedding identities/order.
Every B–E command requires `--reference-manifest`; the runner passes it
explicitly and refuses the old raw-image release. `comparison.json` records
the old and training-matched performance against identical graph references.
The tissue's prior development exposure is unchanged and is documented
off-artwork; the example is not an untouched architecture-selection test.

Full preparation, inference and UMAP commands are in
`frozen_preprocessing/figure3_training_inputs_v1/README.md` under the home
project. Those stages must complete and freeze before final rendering.

All generated typography uses embedded Arial. No rendering script performs
training, checkpoint selection, ablation, UMAP fitting, or whole-section model
inference. The source was chosen by the user for its preferred whole-tissue
morphology. Its stable ID is `sample_008`; the mounted source path resolves to
`D8_SI_R2`, while an older prose alias called it `day6_SI_r2`. The active code
therefore uses only the stable source ID and exact hash-locked image path.

`panel_order.json` records the artist import order, including full reference,
full prediction, reference zoom, prediction zoom for D and E. The reviewed
artwork hash is in the root `final_figure_order.json`. Manual vectors are not
redrawn. The former alternative-zoom and provisional training-example code is
preserved under `lineage_archive/final_figure_organization_2026-09-04/` and is
not part of final regeneration. Saved candidate data and exports remain on SATA.

```bash
/home/amonell/Desktop/spatial_axes_manuscript/Figure2/.venv/bin/python \
  /home/amonell/Desktop/spatial_axes_manuscript/Figure_3/run_all_panels.py \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3 \
  --prepared-only
```

Use `--panels A B`, `--dry-run`, or `--prepare-only` to select final components,
inspect exact commands without writes, or explicitly prepare the locked crop.
`--overwrite` requires prior archiving when invoking this local runner directly;
the root orchestrator archives automatically. Final rendering never regenerates
zoom candidates, fits UMAP, or changes the frozen model/split.

`figure_inputs.py` is shared by component and assembly renderers for paired
views and prediction overlays. The paired thumbnails use the same frozen
`Figure_3_Extended/stored_outputs/paired_view_repair_v1/paired_views.npz` as
Extended Figure 3a (SHA256 `e5982d9a7f2fc4e6a8320a1d2515f9c48a84d203305315c78b50fb7b4787ba2d`),
with its manifest (SHA256 `22c7cede602b605e8b3eed02aa0f4579988427b674bd5ccf8f1c1896758f2e7d`).
Teachers retain clean variant 0. Both student scales use variant 1 (90-degree
rotation) and the same frozen contrast, brightness, and blur parameters.
The renderer adds 102 black patches per student image to illustrate token
masking; actual training replaces randomly masked tokens, not input pixels.
This corrects only four simplified paired thumbnails in the previous baseline;
their registered positions and all other figure content are unchanged. Both
renderers accept explicit paired-manifest and paired-views paths and hash-check
the same center, prepared rows, variants, and arrays. The four
prediction overlays now use the user-approved Xenium example described below;
the original pretraining images and their positions remain unchanged.

## Approved Xenium supervised-side example (6 September 2026)

The three supervised input thumbnails and four prediction thumbnails use
`sample_010_cell_0214763` from Day90 SI replicate 2. Prepared row 4540 is the
golden input cell nearest the geometric center of the already locked Figure 1d
window. Neither predictions nor reference values enter that choice. All 2,744
cells inside its native 3,132-pixel context field are retained; predictions are
copied numerically without change from the corrected Figure 1d frozen snapshot.
The checkpoint is the released Xenium model, not an IF-adapted substitute.

Both rendering paths read the same hash-locked SATA snapshot:
`Figure_3/stored_outputs/panel_A/supervised_xenium_v1/`.

- `manifest.json`: `ca1eac49e2f9d0aefa2b290aa56c62a82efd06ade925923577b3b5836f836f7e`
- `images.npz`: `1489ff8f6eb24cb22ec7db2e431d1bd011882573daec4bd14b3eb1e146d0e858`
- `predictions.csv`: `9759db8a50d033d6b3c242d25154c5f00225df12889b2a97db8b4b07a4258034`

Required arrays are `local_image`, `context_image`, `fine_image` (exact original
supervised uint8 tensors) and `context_dapi` (native microscopy display field).
The table contains stable row/global identities, full-resolution centroids and
both predicted coordinates. `supervised_example.py` validates all fields and
draws fixed-radius vector centroid markers using the unchanged 0–1 viridis
scale. Markers denote cell locations, not segmented nuclear boundaries.

The input fields match the released supervised direct-pyramid/uint8 protocol;
all three illustrated tensors are byte-identical to the saved inputs used for
verified inference. The Cont1 pretraining thumbnails retain their distinct
512-intermediate preparation. Historical Cont1 prediction snapshots are
preserved but are no longer active supervised-side components.

This is an in-domain qualitative inference example from a previously examined
SFT-test section, not a supervised-training member or a fresh performance
evaluation. The original raw-manifest `split` column is deliberately omitted
because it is not the released SFT assignment. Pretraining exposure is not
excluded. Main B–E cells, predictions, embeddings, zooms and geometry are unchanged.

The one-time microscopy/subsetting command is documented in
`panel_A/processing/README.md`. Figure rendering only reads the saved snapshot;
it never reruns inference. `panel_order.json` lists all 13 component stems and
the existing update command publishes the complete A–E final layers together.

`zoom_region_lock.json` contains the unchanged accepted D/E zoom coordinates,
transcribed with the original provenance hash. Neither rendering path selects
zoom candidates or depends on a mutable prior component output for geometry.

For exact final styling, `assembly/` exports transparent A4 layers
`scientific_panel_A.pdf` through `scientific_panel_E.pdf` from the same live
artists as the complete scientific layer. Source-map provenance records axis
ownership. Individual artist-import crops remain available through
`run_all_panels.py --panels A` (or another letter).
