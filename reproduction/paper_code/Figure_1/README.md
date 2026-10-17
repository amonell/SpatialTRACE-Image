# Figure 1 reproducible component package

For the **finished panel layout**, run from the home project:

```bash
python -B update_final_figure.py Figure_1 --overwrite
```

This regenerates the page and publishes `panel_X_final.pdf`,
`panel_X_final.png`, and `panel_X_final_provenance.json` to
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_1/panel_X/`.
These PDFs contain only their own panel on the registered full-page canvas;
import at the page origin without rescaling. `final_panel_manifest.json` in
the SATA figure directory identifies each final renderer and its source hashes.
Use `--dry-run` to inspect commands without writes. Previous final exports
are archived; original assembled PDFs remain untouched. See
[the final-figure editing guide](../FINAL_FIGURE_EDITING.md).
The component commands below provide smaller individual assets, not a second
authoritative final-page layout.

Figure 1 combines newly rendered scientific components with preserved original
manual vectors. The approved `assembly/` adapter composes the complete page;
it never redraws the original arrows or workflow boxes. Individual component
exports and assembly share the same registered scientific regions.

All generated Figure 1 typography uses embedded Arial.

## Panel map

| Panel | Freshly generated components | Frozen lineage |
|---|---|---|
| A | Day 90 whole tissue, gray villus zoom, DAPI Peyer's patch crop | Frozen graph input tables and raw Xenium DAPI |
| B | Sparse continuous label and held-out Peyer's region label | Frozen continuous references and corrected complete Peyer polygons |
| C | Sparse label graph, dense coordinate output, Peyer's probability output | Frozen graph-coordinate and Peyer's GAT checkpoints; the displayed neighbor/expression diagrams remain original conceptual vectors |
| D | Four artwork rectangles: graph-coordinate training crop above paired DAPI; distinct Day 90 alternate-region DAPI above ViT output | Frozen graph visualization output and released Xenium `paired-representation-shared-scale-v2` checkpoint (`d0e02005...f137`) |
| E | Raw multichannel IF crop and repaired production-v2 crypt-villus prediction map | Buffered/shared-reference IF checkpoint (`b512cb49...7b2`), gated on completed evaluation |

All model architecture, training, checkpoint-selection, inference, evaluation,
ablation, and scale-importance code is frozen under
`/home/amonell/Desktop/spatial_axes_manuscript/frozen_preprocessing/`. Figure
scripts consume only frozen outputs or deterministic crops/subsets.

B/C now use the corrected-label Peyer BinaryGAT, not its preserved historical
checkpoint. SATA `Figure_1/stored_outputs/peyer_repair_v1/manifest.json`
(`7939291480055fc565cd061b8188093db84c84c8f0f6bdcda2813909b25a1b5c`)
locks the unchanged cells/region and new labels/probabilities. The graph checkpoint
SHA256 is `c4f9c94b81f47ff01fe25a3f5587a0e9258ec0f2799c4106abd986a9853e82cc`.
E requires the completed shared-reference IF release; the runner cannot fall
back to the old IF corpus. Original source images and Figure 1d inputs remain fixed.

Panel D's exact level-0 windows are locked in
`panel_D/processing/figure1d_region_lock.json`: the left paired training region
is `x=18616:22406, y=9364:11768`, and the right DAPI/model-output region is
`x=17548:23863, y=35764:39652`, both from `day90_SI_r2`. The panel-specific
inference script runs the released model only on the latter fixed region and
writes those bounded predictions beside the prepared Figure 1 inputs. Region
membership is never chosen from prediction performance.

The active D prediction input is SATA
`Figure_1/stored_outputs/figure1_inputs/panel_D_training_input_parity_v1/predictions.csv`
(SHA256 `e68031350928f9848f11b69e2dd05ebb91839fb30dfba3a0b6c851894ec98729`).
The same 4,854 cells and checkpoint now use the pixel-identical supervised crop
protocol: native 783/3,132/196 fields, original pyramid selection, direct
256/256/128 resizing, uint8 quantization, then float32/255. All 75 overlapping
prepared tensors match exactly. The prior inference release remains preserved.
The whole-page and standalone D renderers both require this parity evidence.

## Regenerate from numeric and raw-image inputs

Using Figure 2's existing frozen compact graph tables, run:

```bash
/home/amonell/Desktop/spatial_axes_manuscript/Figure2/.venv/bin/python \
  /home/amonell/Desktop/spatial_axes_manuscript/Figure_1/run_all_panels.py \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_1 \
  --graph-input-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2/stored_outputs/shared_graph_components \
  --prepared-only
```

Rendering existing frozen compact tables, crops, and bounded predictions is
the default; `--prepared-only` explicitly selects that same behavior. No
preprocessing or model inference runs by default. `--prepare-only`
runs explicit input preparation without rendering; the IF multichannel crop is
then reread from the raw VSI with `bffile`. `--panels A D E` selects panels;
`--dry-run` prints shell-escaped commands without executing them.
The immunofluorescence-quantification graphs in panel E are manual artwork and
are intentionally not generated or included in this component package.
Outputs are written to the SATA `Figure_1/panel_X/` folders. Existing outputs
are protected unless `--overwrite` is provided after archiving.

The A/B source crops are shared in `assembly/source_region_lock.json`. B colors
only the original illustrated villus_004 annotation; other labeled villi can
lie within the window. C and D use the identical locked training-region table
and bounds. No render-time region selection occurs. A's dense PDF point layers
now explicitly use the requested export DPI.

For exact final-page styling, assembly also exports `scientific_panel_A.pdf`
through `scientific_panel_E.pdf`: transparent A4 layers drawn from the same
live artists as `scientific_layers.pdf`. Their per-axis panel ownership is
recorded in preparation provenance. Standalone component crops remain useful
for artist import, but are not a second authoritative page-layout renderer.

`component_index.tsv` contains final component hashes and `QA.md` records PDF
preflight. The previous copied component bundle is preserved in
`polished_panel_archive/pre_true_regeneration_2026-09-02/`.

The current root `Figures/Figure1.pdf` determines displayed panel order.
`panel_order.json` lists every final PDF/PNG stem in artwork display order.
Figure 1a's former extra colored-label zoom and Figure 1e's unused epithelial
map are excluded from active rendering. Pre-change code is preserved under
`lineage_archive/final_figure_organization_2026-09-04/Figure_1/`.
