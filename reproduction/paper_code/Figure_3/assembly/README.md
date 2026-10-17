# Complete Figure 3 assembly

`layout.json` registers the five-panel September 5 page. Every microscopy and
map slot is measured from the source image transform; scatter-frame coordinates
are measured from original vector paths. `input_lock.json` locks the reviewed
scientific inputs and summaries. Original artwork hashes and authoritative
upstream release hashes are both verified before rendering.

Panel A retains the original manual shapes, arrows, frames, encoder stacks,
readout grids, and measured text baselines. All original images and text,
scientific prediction polygons, old student masks, and Illustrator private data
are removed structurally. Pretraining fields retain the exact original Cont1
prepared crop; the approved supervised-side replacement uses Xenium. The two students share the frozen spatial
and intensity augmentation; teachers retain the clean original crops. In the
student views, 102 of 256 tokens are masked. The existing text slots identify
shared transformer weights, retained learned scale embeddings, paired
representation learning, and graph-derived coordinate supervision. Crop fields
are labeled by physical width; resized network inputs remain in pixels. The
training counts, schedules, and checkpoint selection are described in Methods
instead of the schematic. This text-only simplification preserves all original
manual paths, scientific components, and registered baselines.

The two transformer stacks preserve their original interleaved fill and stroke
order in the panel-A-only `manual_transformer_stacks.pdf` layer. Separating
their fills and strokes makes hidden rear-card borders show through the front
cards. `manual_path_lock.json` locks all 28 path events, their operations, and
the two stack bounds. Extraction verifies their order and geometry. The short
`Scale embeddings` label retains its original baseline and fits the original
text footprint in 5.1-point Arial. Run the focused regression checks with
`python3 -B -m unittest Figure_3.assembly.test_transformer_art -v`.

The three supervised inputs and four output thumbnails now use the approved
Xenium snapshot in SATA `Figure_3/stored_outputs/panel_A/supervised_xenium_v1/`.
The illustrated cell is prepared row 4540 from Day90 SI replicate 2. It is the
verified input cell closest to the existing Figure 1d window center; selection
uses only geometry. All 2,744 cells within its native context field retain their
exact saved current-model predictions. The three input tensors are byte-identical
to the saved supervised arrays used during verified inference.

`../supervised_example.py` supplies the same frozen arrays and vector centroid
markers to both final-page and component rendering. Markers represent cells,
not reconstructed nuclear masks. The original Cont1 contours (source paint
events 3963, 3966 and 3995) are removed from the three substituted input slots;
adjacent colored frames remain intact. Their exact geometry and operations are
checked before removal. All other manual paths remain unchanged.

This is a qualitative example from a previously examined SFT-test section,
not a training member or a fresh performance evaluation. Rendering performs
no inference. Earlier Cont1 coordinate snapshots and their protocol checks
remain historical records. Pretraining retains its distinct 512-intermediate
recipe; the Xenium supervised inputs use the verified direct-pyramid/uint8
recipe. See `supervised_xenium_review.md` for final source and change-scope QA.

Panel B reads the newly frozen UMAP of training-matched embeddings for the
original 30,000 cells. Panels C–E retain all 50,000 locked cells and the current
graph-derived reference lineage, with newly computed released-model predictions
using exact supervised image preparation. All B–E inputs resolve through
`training_matched_inputs_v1/manifest.json`, pinned in `input_lock.json`.
Metrics are read from the frozen metrics JSON. Zoom limits
are read from `../zoom_region_lock.json`, transcribed unchanged from the
verified original panel D provenance, rather than reselected.
The original example has prior model-development exposure; its spatial maps
are labeled `Example section`. Graph references are computed coordinates.

Crypt-villus maps and UMAP use 0–1; epithelial-distance maps and UMAP retain the
active renderer's 0–0.7 limits. The separate schematic prediction bar uses
0–1. Coordinate gradients are 256 opaque vector bands. Count-colorbar limits
follow the freshly rendered fixed-grid hexbin counts. Every label is embedded
Arial; dense clouds and microscopy are rasterized at 300 dpi.

```bash
python3 -B Figure_3/assembly/prepare.py --output-dir /tmp/figure3-assembly-review
```

The callable `prepare(output_dir, overwrite=False)` returns the shared compositor
contract. It writes fresh scientific and clean static PDF layers and numeric
provenance. The CLI also writes `preparation.json`. Existing outputs require
explicit `--overwrite`. The root complete-page runner composes the final PDF,
PNG, and measured anchor/font/text/private-content checks. Original assembled
PDFs under SATA `Figures/` remain unchanged.
