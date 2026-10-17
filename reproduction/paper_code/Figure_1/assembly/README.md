# Complete Figure 1 assembly

This adapter rebuilds the five-panel page at the measured September 5 artwork
geometry. `layout.json` locks the artist PDF and stores top-left PDF-point image
anchors measured from its image transforms. `input_lock.json` additionally locks
every numeric, microscopy, and model-release input used by rendering.

The source PDF supplies only the reviewed manual paths and measured text
anchors. The conceptual nine-node neighbor diagram, illustrative expression
bars, arrows, frames, and panel E curves retain their original vector paths.
Every original image, live text object, and Illustrator private payload is
removed. Original opaque image-cleanup rectangles 46–48 and colorbar base
rectangles 53, 125, and 132 are explicitly excluded. No background-page copy or
white-mask substitution is used.

Fresh scientific layers are rendered from graph tables, calibrated microscopy
arrays, corrected Peyer labels and probabilities, released-model bounded
predictions, and current IF-release predictions. Training and inference windows
are locked by `panel_D/processing/figure1d_region_lock.json`. Rendering performs
no training, inference, checkpoint selection, split construction, or metric
creation. The histology branch and downstream quantification sketches remain
conceptual workflow illustrations.

All text uses embedded Arial at measured baselines; sizes are bounded by the
original text footprint. Images and dense point clouds are rasterized at 300
dpi. Coordinate bars consist of 256 opaque vector bands. Paths and text remain
editable vectors. The negative region label is `Non-Peyer`.

The existing model blocks additionally display `SpatialTRACE-Graph` and
`SpatialTRACE-Image`. Their centered names are declared in `model_name_labels`
in the layout, with a fit check against the original boxes. Names wrap at the
hyphen and share 9.6-point Arial Bold with every architecture line in their
module. Text is centered, with the graph description moved down within its
original box to accommodate the two-line name. In panel D, the complete image-model
name and architecture description are vertically centered as one text group
within the green box, preserving the existing line spacing. Box paths, arrows
and scientific layers retain their positions. The registry
hash was refreshed after verifying that only the separately authorized main
Figure 3 display-source record changed; Figure 1 checkpoints and inputs did not.

Run through the complete-figure root runner, or prepare components separately:

```bash
python3 -B Figure_1/assembly/prepare.py --output-dir /tmp/figure1-assembly-review
```

The callable `prepare(output_dir, overwrite=False)` returns the shared compositor
contract. Outputs are `scientific_layers.pdf`, its numeric-input provenance,
`manual_background.pdf`, and `manual_foreground.pdf`; the standalone CLI also
writes `preparation.json`. Existing outputs require explicit `--overwrite`.
Root composition creates the complete PDF, PNG, independent anchor/font/text
checks, and visual overlay diagnostics in a new review run. Original assembled
PDFs under SATA `Figures/` are never overwritten.
