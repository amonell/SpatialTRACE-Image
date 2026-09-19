# Train on example images

This example creates synthetic DAPI images, trains a small coordinate model, and predicts.

Run these commands from the repository:

```bash
uv run --locked --extra cpu spatialtrace-image create-demo \
  --output-dir runs/demo

uv run --locked --extra cpu spatialtrace-image train \
  --source-manifest runs/demo/sources.csv \
  --supervised-manifest runs/demo/labels.csv \
  --output-dir runs/demo/model --epochs 2 --device cpu \
  --input-size-px 64 --local-crop-px 64 --context-crop-px 128 \
  --fine-crop-px 32 --fine-input-size-px 32 \
  --embed-dim 64 --depth 1 --num-heads 4

uv run --locked --extra cpu spatialtrace-image predict-cells \
  --source-manifest runs/demo/sources.csv --cells-csv runs/demo/cells.csv \
  --checkpoint runs/demo/model/crypt_villus_vit_model.pt \
  --output-dir runs/demo/predictions --device cpu

uv run --locked --extra cpu spatialtrace-image export-qupath \
  --predictions runs/demo/predictions/predictions.csv \
  --output runs/demo/predictions/predictions.geojson
```

Use a new output directory when repeating the example.

## Your images

Start with [sources.csv](templates/sources.csv) and [cells.csv](templates/cells.csv). Replace the example rows with your image paths and cell centroids. Paths are relative to the source table; centroids are in full-resolution pixels.

See the [input guide](../docs/apply_to_own_data.md) for image formats and calibration.
