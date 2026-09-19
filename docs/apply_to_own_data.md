# Prepare your images

You need DAPI images, image calibration, and cell centroids.

Create `sources.csv`:

```csv
source_id,image_path,section_id,pixel_size_um,channel_index
sample_a,images/sample_a.ome.tif,section_a,0.2125,0
```

Create `cells.csv`:

```csv
cell_id,source_id,centroid_x_fullres_px,centroid_y_fullres_px
cell_001,sample_a,1000.5,2000.5
```

Image paths are relative to `sources.csv`. Centroids use full-resolution pixels: x is horizontal and y is vertical. If your coordinates are in micrometers, divide them by `pixel_size_um` first.

Set `channel_index` to the DAPI channel. Use the image’s measured pixel size.

## Image formats

Supported inputs are 2D NumPy arrays, grayscale PNG, and TIFF or OME-TIFF with spatial-axis metadata. TIFF crops use the image pyramid when available. Z-stacks are max-projected.

Export one timepoint and scene per image. Convert VSI files to calibrated DAPI OME-TIFF first.

## Predict

For an IF image:

```bash
uv run --locked --extra cpu spatialtrace-image predict-cells \
  --source-manifest sources.csv --cells-csv cells.csv \
  --checkpoint weights/spatialtrace-image-if-v1.pt \
  --output-dir runs/if_predictions --device cpu
```

The output includes per-cell coordinates, anatomical gates, and gate fractions. Check the predictions against your tissue anatomy before using the gates.

Crop extraction, normalization, and resizing are handled by the model. Supply the original calibrated DAPI image. The [model card](model_card.md) lists the preprocessing settings.

For multiple images, add one source row per image and keep cell IDs unique. See [QuPath](qupath_export.md) for importing detection tables.
