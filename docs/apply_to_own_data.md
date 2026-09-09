# Apply to your images

Image paths resolve relative to the source CSV directory, unless absolute.

sources.csv:

```csv
source_id,image_path,section_id,pixel_size_um,channel_index
sample_a,images/sample_a.ome.tif,section_a,0.2125,0
```

cells.csv:

```csv
cell_id,source_id,centroid_x_fullres_px,centroid_y_fullres_px
cell_001,sample_a,1000.5,2000.5
```

Centroids use full-resolution pixels (x horizontal, y vertical). Supply the true micrometers-per-pixel value and DAPI channel. Convert micron coordinates first; do not rescale twice. Cell detection/segmentation is an upstream step.

Inputs: 2D NumPy arrays, grayscale PNG, and TIFF/OME-TIFF with explicit spatial axes. TIFF reads are crop-bounded and preserve pyramid levels. A named Z axis is max-projected. Export one timepoint/scene before analysis. Convert proprietary VSI to a calibrated DAPI OME-TIFF.

```bash
uv run --locked --extra cpu tissuemapper-image predict-cells --source-manifest sources.csv --cells-csv cells.csv --checkpoint weights/tissuemapper-image-if-v1.pt --output-dir runs/if_predictions --device cpu
```

Released models use nominal local/context/fine widths 512/2048/128 at 0.325 µm per reference pixel. Native widths are rounded after physical scaling. Local/context inputs become 256 × 256; fine inputs 128 × 128. Each crop uses its own 1st/99.8th-percentile bounds, bilinear resize and uint8 roundtrip. Zero padding precedes normalization. Whole-slide normalization is not equivalent.

Coordinate outputs include per-cell predictions, predefined gates, fractions and a summary. Gates require biological validation under domain shift; predictions are not uncertainty estimates. The separate classifier exports probabilities/logits/classes without coordinate gates.

Native OpenJPEG and original Java JPEG2000 decoding can differ slightly. Acceptance tests measure pixel and prediction differences; software parity does not establish accuracy on a new specimen. Use frozen prepared arrays for exact paper-input reproduction.

Add one source row per slide for multiple tissues. See [QuPath](qupath_export.md) for named measurement-column conversion.
