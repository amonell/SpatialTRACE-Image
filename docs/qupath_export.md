# QuPath

Export cell detections from QuPath as CSV or TSV.

The importer accepts pixel or micrometer centroids. For micrometer coordinates, set `pixel_size_um` in the source table or pass `--pixel-size-um`.

## Predict and export

```bash
uv run --locked --extra cpu tissuemapper-image predict-qupath \
  --source-manifest sources.csv --qupath-csv detections.csv \
  --checkpoint weights/tissuemapper-image-if-v1.pt \
  --output-dir runs/qupath --device cpu

uv run --locked --extra cpu tissuemapper-image export-qupath \
  --predictions runs/qupath/predictions.csv \
  --output runs/qupath/predictions.geojson
```

Import the GeoJSON into QuPath as objects. It contains cell points, predicted coordinates, and anatomical gate classes.

Check that the points align with the original detections.

## Select cells

If the source table contains several images, choose one with `--source-id`.

Use `--class-filter` to select a detection class. `--class-match-mode` accepts `exact`, `contains`, or `regex`. The import summary lists the selected rows and recognized columns.
