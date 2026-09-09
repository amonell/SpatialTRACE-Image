# QuPath input and output

The importer accepts CSV/TSV centroid columns in pixels (centroid_x_fullres_px / centroid_y_fullres_px and common aliases) or micrometers (centroid_x_um / centroid_y_um and QuPath measurement names). Micron coordinates require pixel_size_um in the source manifest or --pixel-size-um.

```bash
uv run --locked --extra cpu tissuemapper-image predict-qupath --source-manifest sources.csv --qupath-csv detections.csv --checkpoint weights/tissuemapper-image-if-v1.pt --output-dir runs/qupath --device cpu
uv run --locked --extra cpu tissuemapper-image export-qupath --predictions runs/qupath/predictions.csv --output runs/qupath/predictions.geojson
```

For multiple source images, specify --source-id. Optional class filtering uses --class-filter with --class-match-mode exact, contains or regex. The import summary records recognized columns and selected rows.

GeoJSON contains per-cell points, coordinate measurements and predicted anatomical gate classes for coordinate models. Import the file as QuPath objects. Confirm that image calibration and coordinate origin match the source microscopy before interpreting overlays.
