Panel B uses the repaired IF release. Its replacement illustrative field
was locked using geometry and reference coverage before repaired test metrics
were computed. It contains 749 retained training cells from `SI1_villus_000`,
`SI1_villus_003`, and `SI1_villus_005`, at full-resolution
x=16540:22092, y=48600:52649. Strict-test labels are never displayed as training
labels. The old 821-cell example is not a valid depiction of the repaired
training split and must not be copied into its replacement.

The exact cells, source image, bounds, display normalization and selection rule
are recorded in the SATA
`Figure_4/stored_outputs/scientific_repair_v1/illustration_inputs/manifest.json`.
The prepared arrays and cell table are frozen plotting inputs.

The horizontal 200-µm scale bar is a vector line/text in an external footer,
not on top of microscopy or sparse labels. Its length is calculated from the
frozen IF source-pixel calibration of 0.16247698804244626 µm/px, yielding
1230.943547 px at full resolution. The calibration is read from the repaired
configuration and hash-verified through the release manifest, not an old default
YAML. Pass all inputs explicitly using the command in `Figure_4/run_all_panels.py`.
The script refuses to render until a completed, hash-verified repaired release
exists. The numerical retry completed four full-stage epochs using standard
math attention and validation selected epoch 2; incomplete failed attempts are
preserved separately and are not counted as completed epochs.

The provenance JSON records the calibration, source hashes, full-resolution
bounds, physical field of view, exact scale-bar endpoints, and normalization.
No inference, fitting, or label selection is performed.
