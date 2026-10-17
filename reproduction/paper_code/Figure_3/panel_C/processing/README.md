Panel C reads SATA `Figure_3/stored_outputs/training_matched_inputs_v1/`:
`predictions_current_references.parquet`, `metrics.json` and `manifest.json`.
The explicit frozen pipeline in `frozen_preprocessing/figure3_training_inputs_v1/`
regenerates model outputs using training-matched image preparation for the same
50,000 cells and current graph-reference values. The renderer verifies input
hashes and reads saved metrics; it does not perform inference or calculate
metrics. The release SHA256 is pinned in `Figure_3/assembly/input_lock.json`.
Do not substitute the old reference columns. Figure-level README
and `run_all_panels.py --dry-run` provide exact paths, hashes and commands.
