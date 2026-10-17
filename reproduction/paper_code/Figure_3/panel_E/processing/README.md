Panel E reads all 50,000 locked `sample_008` production-v2 predictions. The full
map is not display-downsampled, and reference/prediction share color limits and
zoom.

Use SATA `Figure_3/stored_outputs/training_matched_inputs_v1/predictions_current_references.parquet`
and its adjacent manifest, pinned by SHA256 in `Figure_3/assembly/input_lock.json`.
The explicit pipeline `frozen_preprocessing/figure3_training_inputs_v1/`
regenerates image predictions using exact supervised crop preparation. Current
graph references, tissue coordinates, cell membership and zoom bounds stay fixed. Do not read
the outdated reference columns in the original production-sample table.
