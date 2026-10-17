# Corrected-label compact inputs

`prepare_frozen_peyer_gat_panel_inputs.py` forwards to the frozen
`peyer_label_repair_v1/prepare_graph_panels.py` pipeline. Pass an explicit
`--lock`, `--original-prepared-manifest`,
`--expected-original-manifest-sha256`, and new `--output-dir`.
It does not fit a model, select a threshold or choose a window.

The active SATA inputs are under
`Figure_2_Extended_2/stored_outputs/corrected_labels_v1/`. They retain the original
background, 3,457-cell test window and numeric DAPI crop, and join corrected
labels/probabilities from the new BinaryGAT. Threshold 0.95 was selected using
validation F1; the displayed pooled performance includes only the two test
sections. The old preparer is archived in the scientific-repair original_sources.
