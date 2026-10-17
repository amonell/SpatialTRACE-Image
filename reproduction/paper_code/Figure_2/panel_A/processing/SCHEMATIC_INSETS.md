# Numerical schematic insets

`make_panel_A_schematic_insets.py` renders three actual, independently exported
components: `panel_A_sparse_coordinate_example`,
`panel_A_dense_coordinate_example`, and `panel_A_peyer_classifier_example`.
Each has PDF and PNG exports. The component runner and fixed-layout assembly
invoke the same script; no plotted thumbnail is copied from another figure.

The two coordinate examples use `day90_zoom_cells.parquet`,
`sparse_labels.parquet`, and the pinned `graph_component_inputs_manifest.json`
under SATA `Figure_2/stored_outputs/shared_graph_components/`. Required columns
are `x`, `y`, `crypt_villus_prediction`, `crypt_villus_label`, `batch`, and
`training_villus_id`, distributed across the two tables. The frozen manifest
supplies the exact bounds and two manually annotated villus IDs.

The Peyer example uses `panel_c_window_cells.parquet` and the pinned
`prepared_inputs_manifest.json` under SATA
`Figure_2_Extended_2/stored_outputs/corrected_labels_v1/`. Required columns are
`x_spatial`, `y_spatial`, `gat_peyer_probability`, and `gat_peyer_class`.
The displayed calls belong to the separate corrected-label BinaryGAT and are
checked against its saved validation-selected threshold. There is no fitting,
checkpoint selection, or inference during rendering.

The ordinary component destination is SATA `Figure_2/panel_A/`. Assembly uses
the same fresh files in its run-specific output directory and positions the
five image/colorbar viewports using independently measured original PDF image
transforms in `assembly/source_registration.json`.
