# Figure 2e frozen preprocessing

`build_section_disjoint_scaling_manifest.py` created and hash-locked the eight-fold
section-disjoint scaling design before the new models were fitted. Its authoritative
output is:

```text
/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/splits/graph_model_section_disjoint_villus_scaling_v1.json
SHA256 e1779c3caab90164eea05ba38d2dc93c3432ad845711fd7428f281bd52eaa0f8
```

The heavy, resumable fitting/evaluation entry point is
`frozen_preprocessing/graph_coordinate_model/train_section_disjoint_villus_scaling.py`.
It writes 160 independently selected checkpoints and their held-out-section
predictions to the SATA `Figure2/section_disjoint_villus_scaling_v1/` directory.
It is not invoked by any panel renderer.

The superseded 2026-09-02 held-out-villus input builder was moved to
`lineage_archive/final_figure_organization_2026-09-04/inactive_extras/Figure_2/panel_E/processing/build_panel_E_inputs.py`.
It is not part of the active Figure 2e workflow.
