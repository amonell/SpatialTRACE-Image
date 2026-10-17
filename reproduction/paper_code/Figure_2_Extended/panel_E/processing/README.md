# Frozen graph-component comparison

Panel E uses the completed component-only release under SATA
`frozen_preprocessing/model_design_comparisons_v1/graph/release_components/manifest.json`.
Its `outputs.component_metrics` record hash-locks the numeric table. Input
paths/hashes, validation-based checkpoint selection and the original eight
section splits are recorded by the upstream `aggregate_graph.py` pipeline.
No panel-specific preparation, fitting, graph randomization, checkpoint
selection, or aggregation is performed during plotting.

The active renderer wraps the shared graph-comparison implementation in
`Figure_2_Extended_3/panel_A/figure_script/`; both standalone and assembled
versions therefore use identical model ordering, section pairs and values.

```bash
Figure2/.venv/bin/python -B Figure_2_Extended/run_all_panels.py \
  --panels E \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2_Extended \
  --overwrite
python -B update_final_figure.py Figure_2_Extended --overwrite
```

Outputs are `panel_E_graph_component_comparison.pdf`, its PNG/provenance,
and the complete-page `panel_E_final` layer. The upstream scVI representation
was jointly learned across sections; these comparisons do not remove that
feature-lineage limitation. Randomized connections are fixed within-section
degree-preserving permutations applied before separate model training.
