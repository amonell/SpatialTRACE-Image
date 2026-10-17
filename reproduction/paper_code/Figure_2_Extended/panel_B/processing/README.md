Panel B requires the completed matched runtime benchmark from
`frozen_preprocessing/scientific_repair_v1/benchmark_gat_runtime.py`.
It covers eight section-disjoint folds, ten training-villus counts (1–45),
and two coordinates. Every fitting interval includes validation/checkpoint I/O
but excludes data preparation and held-out inference. Benchmark checkpoints are
not scientific-model replacements. The renderer verifies `completed.json` and
its `runtime_records.tsv` hash, then shows median/IQR across eight folds.
Do not use historical approximate times or start any training from the renderer.
Exact source paths, fields and commands are in the figure README.
