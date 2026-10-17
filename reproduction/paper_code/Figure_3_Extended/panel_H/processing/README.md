Panel H reads the hash-locked representation-classifier comparison from SATA
`frozen_preprocessing/peyer_representation_v1/vit/sparse_evaluation_completed.json`.
The four models cross manual/GAT-soft targets with frozen-encoder/full-model SFT.
Every model uses the axis ViT's shared scale-aware backbone, central readout,
fine CNN and representation-pretrained EMA teacher. Newly initialized fine and
classification layers train in every condition.

All variants retain 280 training, 486 validation and 819 test cells (91 positives).
Twenty epochs and validation-AP checkpoint selection are fixed before test
evaluation. Full-model/manual-target SFT is prespecified for the maps; it is not
selected by the test ROC curves. Hard calls retain threshold 0.5.

ROC coordinates and verified AUROC/AP values are saved by explicit preprocessing
under SATA `frozen_preprocessing/peyer_representation_v1/roc/` as `curves.tsv`,
`metrics.tsv` and `manifest.json`. Their hashes resolve through the home
project's `frozen_preprocessing/peyer_representation_v1/active_release.json`.

Rendering reads these tables without fitting, inference, selection, metric
calculation or ROC construction. The optional `prepare_frozen_roc_curves.py`
provides a separate deterministic conversion into a new output directory; it
is never called by rendering and never replaces the active frozen tables.

Use `update_final_figure.py Figure_3_Extended --overwrite` for the final layout.
Run regression checks with
`python -B -m unittest Figure_3_Extended.panel_H.processing.test_frozen_roc -v`.

Previous classifier code, ROC tables, checkpoints and predictions remain
recoverable; the old reconstruction classifier is not a rendering fallback.
