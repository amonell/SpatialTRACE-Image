Panel B reads the frozen 30,000-cell `sample_008` UMAP coordinates and aligned
production-v2 embedding metadata. It does not recompute the embedding or UMAP.

Use SATA `Figure_3/stored_outputs/training_matched_inputs_v1/` for
`embedding_metadata_current_references.csv` and `figure3f_umap.csv`.
Its adjacent `manifest.json` is SHA256-locked in `Figure_3/assembly/input_lock.json`.
The explicit pipeline in `frozen_preprocessing/figure3_training_inputs_v1/`
recomputes training-matched embeddings and fits UMAP once using the original
settings. The 30,000-cell subset and row order are unchanged. The source image is
`D8_SI_R2`; the historical `sample_008` identifier is an alias, not a new split.
