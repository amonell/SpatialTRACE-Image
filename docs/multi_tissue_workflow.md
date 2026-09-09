# Multiple tissues

Use one source-manifest row per microscopy image, with distinct source/section IDs and the correct pixel size and DAPI channel for each image. Cell IDs must remain unique across sources. The predictor groups batches by source to avoid repeatedly opening slides.

See [input preparation](apply_to_own_data.md) and [training splits](training.md). Select training and evaluation regions before fitting; source-grouped batching is an I/O optimization, not a validation design.
