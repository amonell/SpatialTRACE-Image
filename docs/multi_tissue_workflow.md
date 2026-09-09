# Multiple tissues

Add one row per image to `sources.csv`. Each row needs a source ID, section ID, pixel size, and DAPI channel.

Give every cell a unique ID and link it to an image through `source_id`. The predictor processes cells from each image together.

For training, assign sections to training, validation, and test sets before fitting.

See [image preparation](apply_to_own_data.md) and [training](training.md).
