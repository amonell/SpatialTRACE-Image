# TissueMapper-Image

## Model

Local and context image crops share a vision transformer. A separate CNN processes the fine crop. Their features are combined to learn anatomical coordinates from annotations, with outputs from 0 to 1. A binary head can instead learn region labels.

The transformer uses 16-pixel patches, 256-dimensional embeddings, six blocks, eight attention heads, and an MLP expansion factor of four. Positional and scale embeddings are retained during fine-tuning. Each scale averages the central 4 × 4 token features.

The paper developed and evaluated the models using intestinal tissue. The pretrained coordinate models predict crypt–villus position and epithelial distance from DAPI images. The Peyer’s patch classifier uses a binary head and a threshold of 0.5.

## Pretraining

A masked student predicts teacher features within and across scales. The teacher is updated as an exponential moving average of the student’s weights. Gradients pass through the student; teacher targets stay fixed during each update. Variance and covariance losses help prevent collapsed representations.

The teacher initializes the local and context branches for fine-tuning. The fine CNN is trained during fine-tuning.

## Image preparation

Inputs are DAPI images and full-resolution cell centroids.

| Crop | Width at 0.325 µm per pixel | Model input |
| --- | --- | --- |
| Local | 512 pixels | 256 × 256 pixels |
| Context | 2,048 pixels | 256 × 256 pixels |
| Fine | 128 pixels | 128 × 128 pixels |

Crop widths are scaled to each image’s pixel size. Processing uses TIFF pyramid levels, centered crops, and zero padding. Each crop is normalized using its 1st and 99.8th percentiles, resized bilinearly, and converted to uint8.

JPEG2000 decoders can produce small pixel differences. Paper figures use saved prepared inputs.

## Evaluation and use

The Xenium model was trained against graph-derived coordinates. Figure 3 shows agreement with those coordinates in `sample_008`, a section also used for architecture and readout diagnostics. IF evaluation used spatially disjoint regions of two sections. Treatment comparisons are descriptive because there was one section per condition.

Cell detection is a separate preprocessing step. Predictions provide coordinates or class probabilities, without uncertainty estimates. Validate them on new stains, microscopes, and tissues. This software is intended for research.
