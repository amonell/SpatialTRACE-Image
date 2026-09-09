# TissueMapper-Image model card

The released coordinate architecture is paired-representation-shared-scale-v2. Local and context DAPI crops share one transformer: 16-pixel patches, 256-dimensional embeddings, six blocks, eight attention heads and MLP expansion four. Learned positional and scale embeddings are retained during fine-tuning. Centered 4× token readouts summarize each scale. A separate fine CNN supplies the third representation; concatenation and a shared MLP feed two sigmoid coordinate heads.

The representation checkpoint exports the exponential-moving-average teacher from paired-scale masked feature pretraining. Student losses predict teacher features within and across scales, with variance/covariance regularization. Stop-gradient targets prevent gradients through the teacher. The fine branch enters during supervised fine-tuning.

Artifacts include the Xenium coordinate model, the final corrected IF adaptation, the representation teacher and the representation-pretrained Peyer classifier. The classifier keeps the coordinate backbone and replaces coordinate heads with a binary head; its threshold is 0.5. Safe release exports preserve every parameter tensor.

## Input contract

Use DAPI and full-resolution centroids. Nominal local/context/fine crop widths are 512/2048/128 reference pixels at 0.325 µm per pixel. Native widths scale with source calibration; inputs are 256/256/128 pixels. TIFF pyramid selection, centered zero-padded crops, per-crop percentile normalization, direct bilinear resize and uint8 quantization are fixed parts of production inference.

OpenJPEG decoding may differ by one native intensity count from the original Java JPEG2000 reader. Golden checks distinguish exact input identity from tolerance-level downstream equivalence. Exact figure reproduction uses frozen prepared data.

## Scope

Graph-derived coordinates supervise the Xenium image model; these are not direct manual coordinate labels. Main Figure 3 uses the fixed sample_008 field, previously used for architecture/readout diagnostics. It illustrates graph-to-image agreement, not an untouched independent-specimen estimate. IF performance uses spatially disjoint regions of two sections. One section per treatment condition makes treatment comparisons descriptive.

The model does not segment cells or estimate uncertainty. New stains, microscopes, anatomy or image calibration can cause domain shift. Validate predictions and gate interpretation before drawing biological conclusions. Research use, not clinical diagnosis.

See packaged artifacts.json for immutable release/source checkpoint hashes. The original internal checkpoint format strings are retained for backward compatibility.
