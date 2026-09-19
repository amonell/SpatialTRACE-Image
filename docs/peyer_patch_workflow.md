# Peyer’s patch classification

The classifier uses the same pretrained image encoder as the coordinate model, with a binary output head.

## Predict

[Download the Peyer’s patch weights](artifacts.md), then run:

```bash
uv run --locked --extra cpu spatialtrace-image predict-cells \
  --source-manifest sources.csv --cells-csv cells.csv \
  --checkpoint weights/spatialtrace-image-peyer-v1.pt \
  --output-dir runs/peyer --device cpu
```

The released model writes three columns:

- `predicted_peyer_probability`
- `predicted_peyer_logit`
- `predicted_peyer_class`, using a probability threshold of 0.5.

Check probability calibration on new tissue types or imaging conditions.

## Train

Add `peyer_label` and `split` columns to your cell table. Labels must be between 0 and 1, with both classes represented in training.

```bash
uv run --locked --extra cpu spatialtrace-image train \
  --source-manifest sources.csv --supervised-manifest labels.csv \
  --task-type binary_classification --target-column peyer_label \
  --pretrained-checkpoint weights/spatialtrace-image-representation-v1.pt \
  --output-dir runs/peyer_training --epochs 20 --device cpu
```

Training uses binary cross-entropy. Validation average precision selects the saved model. Optional class weights are calculated from training labels.

New classifiers use `peyer_probability` as the default probability column. Change it with `--prediction-column`.
