# Peyer’s patch classification

The released classifier starts from representation pretraining and retains the shared transformer, scale embeddings, centered readouts, fine CNN and fused head. A binary logit replaces the coordinate outputs. It is a separate model, not a threshold on coordinates.

```bash
uv run --locked --extra cpu tissuemapper-image download --model peyer --from-dir /path/to/release-weights --output-dir weights
uv run --locked --extra cpu tissuemapper-image predict-cells --source-manifest sources.csv --cells-csv cells.csv --checkpoint weights/tissuemapper-image-peyer-v1.pt --output-dir runs/peyer --device cpu
```

The released checkpoint writes predicted_peyer_probability, predicted_peyer_logit and predicted_peyer_class. The fixed threshold is 0.5. Display colorbars do not change predictions. Use --prediction-column to choose another probability-column name; newly trained classifiers default to peyer_probability.

To train the matched architecture:

```bash
uv run --locked --extra cpu tissuemapper-image train --source-manifest sources.csv --supervised-manifest labels.csv --task-type binary_classification --target-column peyer_label --pretrained-checkpoint weights/tissuemapper-image-representation-v1.pt --output-dir runs/peyer_training --epochs 20 --device cpu
```

Provide finite [0, 1] targets, explicit train/validation splits, and both classes among training labels. Binary cross entropy fits the head; optional class weights come from training labels only. Validation average precision selects weights. Probability calibration on a new domain requires independent assessment.
