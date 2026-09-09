# Pretraining and fine-tuning

The default supervised CLI uses the released shared local/context transformer, retained scale embeddings, centered token readouts, fine CNN and concatenation head. The older pretrain command is a reconstruction baseline. Production representation pretraining uses pretrain-representation.

## Fine-tune the representation

Provide a source manifest and supervised CSV with inference columns, target_axis, epithelial_distance_clipped_1p0, and split. Use train/validation/test labels and hold out complete sections where possible. Test rows are excluded from optimization and selection. Missing targets are never copied from another coordinate.

```bash
uv run --locked --extra cpu tissuemapper-image download --model representation --from-dir /path/to/release-weights --output-dir weights
uv run --locked --extra cpu tissuemapper-image train --source-manifest sources.csv --supervised-manifest labels.csv --pretrained-checkpoint weights/tissuemapper-image-representation-v1.pt --output-dir runs/fine_tuning --epochs 20 --device cpu
```

For adaptation from complete coordinate weights, use --initial-checkpoint weights/tissuemapper-image-xenium-v1.pt instead. Configurations must match exactly; do not provide both initialization options. Encoder loading is strict, including scale embeddings.

Training uses AdamW and Smooth L1 coordinate losses. Validation loss selects the checkpoint. Both final and best checkpoint files contain selected weights; history records all epochs. --freeze-mode heads freezes image encoders; last1 also trains the final transformer block and normalization; none trains all parameters.

Random cell-level validation is available explicitly with --validation-fraction, but can overlap crops. Use it for development, not section-held-out evaluation. Explicit splits must contain training and validation rows; test rows are never reassigned.

## Train a representation on your images

Supply prespecified centers excluding downstream test sections. Preparation stores matched local/context arrays without fitting.

```bash
uv run --locked --extra cpu tissuemapper-image prepare-pretraining --source-manifest sources.csv --cells-csv pretraining_centers.csv --output-dir runs/prepared
uv run --locked --extra cpu tissuemapper-image pretrain-representation --prepared-metadata runs/prepared/metadata.json --output-dir runs/representation --epochs 35 --device cpu
```

A masked student predicts stop-gradient features from an exponential-moving-average teacher, using within-scale and cross-scale losses with variance/covariance regularization. The teacher encoder initializes supervised local/context encoding. The fine CNN is learned during fine-tuning. Pretraining validation is section-grouped and requires at least two sections.

Own-data preparation uses the direct-crop protocol. Exact manuscript reproduction requires frozen prepared arrays and configuration, including the original augmentation lineage. Generic training does not reconstruct the paper dataset.

See [Peyer classification](peyer_patch_workflow.md) for the matched binary head.
