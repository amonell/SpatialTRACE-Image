# Pretraining and fine-tuning

The image model uses local, context, and fine crops. Local and context crops share a transformer with scale embeddings. A small CNN processes the fine crop.

## Fine-tune a pretrained model

Prepare a label CSV with the cell columns from the [input guide](apply_to_own_data.md), plus:

- `target_axis`: crypt–villus position from 0 to 1.
- `epithelial_distance_clipped_1p0`: epithelial distance from 0 to 1.
- `split`: `train`, `validation`, or `test`.

Hold out complete sections where possible. Training and validation rows are required; test rows are reserved for evaluation.

[Download the representation weights](artifacts.md), then run:

```bash
uv run --locked --extra cpu tissuemapper-image train \
  --source-manifest sources.csv --supervised-manifest labels.csv \
  --pretrained-checkpoint weights/tissuemapper-image-representation-v1.pt \
  --output-dir runs/fine_tuning --epochs 20 --device cpu
```

To adapt the trained coordinate model instead, replace `--pretrained-checkpoint` with `--initial-checkpoint weights/tissuemapper-image-xenium-v1.pt`.

The architecture must match the checkpoint. Scale embeddings are retained during fine-tuning.

## Training settings

Training uses AdamW and Smooth L1 coordinate losses. Validation loss selects the saved model. Both the best and final checkpoint files contain those selected weights.

Choose which layers to train with `--freeze-mode`:

- `heads`: output heads only.
- `last1`: heads, final transformer block, and normalization.
- `none`: all parameters.

Cell-level validation is available through `--validation-fraction`. Nearby cells can have overlapping crops, so section-level splits give a stronger test of generalization.

## Pretrain on your images

Choose cell centers from the pretraining sections, excluding downstream test sections. At least two sections are needed for pretraining and validation.

```bash
uv run --locked --extra cpu tissuemapper-image prepare-pretraining \
  --source-manifest sources.csv --cells-csv pretraining_centers.csv \
  --output-dir runs/prepared

uv run --locked --extra cpu tissuemapper-image pretrain-representation \
  --prepared-metadata runs/prepared/metadata.json \
  --output-dir runs/representation --epochs 35 --device cpu
```

A masked student predicts features from a teacher. The teacher is updated as an exponential moving average of the student’s weights. Its targets are held fixed during each gradient update. Losses compare features within and across scales; variance and covariance terms help prevent collapsed representations.

The teacher initializes the local and context branches for fine-tuning. The fine CNN is trained during fine-tuning.

Use `pretrain-representation` for this workflow. The older `pretrain` command runs a reconstruction baseline.

See [Peyer’s patch classification](peyer_patch_workflow.md) for training the binary model.
