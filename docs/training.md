# Pretraining and fine-tuning

The image model uses local, context, and fine crops. Local and context crops share a transformer with scale embeddings. A small CNN processes the fine crop.

## Fine-tune a pretrained model

Define the coordinates you want to learn for your tissue and scale their annotations from 0 to 1. The default column names below come from the intestinal example. Use `--target-axis-column` and `--target-epithelial-column` to select your own coordinate columns.

For new coordinates, define gates for your tissue. The built-in gate labels follow the intestinal example.

Prepare a label CSV with the cell columns from the [input guide](apply_to_own_data.md), plus:

- `target_axis`: crypt–villus position from 0 to 1.
- `epithelial_distance_clipped_1p0`: epithelial distance from 0 to 1.
- `split`: `train`, `validation`, or `test`.

Hold out complete sections where possible. Training and validation rows are required; test rows are reserved for evaluation.

[Download the representation weights](artifacts.md), then run:

```bash
uv run --locked --extra cpu spatialtrace-image train \
  --source-manifest sources.csv --supervised-manifest labels.csv \
  --pretrained-checkpoint weights/spatialtrace-image-representation-v1.pt \
  --output-dir runs/fine_tuning --epochs 20 --device cpu
```

To adapt the trained coordinate model instead, replace `--pretrained-checkpoint` with `--initial-checkpoint weights/spatialtrace-image-xenium-v1.pt`.

The architecture must match the checkpoint. Scale embeddings are retained during fine-tuning.

## Prepare crops once

For repeated training runs, save the crops first. Training then reads these
shards without decoding the original microscopy images each epoch.

```bash
uv run --locked --extra cpu spatialtrace-image prepare-supervised \
  --source-manifest sources.csv --cells-csv labels.csv \
  --output-dir runs/supervised_crops --num-workers 4

uv run --locked --extra cpu spatialtrace-image train \
  --source-manifest sources.csv \
  --supervised-manifest runs/supervised_crops/manifest.csv \
  --prepared-supervised-metadata runs/supervised_crops/metadata.json \
  --pretrained-checkpoint weights/spatialtrace-image-representation-v1.pt \
  --output-dir runs/fine_tuning_prepared --epochs 20 --device cpu
```

Preparation keeps all input rows, labels, and split assignments. Use the generated
`manifest.csv` for training: its `prepared_index` links each cell to its crops.
Test rows remain test rows; preparing their crops does not fit a model.

Both preparation commands use cached image tiles and four CPU workers by default.
They read nearby cells together, then save them in the original input order.
Crop normalization, resizing, and rounding are unchanged. The optional Rust
backend uses `--crop-backend rust` after installing the `rust` extra.

Preparation needs no GPU. Each worker has a 256 MiB tile cache, plus decoding and
batch memory. Set `--num-workers 0` for serial preparation or
`--tile-cache-mib 64` for a smaller cache. Large machines can try eight workers.
`--batch-size` here controls preparation batches, not training batches.

At the default crop sizes, 50,000 cells need about 7.4 GB for supervised shards,
or 6.6 GB for pretraining pairs. Allow additional free space for training outputs.
The output directory must be empty; interrupted preparation leaves partial files
and must be restarted in a new directory. `metadata.json` is written only after
all shards are complete. The adjacent provenance file records their checksums.

Keep crop dimensions and `--reference-pixel-size-um` matched between preparation
and training. New shard metadata is checked before supervised training starts.
Physical pixel size for each source image belongs in `sources.csv`.

Timing commands and measured results are in `benchmarks/README.md` in the repository.

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
uv run --locked --extra cpu spatialtrace-image prepare-pretraining \
  --source-manifest sources.csv --cells-csv pretraining_centers.csv \
  --output-dir runs/prepared --num-workers 4

uv run --locked --extra cpu spatialtrace-image pretrain-representation \
  --prepared-metadata runs/prepared/metadata.json \
  --output-dir runs/representation --epochs 35 --device cpu
```

A masked student predicts features from a teacher. The teacher is updated as an exponential moving average of the student’s weights. Its targets are held fixed during each gradient update. Losses compare features within and across scales; variance and covariance terms help prevent collapsed representations.

The teacher initializes the local and context branches for fine-tuning. The fine CNN is trained during fine-tuning.

Use `pretrain-representation` for this workflow. The older `pretrain` command runs a reconstruction baseline.

For region classification, use `--task-type binary_classification` and set `--target-column` to your region labels. The [Peyer’s patch example](peyer_patch_workflow.md) shows this workflow.
