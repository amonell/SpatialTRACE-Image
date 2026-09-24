# SpatialTRACE-Image

TRACE stands for Tissue Region and Axis Coordinate Estimation.

Map tissue organization from microscopy images. Pretrain on unlabeled images, then fine-tune with your own annotations to learn anatomical coordinates or identify tissue regions.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
gh repo clone amonell/SpatialTRACE-Image
cd SpatialTRACE-Image
uv sync --locked --extra cpu
```

For an NVIDIA GPU on Linux, use `--extra cu126` instead of `--extra cpu`.

## Train on your tissue

Define the coordinates or regions you want to map, then [pretrain and fine-tune](docs/training.md) on your images.

## Try the paper model

The paper uses intestinal tissue as an example. These pretrained weights predict crypt–villus position and epithelial distance from Xenium DAPI images:

```bash
gh release download v1.0.0rc2 --repo amonell/SpatialTRACE-Image \
  --pattern spatialtrace-image-xenium-v1.pt --dir release-weights

uv run --locked --extra cpu spatialtrace-image download \
  --model xenium --from-dir release-weights --output-dir weights
```

Prepare a [source table and cell-centroid table](docs/apply_to_own_data.md), then predict:

```bash
uv run --locked --extra cpu spatialtrace-image predict-cells \
  --source-manifest sources.csv --cells-csv cells.csv \
  --checkpoint weights/spatialtrace-image-xenium-v1.pt \
  --output-dir runs/predictions --device cpu
```

Predictions are saved in `runs/predictions/predictions.csv`.

See [pretrained models](docs/artifacts.md) for the IF, Peyer’s patch, and pretraining weights. To try the software without your own images, run the [example](examples/README.md).

## More

- [Installation](docs/installation.md)
- [Pretraining and fine-tuning](docs/training.md)
- [QuPath](docs/qupath_export.md)
- [Faster raw-image inference](docs/fast_inference.md)
- [Model details](MODEL_CARD.md)
- [Reproduce the figures](reproduction/README.md)
- [Tests](VALIDATION.md)

Code and pretrained models: [GPL-3.0-only](LICENSE).
