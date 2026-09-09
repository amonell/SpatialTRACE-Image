# TissueMapper-Image

Map crypt–villus position and epithelial distance from DAPI images. A separate model predicts Peyer’s patch probability.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
gh repo clone amonell/TissueMapper-Image
cd TissueMapper-Image
uv sync --locked --extra cpu
```

For an NVIDIA GPU on Linux, use `--extra cu126` instead of `--extra cpu`.

## Use a pretrained model

Download the Xenium coordinate model:

```bash
gh release download v1.0.0rc1 --repo amonell/TissueMapper-Image \
  --pattern tissuemapper-image-xenium-v1.pt --dir release-weights

uv run --locked --extra cpu tissuemapper-image download \
  --model xenium --from-dir release-weights --output-dir weights
```

Prepare a [source table and cell-centroid table](docs/apply_to_own_data.md), then predict:

```bash
uv run --locked --extra cpu tissuemapper-image predict-cells \
  --source-manifest sources.csv --cells-csv cells.csv \
  --checkpoint weights/tissuemapper-image-xenium-v1.pt \
  --output-dir runs/predictions --device cpu
```

Predictions are saved in `runs/predictions/predictions.csv`.

See [pretrained models](docs/artifacts.md) for the IF, Peyer’s patch, and pretraining weights. To try the software without your own images, run the [example](examples/README.md).

## More

- [Installation](docs/installation.md)
- [Pretraining and fine-tuning](docs/training.md)
- [QuPath](docs/qupath_export.md)
- [Model details](MODEL_CARD.md)
- [Reproduce the figures](reproduction/README.md)
- [Tests](VALIDATION.md)

Code and pretrained models: [GPL-3.0-only](LICENSE).
