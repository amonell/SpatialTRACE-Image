# TissueMapper-Image

Map intestinal anatomy from DAPI images and cell centroids. The released multiscale model predicts normalized crypt–villus position and epithelial distance. A separate Peyer’s patch classifier uses the same representation-pretrained shared-scale architecture.

## Install with uv

Clone the repository first (GitHub access is required while it is private):

```bash
gh repo clone amonell/TissueMapper-Image
cd TissueMapper-Image
```

From this checkout, uv installs Python 3.12 and the locked dependencies:

```bash
uv sync --locked --extra cpu --extra dev
uv run --locked --extra cpu tissuemapper-image --help
```

For Linux with an NVIDIA GPU, replace `--extra cpu` with `--extra cu126` in installation and subsequent commands. Do not select both extras.

## Self-contained example

This creates synthetic microscopy with three disjoint sections, trains a small model, and predicts on the test section. It tests execution, not anatomical accuracy.

```bash
uv run --locked --extra cpu tissuemapper-image create-demo --output-dir runs/demo
uv run --locked --extra cpu tissuemapper-image train --source-manifest runs/demo/sources.csv --supervised-manifest runs/demo/labels.csv --output-dir runs/demo/model --epochs 2 --device cpu --input-size-px 64 --local-crop-px 64 --context-crop-px 128 --fine-crop-px 32 --fine-input-size-px 32 --embed-dim 64 --depth 1 --num-heads 4
uv run --locked --extra cpu tissuemapper-image predict-cells --source-manifest runs/demo/sources.csv --cells-csv runs/demo/cells.csv --checkpoint runs/demo/model/crypt_villus_vit_model.pt --output-dir runs/demo/predictions --device cpu
uv run --locked --extra cpu tissuemapper-image export-qupath --predictions runs/demo/predictions/predictions.csv --output runs/demo/predictions/predictions.geojson
```

Use new output directories for each run. The final checkpoint contains validation-selected weights, not the last epoch.

## Trained models on your images

```bash
uv run --locked --extra cpu tissuemapper-image models
uv run --locked --extra cpu tissuemapper-image download --model xenium --from-dir /path/to/release-weights --output-dir weights
uv run --locked --extra cpu tissuemapper-image predict-cells --source-manifest sources.csv --cells-csv cells.csv --checkpoint weights/tissuemapper-image-xenium-v1.pt --output-dir runs/my_predictions --device cpu
```

Artifact IDs: `xenium` (coordinates), `if` (IF-adapted coordinates), `peyer` (classifier), `representation` (EMA teacher for fine-tuning). All carry fixed SHA256 identities and load with tensor-safe deserialization. The checkpoints are attached to the private `v1.0.0rc1` GitHub prerelease. Download them with an authenticated GitHub CLI, then use `--from-dir release-weights` in the command above:

```bash
gh release download v1.0.0rc1 --repo amonell/TissueMapper-Image --pattern '*.pt' --pattern checksums.json --pattern LICENSE --dir release-weights
```

Private assets require repository access; anonymous public download URLs remain unset. See [Artifacts](docs/artifacts.md) for checksum verification.

Supply DAPI microscopy, its true pixel size, and full-resolution cell centroids. Crop preparation is part of the model: preserve pyramid selection, per-crop normalization, resize and uint8 conversion. The model does not segment cells.

## Documentation and tests

- [Installation](docs/installation.md)
- [Apply to your data](docs/apply_to_own_data.md)
- [Pretraining and fine-tuning](docs/training.md)
- [Peyer classification](docs/peyer_patch_workflow.md)
- [QuPath export](docs/qupath_export.md)
- [Model card](MODEL_CARD.md)
- [Artifacts](docs/artifacts.md)
- [Paper figures](reproduction/README.md)
- [Release status](RELEASE_STATUS.md)
- [Executed validation](VALIDATION.md)

```bash
uv run --locked --extra cpu --extra dev pytest -q
```

Python imports remain `crypt_villus_vit` for compatibility. The repository, distribution and preferred CLI use TissueMapper-Image. Code and the four designated pretrained checkpoints are licensed under GPL-3.0-only; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Microscopy and figure-data terms are separate.
