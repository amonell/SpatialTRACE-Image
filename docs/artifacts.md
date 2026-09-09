# Pretrained models

| Model ID | Use |
| --- | --- |
| `xenium` | Coordinates in Xenium DAPI images |
| `if` | Coordinates in immunofluorescence DAPI images |
| `peyer` | Peyer’s patch probability |
| `representation` | Pretrained encoder for fine-tuning |

Download the weights:

```bash
gh release download v1.0.0rc1 --repo amonell/TissueMapper-Image \
  --pattern '*.pt' --pattern checksums.json --pattern LICENSE \
  --dir release-weights
```

Install the model you need:

```bash
uv run --locked --extra cpu tissuemapper-image download \
  --model xenium --from-dir release-weights --output-dir weights
```

Replace `xenium` with another model ID as needed. This command checks the file’s SHA256 checksum before installation. Model filenames and checksums are listed in `src/crypt_villus_vit/artifacts.json`.

Use a new download directory. To reuse a model already installed in `weights/`, pass its path directly to `predict-cells` or `train`.

The four models use GPL-3.0-only. Figure data have separate licensing terms.
