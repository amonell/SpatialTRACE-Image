# Installation

Install uv using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). From a TissueMapper-Image checkout:

```bash
uv sync --locked --extra cpu --extra dev
uv run --locked --extra cpu tissuemapper-image --help
```

The .python-version file selects Python 3.12. uv.lock pins the environment; --locked refuses unintended dependency resolution changes. Keep the same torch extra on subsequent uv run commands.

For Linux/NVIDIA, choose --extra cu126 instead of cpu and --device cuda in model commands. A compatible NVIDIA driver is required; uv installs PyTorch's runtime libraries. Mac and Windows GPU acceleration are not validated. CPU installation does not require CUDA.

Optional extras: dev (tests/build), docs (MkDocs). Graph adds expression for scVI workflows; installing it does not supply the missing trained reference encoder. Image adds embedding for optional UMAP use. Paper rendering does not refit UMAP.

See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for index/extra behavior. Never synchronize an unrelated active environment; run from this checkout.
