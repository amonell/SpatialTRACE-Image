# Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). In the repository:

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu tissuemapper-image --help
```

uv installs Python 3.12 and the versions listed in `uv.lock`.

## NVIDIA GPUs

On Linux, replace `--extra cpu` with `--extra cu126`. Use `--device cuda` when training or predicting. Install a compatible NVIDIA driver first.

Keep the same CPU or GPU option in subsequent `uv run` commands. Linux CPU and NVIDIA GPU workflows have been tested.

## Optional packages

Add an extra to `uv sync` and `uv run` as needed:

- `--extra dev`: tests and package builds.
- `--extra docs`: documentation.
- `--extra figures`: paper figures.
- `--extra embedding`: UMAP.
