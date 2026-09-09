"""Compatibility import for the verified release downloader."""
from pathlib import Path
from .artifacts import download, registry

DEFAULT_MODEL_NAME = "xenium"
MODEL_REGISTRY = registry()
ALIASES = {"paired-representation-shared-scale-v2": "xenium"}


def download_weights(model_name: str, output_dir: Path) -> Path:
    return download(ALIASES.get(model_name, model_name), output_dir)
