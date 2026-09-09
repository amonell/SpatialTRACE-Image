from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tifffile


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    image_path: Path
    condition: str = ""
    section_id: str = ""
    pixel_size_um: float | None = None
    channel_index: int = 0


def _resolve_path(value: object, *, base_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def load_source_manifest(source_manifest_path: Path) -> dict[str, SourceSpec]:
    path = Path(source_manifest_path)
    table = pd.read_csv(path)
    required = {"source_id", "image_path"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Source manifest is missing required column(s): {missing}")
    sources: dict[str, SourceSpec] = {}
    for row in table.to_dict(orient="records"):
        source_id = str(row["source_id"])
        if source_id in sources:
            raise ValueError(f"Duplicate source_id `{source_id}` in {path}")
        image_path = _resolve_path(row["image_path"], base_dir=path.parent)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path for source `{source_id}` does not exist: {image_path}")
        pixel_size = row.get("pixel_size_um")
        sources[source_id] = SourceSpec(
            source_id=source_id,
            image_path=image_path,
            condition="" if pd.isna(row.get("condition", "")) else str(row.get("condition", "")),
            section_id="" if pd.isna(row.get("section_id", "")) else str(row.get("section_id", "")),
            pixel_size_um=None if pixel_size is None or pd.isna(pixel_size) else float(pixel_size),
            channel_index=int(row.get("channel_index", 0) or 0),
        )
    return sources


def load_grayscale_image(
    path: Path,
    *,
    channel_index: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        image = np.load(path, mmap_mode="r")
    elif suffix in {".tif", ".tiff", ".ome.tif", ".ome.tiff"}:
        image = tifffile.imread(path)
    else:
        image = np.asarray(Image.open(path))
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[-1] in {3, 4}:
            image = image[..., int(channel_index)]
        else:
            image = image[int(channel_index)]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image after channel selection, got shape {image.shape}")
    if not normalize:
        return image
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.all():
        image = np.where(finite, image, 0.0)
    lo, hi = np.percentile(image, [1.0, 99.5])
    if hi <= lo:
        hi = float(image.max()) if float(image.max()) > lo else lo + 1.0
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image.astype(np.float32)
