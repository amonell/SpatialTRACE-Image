from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from crypt_villus_vit.augmentation import IntensityAugmentationConfig
from crypt_villus_vit.augmentation import apply_intensity_parameters
from crypt_villus_vit.augmentation import sample_intensity_parameters
from crypt_villus_vit.model import ModelConfig


@dataclass(frozen=True)
class PreparedSupervisedArrays:
    metadata_path: Path
    manifest_path: Path
    local_shard_paths: tuple[Path, ...]
    context_shard_paths: tuple[Path, ...]
    fine_shard_paths: tuple[Path, ...]
    rows_per_shard: int
    row_count: int
    image_dtype: str


@dataclass(frozen=True)
class PreparedPretrainingArrays:
    metadata_path: Path
    manifest_path: Path
    shard_paths: tuple[Path, ...]
    rows_per_shard: int
    row_count: int
    variant_count: int
    image_dtype: str


def _paths_from_metadata(values: list[str], *, base_dir: Path) -> tuple[Path, ...]:
    paths = []
    for value in values:
        path = Path(value)
        paths.append(path if path.is_absolute() else base_dir / path)
    return tuple(paths)


def load_prepared_supervised_arrays(metadata_path: Path) -> PreparedSupervisedArrays:
    path = Path(metadata_path)
    payload = json.loads(path.read_text())
    base_dir = path.parent
    manifest = Path(payload["manifest_path"])
    if not manifest.is_absolute():
        manifest = base_dir / manifest
    arrays = PreparedSupervisedArrays(
        metadata_path=path,
        manifest_path=manifest,
        local_shard_paths=_paths_from_metadata(payload.get("local_shard_paths", []), base_dir=base_dir),
        context_shard_paths=_paths_from_metadata(payload.get("context_shard_paths", []), base_dir=base_dir),
        fine_shard_paths=_paths_from_metadata(payload.get("fine_shard_paths", []), base_dir=base_dir),
        rows_per_shard=int(payload["rows_per_shard"]),
        row_count=int(payload.get("completed_row_count", payload.get("total_row_count", 0))),
        image_dtype=str(payload.get("image_dtype", "")),
    )
    if arrays.row_count <= 0:
        raise ValueError(f"Prepared metadata `{path}` has no completed rows.")
    if arrays.rows_per_shard <= 0:
        raise ValueError(f"Prepared metadata `{path}` has an invalid rows_per_shard.")
    for shard_path in (*arrays.local_shard_paths, *arrays.context_shard_paths, *arrays.fine_shard_paths):
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
    if not arrays.manifest_path.exists():
        raise FileNotFoundError(arrays.manifest_path)
    return arrays


def load_prepared_pretraining_arrays(metadata_path: Path) -> PreparedPretrainingArrays:
    path = Path(metadata_path)
    payload = json.loads(path.read_text())
    base_dir = path.parent
    manifest = Path(payload["manifest_path"])
    if not manifest.is_absolute():
        manifest = base_dir / manifest
    arrays = PreparedPretrainingArrays(
        metadata_path=path,
        manifest_path=manifest,
        shard_paths=_paths_from_metadata(payload.get("shard_paths", []), base_dir=base_dir),
        rows_per_shard=int(payload["rows_per_shard"]),
        row_count=int(payload.get("completed_row_count", payload.get("total_row_count", 0))),
        variant_count=int(payload.get("variant_count", 1)),
        image_dtype=str(payload.get("image_dtype", "")),
    )
    if arrays.row_count <= 0:
        raise ValueError(f"Prepared pretraining metadata `{path}` has no completed rows.")
    if arrays.rows_per_shard <= 0:
        raise ValueError(f"Prepared pretraining metadata `{path}` has an invalid rows_per_shard.")
    if arrays.variant_count <= 0:
        raise ValueError(f"Prepared pretraining metadata `{path}` has an invalid variant_count.")
    if not arrays.shard_paths:
        raise ValueError(f"Prepared pretraining metadata `{path}` has no shard paths.")
    for shard_path in arrays.shard_paths:
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
    if not arrays.manifest_path.exists():
        raise FileNotFoundError(arrays.manifest_path)
    return arrays


class PreparedPretrainingDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        metadata_path: Path,
        *,
        seed: int = 0,
        cycle_variants: bool = True,
        intensity_augmentation: IntensityAugmentationConfig | None = None,
    ):
        self.rows = rows.reset_index(drop=True).copy()
        missing = [column for column in ("prepared_index", "scale_id") if column not in self.rows]
        if missing:
            raise ValueError(f"Prepared pretraining rows require column(s): {missing}.")
        self.rows["prepared_index"] = pd.to_numeric(
            self.rows["prepared_index"], errors="raise"
        ).astype(int)
        self.rows["scale_id"] = pd.to_numeric(self.rows["scale_id"], errors="raise").astype(int)
        self.arrays = load_prepared_pretraining_arrays(metadata_path)
        self.seed = int(seed)
        self.cycle_variants = bool(cycle_variants)
        self.intensity_augmentation = intensity_augmentation
        self.epoch = 0
        self.shards = [np.load(path, mmap_mode="r") for path in self.arrays.shard_paths]

    def __len__(self) -> int:
        return int(len(self.rows))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _variant_index(self, prepared_index: int) -> int:
        if not self.cycle_variants or self.arrays.variant_count == 1:
            return 0
        offset = (self.seed * 1_000_003 + int(prepared_index) * 97) % self.arrays.variant_count
        return int((offset + self.epoch) % self.arrays.variant_count)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        prepared_index = int(row["prepared_index"])
        shard_index = prepared_index // int(self.arrays.rows_per_shard)
        row_index = prepared_index % int(self.arrays.rows_per_shard)
        if shard_index < 0 or shard_index >= len(self.shards):
            raise IndexError(f"Prepared index {prepared_index} is outside pretraining shard coverage.")
        stored = self.shards[shard_index][row_index]
        if stored.ndim == 3:
            stored = stored[self._variant_index(prepared_index)]
        image = np.asarray(stored, dtype=np.float32) / 255.0
        if image.ndim == 2:
            image = image[None]
        if image.ndim != 3 or int(image.shape[0]) != 1:
            raise ValueError(f"Prepared pretraining image has unexpected shape {image.shape}.")
        if self.intensity_augmentation is not None:
            params = sample_intensity_parameters(
                self.intensity_augmentation,
                sample_index=int(index),
                epoch=int(self.epoch),
            )
            if params is not None:
                contrast, brightness, blur_sigma = params
                image = apply_intensity_parameters(
                    image,
                    contrast=contrast,
                    brightness=brightness,
                    gaussian_blur_sigma=blur_sigma,
                )
        return {
            "index": int(index),
            "image": image.astype(np.float32, copy=False),
            "scale_id": int(row["scale_id"]),
        }


class PreparedPairedPretrainingDataset(Dataset):
    """Paired local/context views for the same pretraining center."""

    def __init__(
        self,
        rows: pd.DataFrame,
        metadata_path: Path,
        *,
        seed: int = 0,
        cycle_student_variants: bool = True,
        student_intensity_augmentation: IntensityAugmentationConfig | None = None,
    ):
        table = rows.reset_index(drop=True).copy()
        required = {"prepared_index", "scale_id", "center_id"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Paired pretraining rows require column(s): {missing}.")
        table["prepared_index"] = pd.to_numeric(table["prepared_index"], errors="raise").astype(int)
        table["scale_id"] = pd.to_numeric(table["scale_id"], errors="raise").astype(int)
        pair_columns = ["center_id"]
        if "source_id" in table.columns:
            pair_columns.insert(0, "source_id")
        if table.duplicated(pair_columns + ["scale_id"]).any():
            raise ValueError("Paired pretraining rows contain duplicate center/scale combinations.")
        pairs = table.pivot(index=pair_columns, columns="scale_id", values="prepared_index")
        if 0 not in pairs.columns or 1 not in pairs.columns:
            raise ValueError("Paired pretraining requires both scale_id 0 and 1.")
        incomplete = pairs[[0, 1]].isna().any(axis=1)
        if bool(incomplete.any()):
            raise ValueError(f"Paired pretraining found {int(incomplete.sum())} centers missing a scale.")
        self.pairs = pairs[[0, 1]].astype(int).reset_index(drop=True)
        self.arrays = load_prepared_pretraining_arrays(metadata_path)
        self.seed = int(seed)
        self.cycle_student_variants = bool(cycle_student_variants)
        self.student_intensity_augmentation = student_intensity_augmentation
        self.epoch = 0
        self.shards = [np.load(path, mmap_mode="r") for path in self.arrays.shard_paths]

    def __len__(self) -> int:
        return int(len(self.pairs))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _student_variant_index(self, index: int) -> int:
        if not self.cycle_student_variants or self.arrays.variant_count == 1:
            return 0
        offset = (self.seed * 1_000_003 + int(index) * 97) % self.arrays.variant_count
        return int((offset + self.epoch) % self.arrays.variant_count)

    def _load_image(self, prepared_index: int, variant_index: int) -> np.ndarray:
        shard_index = int(prepared_index) // int(self.arrays.rows_per_shard)
        row_index = int(prepared_index) % int(self.arrays.rows_per_shard)
        if shard_index < 0 or shard_index >= len(self.shards):
            raise IndexError(f"Prepared index {prepared_index} is outside pretraining shard coverage.")
        stored = self.shards[shard_index][row_index]
        if stored.ndim == 3:
            stored = stored[int(variant_index)]
        image = np.asarray(stored, dtype=np.float32) / 255.0
        if image.ndim == 2:
            image = image[None]
        if image.ndim != 3 or int(image.shape[0]) != 1:
            raise ValueError(f"Prepared paired pretraining image has unexpected shape {image.shape}.")
        return image.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs.iloc[int(index)]
        local_index = int(pair.loc[0])
        context_index = int(pair.loc[1])
        student_variant = self._student_variant_index(int(index))
        item = {
            "index": int(index),
            "local_student_image": self._load_image(local_index, student_variant),
            "context_student_image": self._load_image(context_index, student_variant),
            "local_teacher_image": self._load_image(local_index, 0),
            "context_teacher_image": self._load_image(context_index, 0),
        }
        if self.student_intensity_augmentation is not None:
            params = sample_intensity_parameters(
                self.student_intensity_augmentation,
                sample_index=int(index),
                epoch=int(self.epoch),
            )
            if params is not None:
                contrast, brightness, blur_sigma = params
                for key in ("local_student_image", "context_student_image"):
                    item[key] = apply_intensity_parameters(
                        item[key],
                        contrast=contrast,
                        brightness=brightness,
                        gaussian_blur_sigma=blur_sigma,
                    )
        return item


class PreparedCellCropDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        metadata_path: Path,
        config: ModelConfig,
        *,
        intensity_augmentation: IntensityAugmentationConfig | None = None,
    ):
        self.rows = rows.reset_index(drop=True).copy()
        if "prepared_index" not in self.rows.columns:
            raise ValueError("Prepared supervised rows require a `prepared_index` column.")
        self.rows["prepared_index"] = pd.to_numeric(self.rows["prepared_index"], errors="raise").astype(int)
        self.arrays = load_prepared_supervised_arrays(metadata_path)
        self.config = config
        self.intensity_augmentation = intensity_augmentation
        self.epoch = 0
        self.local_shards = (
            [np.load(path, mmap_mode="r") for path in self.arrays.local_shard_paths]
            if config.use_local_branch
            else []
        )
        self.context_shards = (
            [np.load(path, mmap_mode="r") for path in self.arrays.context_shard_paths]
            if config.use_context_branch
            else []
        )
        self.fine_shards = (
            [np.load(path, mmap_mode="r") for path in self.arrays.fine_shard_paths]
            if config.use_fine_branch
            else []
        )

    def __len__(self) -> int:
        return int(len(self.rows))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_image(self, shards: list[np.ndarray], prepared_index: int, *, branch_name: str) -> np.ndarray:
        shard_index = int(prepared_index) // int(self.arrays.rows_per_shard)
        row_index = int(prepared_index) % int(self.arrays.rows_per_shard)
        if shard_index < 0 or shard_index >= len(shards):
            raise IndexError(f"Prepared index {prepared_index} is outside {branch_name} shard coverage.")
        image = np.asarray(shards[shard_index][row_index], dtype=np.float32) / 255.0
        if image.ndim == 2:
            image = image[None]
        if image.ndim != 3 or int(image.shape[0]) != 1:
            raise ValueError(f"Prepared {branch_name} image has unexpected shape {image.shape}.")
        return image.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        prepared_index = int(row["prepared_index"])
        item: dict[str, Any] = {"index": int(index)}
        if self.config.use_local_branch:
            item["local_image"] = self._load_image(self.local_shards, prepared_index, branch_name="local")
        if self.config.use_context_branch:
            item["context_image"] = self._load_image(self.context_shards, prepared_index, branch_name="context")
        if self.config.use_fine_branch:
            item["fine_image"] = self._load_image(self.fine_shards, prepared_index, branch_name="fine")
        if self.intensity_augmentation is not None:
            params = sample_intensity_parameters(
                self.intensity_augmentation,
                sample_index=int(index),
                epoch=int(self.epoch),
            )
            if params is not None:
                contrast, brightness, blur_sigma = params
                for key in [name for name in item if name.endswith("_image")]:
                    item[key] = apply_intensity_parameters(
                        item[key],
                        contrast=contrast,
                        brightness=brightness,
                        gaussian_blur_sigma=blur_sigma,
                    )
        return item


def prepared_summary(metadata_path: Path) -> dict[str, Any]:
    arrays = load_prepared_supervised_arrays(metadata_path)
    return {
        "metadata_path": str(arrays.metadata_path),
        "manifest_path": str(arrays.manifest_path),
        "row_count": int(arrays.row_count),
        "rows_per_shard": int(arrays.rows_per_shard),
        "local_shard_count": int(len(arrays.local_shard_paths)),
        "context_shard_count": int(len(arrays.context_shard_paths)),
        "fine_shard_count": int(len(arrays.fine_shard_paths)),
        "image_dtype": arrays.image_dtype,
    }


def prepared_pretraining_summary(metadata_path: Path) -> dict[str, Any]:
    arrays = load_prepared_pretraining_arrays(metadata_path)
    return {
        "metadata_path": str(arrays.metadata_path),
        "manifest_path": str(arrays.manifest_path),
        "row_count": int(arrays.row_count),
        "rows_per_shard": int(arrays.rows_per_shard),
        "shard_count": int(len(arrays.shard_paths)),
        "variant_count": int(arrays.variant_count),
        "image_dtype": arrays.image_dtype,
    }
