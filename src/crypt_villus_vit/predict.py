from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from tqdm import tqdm

from crypt_villus_vit.augmentation import IntensityAugmentationConfig
from crypt_villus_vit.augmentation import apply_intensity_parameters
from crypt_villus_vit.augmentation import sample_intensity_parameters
from crypt_villus_vit.crops import CropExtractor
from crypt_villus_vit.crops import native_crop_size_px
from crypt_villus_vit.gates import assign_axis_epithelial_gates
from crypt_villus_vit.gates import summarize_gate_percentages
from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.model import load_model_checkpoint
from crypt_villus_vit.sources import SourceSpec


class CellCropDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        sources: dict[str, SourceSpec],
        config: ModelConfig,
        *,
        reference_pixel_size_um: float | None = None,
        intensity_augmentation: IntensityAugmentationConfig | None = None,
        input_protocol: str = "legacy_float_v1",
    ):
        self.rows = rows.reset_index(drop=True).copy()
        if input_protocol == "direct_pyramid_crop_normalize_resize_uint8_v1":
            from .production_crops import ProductionCropExtractor
            self.extractor = ProductionCropExtractor(sources)
        elif input_protocol == "legacy_float_v1":
            self.extractor = CropExtractor(sources)
        else:
            raise ValueError(f"Unknown input protocol: {input_protocol}")
        self.config = config
        self.reference_pixel_size_um = (
            None if reference_pixel_size_um is None else float(reference_pixel_size_um)
        )
        self.intensity_augmentation = intensity_augmentation
        self.epoch = 0

    def crop_size_for_source(self, source_id: str, reference_crop_px: int) -> int:
        if self.reference_pixel_size_um is None:
            return int(reference_crop_px)
        source = self.extractor.sources[str(source_id)]
        if source.pixel_size_um is None:
            raise ValueError(
                f"Source `{source_id}` requires pixel_size_um when reference-pixel scaling is enabled."
            )
        return native_crop_size_px(
            int(reference_crop_px),
            reference_pixel_size_um=self.reference_pixel_size_um,
            source_pixel_size_um=float(source.pixel_size_um),
        )

    def __len__(self) -> int:
        return int(len(self.rows))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        source_id = str(row["source_id"])
        x = float(row["centroid_x_fullres_px"])
        y = float(row["centroid_y_fullres_px"])
        item: dict[str, Any] = {"index": int(index)}
        if self.config.use_local_branch:
            item["local_image"] = self.extractor.crop_for_source(
                source_id,
                center_x=x,
                center_y=y,
                crop_size_px=self.crop_size_for_source(source_id, self.config.local_crop_px),
                output_size_px=self.config.input_size_px,
            )[None]
        if self.config.use_context_branch:
            item["context_image"] = self.extractor.crop_for_source(
                source_id,
                center_x=x,
                center_y=y,
                crop_size_px=self.crop_size_for_source(source_id, self.config.context_crop_px),
                output_size_px=self.config.input_size_px,
            )[None]
        if self.config.use_fine_branch:
            item["fine_image"] = self.extractor.crop_for_source(
                source_id,
                center_x=x,
                center_y=y,
                crop_size_px=self.crop_size_for_source(source_id, self.config.fine_crop_px),
                output_size_px=self.config.fine_input_size_px,
            )[None]
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


class SourceGroupedBatchSampler(Sampler[list[int]]):
    """Yield one-source-at-a-time batches to avoid repeatedly loading large source images."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
    ):
        if "source_id" not in rows.columns:
            raise ValueError("SourceGroupedBatchSampler requires a `source_id` column.")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        source_ids = rows["source_id"].astype(str).to_numpy()
        self.source_to_indices: dict[str, np.ndarray] = {}
        for source_id in pd.unique(source_ids):
            self.source_to_indices[str(source_id)] = np.flatnonzero(source_ids == str(source_id)).astype(np.int64)
        self.source_order = sorted(self.source_to_indices)
        self.batch_count = int(sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.source_to_indices.values()))

    def __iter__(self):
        rng = np.random.default_rng(int(self.seed + self.epoch))
        self.epoch += 1
        source_order = list(self.source_order)
        if self.shuffle:
            rng.shuffle(source_order)
        for source_id in source_order:
            indices = np.array(self.source_to_indices[source_id], copy=True)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, int(len(indices)), self.batch_size):
                yield indices[start : start + self.batch_size].astype(int).tolist()

    def __len__(self) -> int:
        return int(self.batch_count)


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    keys = [key for key in batch[0] if key != "index"]
    out = {"index": torch.tensor([item["index"] for item in batch], dtype=torch.long)}
    for key in keys:
        out[key] = torch.from_numpy(np.stack([item[key] for item in batch]).astype(np.float32))
    return out


def predict_cells(
    rows: pd.DataFrame,
    *,
    sources: dict[str, SourceSpec],
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    strict_checkpoint: bool = True,
    task_type: str | None = None,
    prediction_column: str | None = None,
    classification_threshold: float = 0.5,
    local_crop_px: int | None = None,
    context_crop_px: int | None = None,
    fine_crop_px: int | None = None,
    reference_pixel_size_um: float | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    resolved_device = str(device)
    if resolved_device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install the cu126 extra or select --device cpu.")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device=resolved_device, strict=strict_checkpoint)
    metadata = dict(checkpoint.get("metadata", {}))
    input_protocol = metadata.get("input_protocol", "legacy_float_v1")
    if reference_pixel_size_um is None:
        reference_pixel_size_um = metadata.get("reference_pixel_size_um")
    elif metadata.get('release_model') and reference_pixel_size_um != metadata.get('reference_pixel_size_um'):
        raise ValueError('Released reference pixel size is fixed; set the source calibration in its manifest.')
    resolved_task_type = str(task_type or metadata.get("task_type", "axis_regression")).strip().lower().replace("-", "_")
    if resolved_task_type in {"axis", "regression"}:
        resolved_task_type = "axis_regression"
    elif resolved_task_type in {"binary", "classification"}:
        resolved_task_type = "binary_classification"
    if resolved_task_type not in {"axis_regression", "binary_classification"}:
        raise ValueError("task_type must be one of: axis_regression, binary_classification.")
    if not 0.0 <= float(classification_threshold) <= 1.0:
        raise ValueError("classification_threshold must be between 0 and 1.")
    if prediction_column is None:
        prediction_column = str(
            metadata.get(
                "prediction_column",
                "peyer_probability" if resolved_task_type == "binary_classification" else "predicted_axis_coordinate",
            )
        )
    config = model.config
    crop_overrides = {
        key: int(value)
        for key, value in {
            "local_crop_px": local_crop_px,
            "context_crop_px": context_crop_px,
            "fine_crop_px": fine_crop_px,
        }.items()
        if value is not None
    }
    if crop_overrides:
        if metadata.get('release_model') and any(getattr(config, k) != v for k, v in crop_overrides.items()):
            raise ValueError('Released model crop widths are fixed; specify the source pixel size for physical scaling.')
        config = replace(config, **crop_overrides)
    missing = sorted({"source_id", "centroid_x_fullres_px", "centroid_y_fullres_px"} - set(rows.columns))
    if missing:
        raise ValueError(f"Prediction rows are missing required column(s): {missing}")
    if rows.empty:
        raise ValueError('Prediction table is empty')
    if 'cell_id' in rows and (rows.cell_id.isna().any() or rows.cell_id.duplicated().any()):
        raise ValueError('cell_id must be present and unique for every row when supplied')
    dataset = CellCropDataset(
        rows,
        sources,
        config,
        reference_pixel_size_um=reference_pixel_size_um,
        input_protocol=input_protocol,
    )
    loader_kwargs = {
        "batch_sampler": SourceGroupedBatchSampler(rows, batch_size=int(batch_size), shuffle=False),
        "collate_fn": _collate,
        "num_workers": int(num_workers),
    }
    if int(num_workers) > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    axis_predictions = np.empty(len(rows), dtype=np.float32)
    epithelial_predictions = np.empty(len(rows), dtype=np.float32)
    binary_logits = np.empty(len(rows), dtype=np.float32)
    print(
        "Prediction rows | "
        f"row_count={len(rows):,} | "
        f"source_count={rows['source_id'].astype(str).nunique():,} | "
        f"batch_size={int(batch_size)} | "
        f"num_workers={int(num_workers)} | "
        f"device={resolved_device}",
        flush=True,
    )
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="predict", unit="batch"):
            indices = batch.pop("index").numpy()
            batch = {key: value.to(resolved_device) for key, value in batch.items()}
            output = model(batch)
            if resolved_task_type == "axis_regression":
                axis_predictions[indices] = output["predicted_axis_coordinate"].detach().cpu().numpy()
                epithelial_predictions[indices] = (
                    output["predicted_epithelial_distance_clipped_1p0"].detach().cpu().numpy()
                )
            if resolved_task_type == "binary_classification":
                axis_logit = output.get("axis_logit")
                if axis_logit is None:
                    probabilities = torch.clamp(output["predicted_axis_coordinate"], 1e-6, 1.0 - 1e-6)
                    axis_logit = torch.logit(probabilities)
                binary_logits[indices] = axis_logit.detach().cpu().numpy()
    export = rows.reset_index(drop=True).copy()
    if resolved_task_type == "binary_classification":
        class_column = (
            prediction_column[: -len("_probability")] + "_class"
            if prediction_column.endswith("_probability")
            else f"{prediction_column}_class"
        )
        logit_column = (
            prediction_column[: -len("_probability")] + "_logit"
            if prediction_column.endswith("_probability")
            else f"{prediction_column}_logit"
        )
        export[prediction_column] = torch.from_numpy(binary_logits).sigmoid().numpy()
        export[logit_column] = binary_logits
        export[class_column] = (export[prediction_column].to_numpy(dtype=float) >= float(classification_threshold)).astype(int)
    else:
        export["predicted_axis_coordinate"] = axis_predictions
        export["predicted_epithelial_distance_clipped_1p0"] = epithelial_predictions
        export["predicted_bin"] = np.clip(np.floor(axis_predictions * 32), 0, 31).astype(int)
        export["predicted_offset"] = axis_predictions * 32 - export["predicted_bin"]
        export = assign_axis_epithelial_gates(export)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_csv = output_dir / "predictions.csv"
    summary_json = output_dir / "summary.json"
    export.to_csv(prediction_csv, index=False)
    gate_csv = None
    if resolved_task_type == "axis_regression":
        gate_csv = output_dir / "gate_percentages.csv"
        gate_table = summarize_gate_percentages(export, group_columns=("source_id",))
        gate_table.to_csv(gate_csv, index=False)
    summary = {
        "task_type": resolved_task_type,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_format": checkpoint.get("format", "unknown"),
        "input_protocol": input_protocol,
        "output_dir": str(output_dir),
        "prediction_csv": str(prediction_csv),
        "gate_percentages_csv": None if gate_csv is None else str(gate_csv),
        "row_count": int(len(export)),
        "source_count": int(export["source_id"].astype(str).nunique()),
        "device": resolved_device,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "prefetch_factor": None if int(num_workers) <= 0 else int(prefetch_factor or 0),
        "model_config": asdict(model.config),
        "inference_config": asdict(config),
        "prediction_column": str(prediction_column),
        "classification_threshold": float(classification_threshold),
        "crop_overrides": crop_overrides,
        "reference_pixel_size_um": reference_pixel_size_um,
        "resolved_crop_sizes_by_source": {
            source_id: {
                branch: dataset.crop_size_for_source(source_id, crop_size)
                for branch, crop_size in {
                    "local": config.local_crop_px,
                    "context": config.context_crop_px,
                    "fine": config.fine_crop_px,
                }.items()
                if getattr(config, f"use_{branch}_branch")
            }
            for source_id in sorted(rows["source_id"].astype(str).unique())
        },
    }
    if resolved_task_type == "binary_classification":
        summary["prediction_mean"] = float(export[prediction_column].mean())
        summary["prediction_positive_fraction"] = float((export[prediction_column] >= float(classification_threshold)).mean())
    else:
        summary["axis_prediction_mean"] = float(export["predicted_axis_coordinate"].mean())
        summary["epithelial_prediction_mean"] = float(export["predicted_epithelial_distance_clipped_1p0"].mean())
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary
