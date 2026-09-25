from __future__ import annotations

from dataclasses import asdict
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from crypt_villus_vit.augmentation import make_intensity_augmentation_config
from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.model import MultitaskDapiVit
from crypt_villus_vit.model import save_model_checkpoint
from crypt_villus_vit.prepared import PreparedCellCropDataset
from crypt_villus_vit.prepared import prepared_summary
from crypt_villus_vit.prepared import validate_prepared_crop_configuration
from crypt_villus_vit.predict import CellCropDataset
from crypt_villus_vit.predict import SourceGroupedBatchSampler
from crypt_villus_vit.predict import _collate
from crypt_villus_vit.sources import SourceSpec
from crypt_villus_vit.raw_training import (
    RamCropCache, close_training_loader, make_raw_training_loader, warm_ram_cache,
)


_REQUIRED_CELL_COLUMNS = ("source_id", "centroid_x_fullres_px", "centroid_y_fullres_px")
_AXIS_REGRESSION_TASK = "axis_regression"
_BINARY_CLASSIFICATION_TASK = "binary_classification"


def _normalize_task_type(task_type: str) -> str:
    value = str(task_type or _AXIS_REGRESSION_TASK).strip().lower().replace("-", "_")
    aliases = {
        "axis": _AXIS_REGRESSION_TASK,
        "regression": _AXIS_REGRESSION_TASK,
        "axis_regression": _AXIS_REGRESSION_TASK,
        "binary": _BINARY_CLASSIFICATION_TASK,
        "classification": _BINARY_CLASSIFICATION_TASK,
        "binary_classification": _BINARY_CLASSIFICATION_TASK,
    }
    if value not in aliases:
        raise ValueError("task_type must be one of: axis_regression, binary_classification.")
    return aliases[value]


def _binary_classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if y.size == 0:
        return {
            "binary_accuracy": float("nan"),
            "binary_precision": float("nan"),
            "binary_recall": float("nan"),
            "binary_f1": float("nan"),
            "binary_average_precision": float("nan"),
            "binary_auroc": float("nan"),
            "binary_prevalence": float("nan"),
            "binary_scored_n": 0.0,
            "binary_positive_n": 0.0,
        }
    hard = y >= 0.5
    predicted = p >= float(threshold)
    tp = float(np.logical_and(predicted, hard).sum())
    fp = float(np.logical_and(predicted, ~hard).sum())
    tn = float(np.logical_and(~predicted, ~hard).sum())
    fn = float(np.logical_and(~predicted, hard).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and (precision + recall) else float("nan")

    positive_count = int(hard.sum())
    negative_count = int((~hard).sum())
    from sklearn.metrics import average_precision_score, roc_auc_score
    auroc = float(roc_auc_score(hard, p)) if positive_count and negative_count else float("nan")
    average_precision = float(average_precision_score(hard, p)) if positive_count else float("nan")

    return {
        "binary_accuracy": (tp + tn) / max(float(len(y)), 1.0),
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "binary_average_precision": average_precision,
        "binary_auroc": auroc,
        "binary_prevalence": float(hard.mean()),
        "binary_scored_n": float(len(y)),
        "binary_positive_n": float(positive_count),
    }


def _resolve_positive_class_weight(targets: pd.Series, value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "off", "false"}:
            return None
        if normalized != "auto":
            weight = float(value)
            if weight <= 0:
                raise ValueError("positive_class_weight must be positive.")
            return weight
        hard = pd.to_numeric(targets, errors="raise").to_numpy(dtype=np.float64) >= 0.5
        positive = int(hard.sum())
        negative = int((~hard).sum())
        if positive == 0 or negative == 0:
            raise ValueError("Binary training requires at least one positive and one negative training row.")
        return float(negative / positive)
    weight = float(value)
    if weight <= 0:
        raise ValueError("positive_class_weight must be positive.")
    return weight


def prepare_supervised_manifest(
    rows: pd.DataFrame,
    *,
    target_axis_column: str = "target_axis",
    target_epithelial_column: str = "epithelial_distance_clipped_1p0",
    require_coordinates: bool = True,
) -> pd.DataFrame:
    """Validate and normalize a supervised training manifest."""

    required = (*_REQUIRED_CELL_COLUMNS, target_axis_column) if require_coordinates else ("source_id", target_axis_column)
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Training rows are missing required column(s): {missing}")
    out = rows.copy()
    if target_axis_column != "target_axis":
        out["target_axis"] = out[target_axis_column]
    if target_epithelial_column in out.columns:
        if target_epithelial_column != "epithelial_distance_clipped_1p0":
            out["epithelial_distance_clipped_1p0"] = out[target_epithelial_column]
    elif "epithelial_distance_clipped_1p0" not in out.columns:
        # A missing target is unobserved, never a copy of a different coordinate.
        out["epithelial_distance_clipped_1p0"] = np.nan
    numeric_columns = (
        "target_axis",
        "epithelial_distance_clipped_1p0",
    )
    required_numeric_columns = ("target_axis",)
    if require_coordinates:
        numeric_columns = ("centroid_x_fullres_px", "centroid_y_fullres_px", *numeric_columns)
        required_numeric_columns = ("centroid_x_fullres_px", "centroid_y_fullres_px", *required_numeric_columns)
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    finite = np.ones(len(out), dtype=bool)
    for column in required_numeric_columns:
        finite &= np.isfinite(out[column].to_numpy(dtype=np.float64))
    if not bool(finite.all()):
        dropped = int((~finite).sum())
        out = out.loc[finite].copy()
        if out.empty:
            raise ValueError(f"All training rows were non-finite after validation; dropped {dropped} row(s).")
    out["source_id"] = out["source_id"].astype(str)
    return out.reset_index(drop=True)


def split_supervised_manifest(
    rows: pd.DataFrame,
    *,
    validation_fraction: float = 0.0,
    seed: int = 0,
    split_column: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return train/validation manifests using an explicit split column or a random split."""

    if split_column in rows.columns:
        split = rows[split_column].fillna("").astype(str).str.strip().str.lower()
        allowed = {"train", "training", "val", "valid", "validation", "test"}
        if not split.isin(allowed).all():
            raise ValueError("Explicit splits must contain only train, validation or test labels; blank/unknown labels are ambiguous.")
        val_mask = split.isin({"val", "valid", "validation"})
        train_mask = split.isin({"train", "training"})
        if bool(val_mask.any()):
            train_rows = rows.loc[train_mask].copy()
            val_rows = rows.loc[val_mask].copy()
            if train_rows.empty:
                raise ValueError(f"`{split_column}` selected validation rows but no training rows.")
            return train_rows.reset_index(drop=True), val_rows.reset_index(drop=True)
        raise ValueError("An explicit split must contain training and validation rows. Test rows are never reassigned.")

    fraction = float(validation_fraction)
    if fraction <= 0:
        return rows.reset_index(drop=True), None
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(rows) < 2:
        raise ValueError("At least two rows are required for a random validation split.")
    rng = np.random.default_rng(int(seed))
    warnings.warn("Random cell-level validation can overlap neighboring image crops. Use explicit section-disjoint splits for evaluation.", stacklevel=2)
    indices = np.arange(len(rows))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(rows) * fraction)))
    val_count = min(val_count, len(rows) - 1)
    val_index = indices[:val_count]
    train_index = indices[val_count:]
    return rows.iloc[train_index].reset_index(drop=True), rows.iloc[val_index].reset_index(drop=True)


def _target_tensors(rows: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_axis = torch.tensor(rows["target_axis"].to_numpy(dtype="float32"), dtype=torch.float32)
    target_epi = torch.tensor(rows["epithelial_distance_clipped_1p0"].to_numpy(dtype="float32"), dtype=torch.float32)
    target_epi_mask = torch.isfinite(target_epi)
    return target_axis, target_epi, target_epi_mask


def _set_dataset_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    set_epoch = getattr(dataset, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))
    sampler_set_epoch = getattr(loader.batch_sampler, "set_epoch", None)
    if callable(sampler_set_epoch):
        sampler_set_epoch(int(epoch))


def _run_epoch(
    model: MultitaskDapiVit,
    loader: DataLoader,
    *,
    target_axis: torch.Tensor,
    target_epi: torch.Tensor,
    target_epi_mask: torch.Tensor,
    device: str,
    task_type: str = _AXIS_REGRESSION_TASK,
    binary_positive_class_weight: float | None = None,
    classification_threshold: float = 0.5,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
    desc: str | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_axis_abs = 0.0
    total_epi_abs = 0.0
    total_n = 0
    total_epi_n = 0
    probability_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    pos_weight = (
        None
        if binary_positive_class_weight is None
        else torch.tensor(float(binary_positive_class_weight), dtype=torch.float32, device=torch.device(device))
    )
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        batches = tqdm(loader, desc=desc, unit="batch") if desc else loader
        for batch in batches:
            indices = batch.pop("index")
            batch = {key: value.to(device, non_blocking=loader.pin_memory) for key, value in batch.items()}
            y_axis = target_axis[indices].to(device)
            y_epi = target_epi[indices].to(device)
            y_epi_mask = target_epi_mask[indices].to(device)
            output = model(batch)
            axis_pred = output["predicted_axis_coordinate"]
            epi_pred = output.get("predicted_epithelial_distance_clipped_1p0")
            if task_type == _BINARY_CLASSIFICATION_TASK:
                axis_logit = output.get("axis_logit")
                if axis_logit is None:
                    clipped = torch.clamp(axis_pred, 1e-6, 1.0 - 1e-6)
                    axis_logit = torch.logit(clipped)
                loss = F.binary_cross_entropy_with_logits(axis_logit, y_axis, pos_weight=pos_weight)
                probability_chunks.append(torch.sigmoid(axis_logit).detach().cpu().numpy())
                target_chunks.append(y_axis.detach().cpu().numpy())
            else:
                loss = F.smooth_l1_loss(axis_pred, y_axis)
                if bool(y_epi_mask.any()):
                    loss = loss + F.smooth_l1_loss(epi_pred[y_epi_mask], y_epi[y_epi_mask])
            if is_train:
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Supervised loss became non-finite before the optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(gradient_clip_norm),
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        raise FloatingPointError(
                            "Supervised gradient norm became non-finite before the optimizer step."
                        )
                optimizer.step()
            batch_n = int(len(indices))
            batch_epi_n = int(y_epi_mask.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * batch_n
            if task_type != _BINARY_CLASSIFICATION_TASK:
                total_axis_abs += float(torch.abs(axis_pred.detach() - y_axis).sum().cpu())
                if batch_epi_n:
                    total_epi_abs += float(torch.abs(epi_pred.detach()[y_epi_mask] - y_epi[y_epi_mask]).sum().cpu())
            total_n += batch_n
            total_epi_n += batch_epi_n
    denominator = max(total_n, 1)
    if task_type == _BINARY_CLASSIFICATION_TASK:
        probabilities = np.concatenate(probability_chunks) if probability_chunks else np.array([], dtype=np.float32)
        targets = np.concatenate(target_chunks) if target_chunks else np.array([], dtype=np.float32)
        metrics = _binary_classification_metrics(targets, probabilities, threshold=float(classification_threshold))
        return {
            "loss": total_loss / denominator,
            "binary_bce": total_loss / denominator,
            **metrics,
        }
    epi_denominator = max(total_epi_n, 1)
    return {
        "loss": total_loss / denominator,
        "axis_mae": total_axis_abs / denominator,
        "epithelial_mae": total_epi_abs / epi_denominator if total_epi_n else float("nan"),
        "axis_scored_n": float(total_n),
        "epithelial_scored_n": float(total_epi_n),
    }


def _apply_freeze_mode(model: MultitaskDapiVit, freeze_mode: str) -> dict[str, object]:
    mode = str(freeze_mode).strip().lower()
    if mode in {"", "none"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        mode = "none"
    elif mode in {"heads", "head", "head_only", "head-only"}:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module_name in ("head", "axis_head", "epithelial_head"):
            module = getattr(model, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True
        if getattr(model, "fine_gate", None) is not None:
            model.fine_gate.requires_grad = True
        mode = "heads"
    elif mode in {"last1", "last_1", "last-block", "last_block"}:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module_name in ("head", "axis_head", "epithelial_head"):
            module = getattr(model, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True
        if getattr(model, "fine_gate", None) is not None:
            model.fine_gate.requires_grad = True
        for encoder_name in ("shared_encoder", "local_encoder", "context_encoder"):
            encoder = getattr(model, encoder_name, None)
            if encoder is None:
                continue
            layers = getattr(getattr(encoder, "encoder", None), "layers", None)
            if layers is not None and len(layers):
                for parameter in layers[-1].parameters():
                    parameter.requires_grad = True
            for parameter in encoder.norm.parameters():
                parameter.requires_grad = True
        mode = "last1"
    else:
        raise ValueError("freeze_mode must be one of: 'none', 'heads', 'last1'.")

    trainable_parameter_count = 0
    frozen_parameter_count = 0
    trainable_modules: set[str] = set()
    frozen_modules: set[str] = set()
    for name, parameter in model.named_parameters():
        module_name = name.split(".", 1)[0]
        if parameter.requires_grad:
            trainable_parameter_count += int(parameter.numel())
            trainable_modules.add(module_name)
        else:
            frozen_parameter_count += int(parameter.numel())
            frozen_modules.add(module_name)
    return {
        "freeze_mode": mode,
        "trainable_parameter_count": int(trainable_parameter_count),
        "frozen_parameter_count": int(frozen_parameter_count),
        "trainable_modules": sorted(trainable_modules),
        "frozen_modules": sorted(frozen_modules),
    }


def train_model(
    rows: pd.DataFrame,
    *,
    sources: dict[str, SourceSpec],
    output_dir: Path,
    config: ModelConfig | None = None,
    epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    validation_fraction: float = 0.0,
    seed: int = 0,
    num_workers: int = 0,
    task_type: str = _AXIS_REGRESSION_TASK,
    target_column: str | None = None,
    prediction_column: str | None = None,
    positive_class_weight: str | float | int | None = None,
    classification_threshold: float = 0.5,
    target_axis_column: str = "target_axis",
    target_epithelial_column: str = "epithelial_distance_clipped_1p0",
    initial_checkpoint: Path | None = None,
    pretrained_checkpoint: Path | None = None,
    prepared_supervised_metadata: Path | None = None,
    augment_intensity_probability: float = 0.0,
    augment_brightness_delta: float = 0.0,
    augment_contrast_range: tuple[float, float] = (1.0, 1.0),
    augment_gaussian_blur_sigma_range: tuple[float, float] = (0.0, 0.0),
    gradient_clip_norm: float | None = None,
    reference_pixel_size_um: float | None = None,
    freeze_mode: str = "none",
    input_protocol: str = "direct_pyramid_crop_normalize_resize_uint8_v1",
    crop_backend: str = "reference",
    tile_cache_mib: int = 256,
    ram_crop_cache_mib: int = 0,
    warm_crop_cache: bool = True,
    persistent_workers: bool | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if num_workers < 0 or tile_cache_mib < 0 or ram_crop_cache_mib < 0:
        raise ValueError("Worker and cache sizes must be nonnegative")
    if crop_backend not in {"reference", "cached", "rust"}:
        raise ValueError("crop_backend must be reference, cached, or rust")
    fast_raw = crop_backend != "reference"
    if prepared_supervised_metadata is not None and (fast_raw or ram_crop_cache_mib or persistent_workers):
        raise ValueError("Raw-image cache options cannot be combined with prepared-supervised-metadata")
    if not fast_raw and (ram_crop_cache_mib or persistent_workers):
        raise ValueError("RAM caching and persistent workers require crop-backend cached or rust")
    if fast_raw and input_protocol != "direct_pyramid_crop_normalize_resize_uint8_v1":
        raise ValueError("Optimized raw training requires the production input protocol")
    resolved_persistent = bool(num_workers and fast_raw and persistent_workers is not False)
    if initial_checkpoint is not None and pretrained_checkpoint is not None:
        raise ValueError("Use either `initial_checkpoint` or `pretrained_checkpoint`, not both.")
    if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when provided.")
    task_type = _normalize_task_type(task_type)
    if target_column is not None:
        target_axis_column = str(target_column)
    if prediction_column is None:
        prediction_column = "peyer_probability" if task_type == _BINARY_CLASSIFICATION_TASK else "predicted_axis_coordinate"
    if not 0.0 <= float(classification_threshold) <= 1.0:
        raise ValueError("classification_threshold must be between 0 and 1.")
    config = config or ModelConfig()
    if prepared_supervised_metadata is not None:
        reference_pixel_size_um = validate_prepared_crop_configuration(
            prepared_supervised_metadata, config, input_protocol=input_protocol,
            reference_pixel_size_um=reference_pixel_size_um)
    rows = prepare_supervised_manifest(
        rows,
        target_axis_column=target_axis_column,
        target_epithelial_column=target_epithelial_column,
        require_coordinates=prepared_supervised_metadata is None,
    )
    if task_type == _BINARY_CLASSIFICATION_TASK:
        targets = rows["target_axis"].to_numpy(dtype=np.float64)
        finite_targets = np.isfinite(targets)
        if not bool(finite_targets.all()):
            raise ValueError("Binary classification targets must be finite.")
        if bool(((targets < 0.0) | (targets > 1.0)).any()):
            raise ValueError("Binary classification targets must be in [0, 1].")
    train_rows, val_rows = split_supervised_manifest(rows, validation_fraction=validation_fraction, seed=seed)
    if task_type == _AXIS_REGRESSION_TASK and not np.isfinite(train_rows['epithelial_distance_clipped_1p0']).any():
        raise ValueError("Coordinate training needs epithelial-distance labels as well as target_axis; missing labels cannot be replaced by the other coordinate.")
    resolved_positive_class_weight = (
        _resolve_positive_class_weight(train_rows["target_axis"], positive_class_weight)
        if task_type == _BINARY_CLASSIFICATION_TASK
        else None
    )
    resolved_device = str(device)
    if resolved_device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install the cu126 extra or select --device cpu.")
    torch.manual_seed(int(seed))
    intensity_augmentation = make_intensity_augmentation_config(
        probability=float(augment_intensity_probability),
        brightness_delta=float(augment_brightness_delta),
        contrast_range=augment_contrast_range,
        gaussian_blur_sigma_range=augment_gaussian_blur_sigma_range,
        seed=int(seed),
    )
    intensity_augmentation_summary = None if intensity_augmentation is None else intensity_augmentation.to_dict()
    ram_cache = (RamCropCache(len(train_rows) + (0 if val_rows is None else len(val_rows)), config,
                             ram_crop_cache_mib) if ram_crop_cache_mib else None)
    raw_options = dict(batch_size=int(batch_size), seed=int(seed), num_workers=int(num_workers),
                       device=resolved_device, crop_backend=crop_backend, tile_cache_mib=tile_cache_mib,
                       reference_pixel_size_um=reference_pixel_size_um, input_protocol=input_protocol,
                       cache=ram_cache, persistent_workers=resolved_persistent)
    loading_settings = dict(crop_backend="prepared" if prepared_supervised_metadata else crop_backend,
                            num_workers=int(num_workers), tile_cache_mib_per_worker=tile_cache_mib if fast_raw else 0,
                            ram_crop_cache_budget_mib=ram_crop_cache_mib,
                            ram_crop_cache_bytes=0 if ram_cache is None else ram_cache.allocated_bytes,
                            ram_crop_cache_rows=0 if ram_cache is None else ram_cache.capacity,
                            warm_crop_cache=bool(ram_cache is not None and warm_crop_cache),
                            persistent_workers=resolved_persistent, training_order_changed=False)
    if fast_raw:
        batching_strategy = "source_grouped"
        loader = make_raw_training_loader(train_rows, sources, config, shuffle=True,
                                         intensity_augmentation=intensity_augmentation, **raw_options)
    elif prepared_supervised_metadata is None:
        dataset = CellCropDataset(
            train_rows,
            sources,
            config,
            reference_pixel_size_um=reference_pixel_size_um,
            intensity_augmentation=intensity_augmentation,
            input_protocol=input_protocol,
        )
    else:
        dataset = PreparedCellCropDataset(
            train_rows,
            prepared_supervised_metadata,
            config,
            intensity_augmentation=intensity_augmentation,
        )
    if not fast_raw and prepared_supervised_metadata is None:
        batching_strategy = "source_grouped"
        loader = DataLoader(
            dataset,
            batch_sampler=SourceGroupedBatchSampler(
                train_rows,
                batch_size=int(batch_size),
                shuffle=True,
                seed=int(seed),
            ),
            collate_fn=_collate,
            num_workers=int(num_workers),
        )
    elif not fast_raw:
        batching_strategy = "global_shuffle"
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            generator=generator,
            collate_fn=_collate,
            num_workers=int(num_workers),
        )
    val_loader = None
    if val_rows is not None and not val_rows.empty:
        if fast_raw:
            val_loader = make_raw_training_loader(val_rows, sources, config, shuffle=False,
                                                 cache_offset=len(train_rows), **raw_options)
        elif prepared_supervised_metadata is None:
            val_dataset = CellCropDataset(
                val_rows,
                sources,
                config,
                reference_pixel_size_um=reference_pixel_size_um,
                input_protocol=input_protocol,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=SourceGroupedBatchSampler(
                    val_rows,
                    batch_size=int(batch_size),
                    shuffle=False,
                    seed=int(seed),
                ),
                collate_fn=_collate,
                num_workers=int(num_workers),
            )
        else:
            val_dataset = PreparedCellCropDataset(val_rows, prepared_supervised_metadata, config)
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(batch_size),
                shuffle=False,
                collate_fn=_collate,
                num_workers=int(num_workers),
            )
    if task_type == _BINARY_CLASSIFICATION_TASK and config.encoder_architecture == 'shared_scale_aware':
        from crypt_villus_vit.peyer import PeyerClassifier
        model = PeyerClassifier(config)
    else:
        model = MultitaskDapiVit(config)
    initial_checkpoint_loaded = None
    if initial_checkpoint is not None:
        checkpoint_path = Path(initial_checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_config = payload.get("model_config")
        if checkpoint_config is None:
            raise ValueError(f"Initial checkpoint `{checkpoint_path}` is not a crypt-villus-vit/v1 checkpoint.")
        if dict(checkpoint_config) != asdict(config):
            raise ValueError(
                f"Initial checkpoint `{checkpoint_path}` model_config does not match requested training config."
            )
        state = payload.get("model_state_dict") or payload.get("state_dict")
        if state is None:
            raise ValueError(f"Initial checkpoint `{checkpoint_path}` has no model state.")
        incompatible = model.load_state_dict(state, strict=True)
        initial_checkpoint_loaded = {
            "path": str(checkpoint_path),
            "format": str(payload.get("format", "unknown")),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "source_metadata": dict(payload.get("metadata", {})),
        }
    model = model.to(resolved_device)
    pretrained_loaded = None
    if pretrained_checkpoint is not None:
        from crypt_villus_vit.pretrain import load_pretrained_encoders

        pretrained_loaded = load_pretrained_encoders(model, Path(pretrained_checkpoint), strict=True)
    freeze_summary = _apply_freeze_mode(model, freeze_mode)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying freeze_mode.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(learning_rate), weight_decay=1e-4)
    target_axis, target_epi, target_epi_mask = _target_tensors(train_rows)
    val_target_axis = val_target_epi = val_target_epi_mask = None
    if val_rows is not None and not val_rows.empty:
        val_target_axis, val_target_epi, val_target_epi_mask = _target_tensors(val_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Supervised training split | "
        f"train_rows={len(train_rows):,} | "
        f"validation_rows={0 if val_rows is None else len(val_rows):,} | "
        f"batch_size={int(batch_size)} | batching={batching_strategy} | "
        f"device={resolved_device} | freeze_mode={freeze_summary['freeze_mode']} | "
        f"task_type={task_type}",
        flush=True,
    )
    print(
        "Supervised crop scaling | "
        f"reference_pixel_size_um={None if reference_pixel_size_um is None else float(reference_pixel_size_um)}",
        flush=True,
    )
    print(
        "Supervised intensity augmentation | "
        f"enabled={intensity_augmentation is not None} | "
        f"settings={json.dumps(intensity_augmentation_summary, sort_keys=True)}",
        flush=True,
    )
    history: list[dict[str, float]] = []
    if task_type == _BINARY_CLASSIFICATION_TASK:
        best_metric_name = (
            "validation_binary_average_precision"
            if val_rows is not None and not val_rows.empty
            else "train_binary_average_precision"
        )
        best_metric_mode = "max"
        best_metric_value = -float("inf")
    else:
        best_metric_name = "validation_loss" if val_rows is not None and not val_rows.empty else "train_loss"
        best_metric_mode = "min"
        best_metric_value = float("inf")
    best_epoch = None
    best_checkpoint_path = output_dir / "best_crypt_villus_vit_model.pt"
    try:
        warmup_started = time.perf_counter()
        if ram_cache is not None and warm_crop_cache:
            warm_ram_cache(ram_cache, train_rows, val_rows, sources, config, num_workers=int(num_workers),
                           crop_backend=crop_backend, tile_cache_mib=tile_cache_mib,
                           reference_pixel_size_um=reference_pixel_size_um, input_protocol=input_protocol)
        cache_warmup_seconds = time.perf_counter() - warmup_started
        for epoch in range(int(epochs)):
            print(f"Epoch {epoch + 1}/{int(epochs)} started.", flush=True)
            _set_dataset_epoch(loader, epoch)
            train_started = time.perf_counter()
            train_metrics = _run_epoch(
                model,
                loader,
                target_axis=target_axis,
                target_epi=target_epi,
                target_epi_mask=target_epi_mask,
                device=resolved_device,
                task_type=task_type,
                binary_positive_class_weight=resolved_positive_class_weight,
                classification_threshold=float(classification_threshold),
                optimizer=optimizer,
                gradient_clip_norm=gradient_clip_norm,
                desc=f"train epoch {epoch + 1}",
            )
            row = {"epoch": float(epoch + 1), "train_seconds": time.perf_counter() - train_started}
            row.update({f"train_{key}" if key != "loss" else "train_loss": value for key, value in train_metrics.items()})
            if (
                val_loader is not None
                and val_target_axis is not None
                and val_target_epi is not None
                and val_target_epi_mask is not None
            ):
                _set_dataset_epoch(val_loader, epoch)
                validation_started = time.perf_counter()
                val_metrics = _run_epoch(
                    model,
                    val_loader,
                    target_axis=val_target_axis,
                    target_epi=val_target_epi,
                    target_epi_mask=val_target_epi_mask,
                    device=resolved_device,
                    task_type=task_type,
                    binary_positive_class_weight=resolved_positive_class_weight,
                    classification_threshold=float(classification_threshold),
                    desc=f"validation epoch {epoch + 1}",
                )
                row.update(
                    {f"validation_{key}" if key != "loss" else "validation_loss": value for key, value in val_metrics.items()}
                )
                row["validation_seconds"] = time.perf_counter() - validation_started
            history.append(row)
            print("Epoch metrics: " + json.dumps(row, sort_keys=True), flush=True)
            pd.DataFrame(history).to_csv(output_dir / "history.partial.csv", index=False)
            selection_value = float(row[best_metric_name])
            improved = selection_value > best_metric_value if best_metric_mode == "max" else selection_value < best_metric_value
            if improved:
                best_metric_value = selection_value
                best_epoch = int(epoch + 1)
                save_model_checkpoint(
                    model,
                    best_checkpoint_path,
                    metadata={
                        "input_protocol": input_protocol,
                        "epochs_completed": int(epoch + 1),
                        "best_epoch": int(best_epoch),
                        "selection_metric": best_metric_name,
                        "selection_mode": best_metric_mode,
                        "selection_value": float(best_metric_value),
                        "task_type": task_type,
                        "row_count": int(len(rows)),
                        "train_row_count": int(len(train_rows)),
                        "validation_row_count": int(0 if val_rows is None else len(val_rows)),
                        "target_column": str(target_axis_column),
                        "prediction_column": str(prediction_column),
                        "positive_class_weight": resolved_positive_class_weight,
                        "classification_threshold": float(classification_threshold),
                        "target_axis_column": str(target_axis_column),
                        "target_epithelial_column": str(target_epithelial_column),
                        "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
                        "initial_checkpoint_loaded": initial_checkpoint_loaded,
                        "pretrained_checkpoint": None if pretrained_checkpoint is None else str(pretrained_checkpoint),
                        "pretrained_loaded": pretrained_loaded,
                        "prepared_supervised_metadata": (
                            None if prepared_supervised_metadata is None else str(prepared_supervised_metadata)
                        ),
                        "reference_pixel_size_um": (
                            None if reference_pixel_size_um is None else float(reference_pixel_size_um)
                        ),
                        "batching_strategy": batching_strategy,
                        "data_loading": loading_settings,
                        "freeze_summary": freeze_summary,
                        "intensity_augmentation": intensity_augmentation_summary,
                        "gradient_clip_norm": (
                            None if gradient_clip_norm is None else float(gradient_clip_norm)
                        ),
                        "model_config": asdict(config),
                        "seed": int(seed),
                    },
                )
        checkpoint_path = output_dir / "crypt_villus_vit_model.pt"
        if best_epoch is None:
            raise RuntimeError("No finite checkpoint-selection metric was produced; no final model exported.")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=resolved_device, weights_only=True)['model_state_dict'], strict=True)
        metadata = {
            "input_protocol": input_protocol,
            "exported_weights": "best_selection_metric",
            "task_type": task_type,
            "epochs": int(epochs),
            "row_count": int(len(rows)),
            "train_row_count": int(len(train_rows)),
            "validation_row_count": int(0 if val_rows is None else len(val_rows)),
            "target_column": str(target_axis_column),
            "prediction_column": str(prediction_column),
            "positive_class_weight": resolved_positive_class_weight,
            "classification_threshold": float(classification_threshold),
            "target_axis_column": str(target_axis_column),
            "target_epithelial_column": str(target_epithelial_column),
            "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
            "initial_checkpoint_loaded": initial_checkpoint_loaded,
            "pretrained_checkpoint": None if pretrained_checkpoint is None else str(pretrained_checkpoint),
            "pretrained_loaded": pretrained_loaded,
            "prepared_supervised_metadata": (
                None if prepared_supervised_metadata is None else str(prepared_supervised_metadata)
            ),
            "reference_pixel_size_um": None if reference_pixel_size_um is None else float(reference_pixel_size_um),
            "batching_strategy": batching_strategy,
            "data_loading": loading_settings,
            "freeze_summary": freeze_summary,
            "prepared_supervised_summary": (
                None if prepared_supervised_metadata is None else prepared_summary(prepared_supervised_metadata)
            ),
            "intensity_augmentation": intensity_augmentation_summary,
            "gradient_clip_norm": None if gradient_clip_norm is None else float(gradient_clip_norm),
            "seed": int(seed),
            "best_checkpoint_path": str(best_checkpoint_path),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "selection_metric": best_metric_name,
            "selection_mode": best_metric_mode,
            "selection_value": None if best_epoch is None else float(best_metric_value),
        }
        save_model_checkpoint(model, checkpoint_path, metadata=metadata)
        history_csv = output_dir / "history.csv"
        pd.DataFrame(history).to_csv(history_csv, index=False)
        summary = {
            "task_type": task_type,
            "checkpoint_path": str(checkpoint_path),
            "history_csv": str(history_csv),
            "row_count": int(len(rows)),
            "train_row_count": int(len(train_rows)),
            "validation_row_count": int(0 if val_rows is None else len(val_rows)),
            "target_column": str(target_axis_column),
            "prediction_column": str(prediction_column),
            "positive_class_weight": resolved_positive_class_weight,
            "classification_threshold": float(classification_threshold),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "device": resolved_device,
            "model_config": asdict(config),
            "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
            "initial_checkpoint_loaded": initial_checkpoint_loaded,
            "pretrained_checkpoint": None if pretrained_checkpoint is None else str(pretrained_checkpoint),
            "pretrained_loaded": pretrained_loaded,
            "prepared_supervised_metadata": (
                None if prepared_supervised_metadata is None else str(prepared_supervised_metadata)
            ),
            "reference_pixel_size_um": None if reference_pixel_size_um is None else float(reference_pixel_size_um),
            "batching_strategy": batching_strategy,
            "data_loading": loading_settings,
            "cache_warmup_seconds": cache_warmup_seconds,
            "training_wall_seconds": time.perf_counter() - started,
            "freeze_summary": freeze_summary,
            "prepared_supervised_summary": (
                None if prepared_supervised_metadata is None else prepared_summary(prepared_supervised_metadata)
            ),
            "intensity_augmentation": intensity_augmentation_summary,
            "gradient_clip_norm": None if gradient_clip_norm is None else float(gradient_clip_norm),
            "best_checkpoint_path": str(best_checkpoint_path),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "selection_metric": best_metric_name,
            "selection_mode": best_metric_mode,
            "selection_value": None if best_epoch is None else float(best_metric_value),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_validation_loss": history[-1].get("validation_loss") if history else None,
        }
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        close_training_loader(loader)
        close_training_loader(val_loader)
