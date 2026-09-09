from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from crypt_villus_vit.augmentation import make_intensity_augmentation_config
from crypt_villus_vit.model import LegacyDapiVisionTransformer
from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.model import PatchEncoder
from crypt_villus_vit.prepared import PreparedPretrainingDataset
from crypt_villus_vit.prepared import prepared_pretraining_summary
from crypt_villus_vit.predict import CellCropDataset
from crypt_villus_vit.predict import _collate
from crypt_villus_vit.sources import SourceSpec
from crypt_villus_vit.train import prepare_supervised_manifest


class MaskedCropAutoencoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if not (config.use_local_branch or config.use_context_branch):
            raise ValueError("Pretraining requires at least one local or context branch.")
        self.local_encoder = (
            PatchEncoder(
                image_size=config.input_size_px,
                patch_size=config.patch_size_px,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                readout=config.local_readout,
            )
            if config.use_local_branch
            else None
        )
        self.context_encoder = (
            PatchEncoder(
                image_size=config.input_size_px,
                patch_size=config.patch_size_px,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                readout=config.context_readout,
            )
            if config.use_context_branch
            else None
        )
        output_dim = int(config.input_size_px * config.input_size_px)
        self.local_decoder = nn.Linear(config.embed_dim, output_dim) if self.local_encoder is not None else None
        self.context_decoder = nn.Linear(config.embed_dim, output_dim) if self.context_encoder is not None else None

    def _decode(self, encoded: torch.Tensor, decoder: nn.Linear) -> torch.Tensor:
        side = int(self.config.input_size_px)
        return torch.sigmoid(decoder(encoded)).reshape(encoded.shape[0], 1, side, side)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if self.local_encoder is not None and self.local_decoder is not None:
            outputs["local_image"] = self._decode(self.local_encoder(batch["local_image"]), self.local_decoder)
        if self.context_encoder is not None and self.context_decoder is not None:
            outputs["context_image"] = self._decode(self.context_encoder(batch["context_image"]), self.context_decoder)
        return outputs


class PreparedMaskedPatchAutoencoder(nn.Module):
    """Shared scale-aware encoder used with the written pretraining shards."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = LegacyDapiVisionTransformer(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(config.embed_dim)))
        patch_output_dim = int(config.patch_size_px * config.patch_size_px)
        self.decoder = nn.Sequential(
            nn.LayerNorm(int(config.embed_dim)),
            nn.Linear(int(config.embed_dim), patch_output_dim),
        )

    def _patch_targets(self, images: torch.Tensor) -> torch.Tensor:
        patch_size = int(self.config.patch_size_px)
        patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
        return patches.transpose(1, 2)

    def forward(
        self,
        images: torch.Tensor,
        scale_ids: torch.Tensor,
        *,
        mask_fraction: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_tokens = self.encoder.patch_embed(images).flatten(2).transpose(1, 2)
        batch_size, patch_count, _ = patch_tokens.shape
        mask = torch.rand((batch_size, patch_count), device=images.device) < float(mask_fraction)
        empty_mask_rows = ~mask.any(dim=1)
        if bool(empty_mask_rows.any()):
            mask[empty_mask_rows, 0] = True
        masked_tokens = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.expand(batch_size, patch_count, -1),
            patch_tokens,
        )
        cls_tokens = self.encoder.cls_token.expand(batch_size, -1, -1)
        scale_embedding = self.encoder.scale_embeddings[scale_ids.long()].unsqueeze(1)
        tokens = torch.cat([cls_tokens, masked_tokens], dim=1)
        tokens = tokens + self.encoder.pos_embed[:, : tokens.shape[1]] + scale_embedding
        encoded = self.encoder.norm(self.encoder.encoder(tokens))
        reconstructed_patches = self.decoder(encoded[:, 1:])
        return reconstructed_patches, self._patch_targets(images), mask


def _mask_images(images: torch.Tensor, mask_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(images) < float(mask_fraction)
    masked = images.masked_fill(mask, 0.0)
    return masked, mask


def _pretrain_payload(model: MaskedCropAutoencoder, *, metadata: dict[str, Any]) -> dict[str, Any]:
    encoder_state: dict[str, dict[str, torch.Tensor]] = {}
    if model.local_encoder is not None:
        encoder_state["local_encoder"] = model.local_encoder.state_dict()
    if model.context_encoder is not None:
        encoder_state["context_encoder"] = model.context_encoder.state_dict()
    return {
        "format": "crypt-villus-vit/pretrain-v1",
        "model_config": asdict(model.config),
        "encoder_state_dict": encoder_state,
        "autoencoder_state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }


def _is_legacy_shared_encoder_state(encoder_state: dict[str, Any]) -> bool:
    return {
        "patch_embed.weight",
        "patch_embed.bias",
        "pos_embed",
    }.issubset(encoder_state)


def _legacy_shared_encoder_to_branch_state(
    encoder_state: dict[str, torch.Tensor],
    branch: nn.Module,
) -> dict[str, torch.Tensor]:
    """Map the older shared scale-aware ViT encoder into one branch encoder."""

    target_state = branch.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, target_value in target_state.items():
        source_value: torch.Tensor | None = None
        if key == "patch.weight":
            source_value = encoder_state.get("patch_embed.weight")
        elif key == "patch.bias":
            source_value = encoder_state.get("patch_embed.bias")
        elif key == "positional_embedding":
            pos_embed = encoder_state.get("pos_embed")
            if pos_embed is not None:
                if int(pos_embed.shape[1]) == int(target_value.shape[1]) + 1:
                    source_value = pos_embed[:, 1:, :]
                elif tuple(pos_embed.shape) == tuple(target_value.shape):
                    source_value = pos_embed
        elif key == "scale_embeddings":
            source_value = encoder_state.get("scale_embeddings")
        elif key.startswith("encoder.") or key.startswith("norm."):
            source_value = encoder_state.get(key)
        if source_value is None:
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                f"Cannot map legacy encoder tensor `{key}`: source shape "
                f"{tuple(source_value.shape)} does not match target shape {tuple(target_value.shape)}."
            )
        mapped[key] = source_value.detach().clone()
    return mapped


def _load_branch_state(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    strict: bool,
) -> dict[str, object]:
    result = module.load_state_dict(state, strict=bool(strict))
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _prepared_pretrain_payload(
    model: PreparedMaskedPatchAutoencoder,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "crypt-villus-vit/pretrain-v1",
        "model_config": asdict(model.config),
        "encoder_state_dict": model.encoder.state_dict(),
        "pretrainer_state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }


def _split_pretraining_rows(
    rows: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    out = rows.reset_index(drop=True).copy()
    if "split" in out:
        split = out["split"].fillna("").astype(str).str.strip().str.lower()
        validation_mask = split.isin({"val", "valid", "validation"})
        train_mask = split.isin({"", "train", "training"})
        if bool(validation_mask.any()):
            train_rows = out.loc[train_mask].copy()
            validation_rows = out.loc[validation_mask].copy()
            if train_rows.empty:
                raise ValueError("Prepared pretraining split has validation rows but no training rows.")
            return train_rows.reset_index(drop=True), validation_rows.reset_index(drop=True)
    fraction = float(validation_fraction)
    if fraction <= 0:
        return out, None
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(out))
    rng.shuffle(indices)
    validation_count = max(1, min(len(out) - 1, int(round(len(out) * fraction))))
    return (
        out.iloc[indices[validation_count:]].reset_index(drop=True),
        out.iloc[indices[:validation_count]].reset_index(drop=True),
    )


def _pretraining_manifest_summary(rows: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "row_count": int(len(rows)),
        "prepared_index_unique_count": (
            int(rows["prepared_index"].nunique()) if "prepared_index" in rows else None
        ),
    }
    for column in ("split", "source_group", "scale_id"):
        if column in rows:
            summary[f"{column}_counts"] = (
                rows[column].fillna("").astype(str).value_counts().sort_index().to_dict()
            )
    return summary


def _run_prepared_pretraining_epoch(
    model: PreparedMaskedPatchAutoencoder,
    loader: DataLoader,
    *,
    device: str,
    mask_fraction: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
    desc: str,
) -> float:
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_images = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            scale_ids = batch["scale_id"].to(device, non_blocking=True)
            reconstruction, targets, mask = model(
                images,
                scale_ids,
                mask_fraction=float(mask_fraction),
            )
            loss = F.mse_loss(reconstruction[mask], targets[mask])
            if is_train:
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Pretraining loss became non-finite before the optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(gradient_clip_norm),
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        raise FloatingPointError(
                            "Pretraining gradient norm became non-finite before the optimizer step."
                        )
                optimizer.step()
            batch_size = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_images += batch_size
    return total_loss / max(total_images, 1)


def pretrain_prepared_image_encoder(
    rows: pd.DataFrame,
    *,
    prepared_pretraining_metadata: Path,
    output_dir: Path,
    config: ModelConfig | None = None,
    epochs: int = 35,
    batch_size: int = 8,
    learning_rate: float = 3e-6,
    weight_decay: float = 1e-4,
    mask_fraction: float = 0.40,
    validation_fraction: float = 0.10,
    device: str = "cuda",
    seed: int = 0,
    num_workers: int = 0,
    augment_intensity_probability: float = 0.0,
    augment_brightness_delta: float = 0.0,
    augment_contrast_range: tuple[float, float] = (1.0, 1.0),
    augment_gaussian_blur_sigma_range: tuple[float, float] = (0.0, 0.0),
    gradient_clip_norm: float | None = None,
) -> dict[str, object]:
    config = config or ModelConfig(use_fine_branch=False)
    resolved_device = "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when provided.")
    train_rows, validation_rows = _split_pretraining_rows(
        rows,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
    )
    training_manifest_summary = _pretraining_manifest_summary(rows)
    image_augmentation = make_intensity_augmentation_config(
        probability=float(augment_intensity_probability),
        brightness_delta=float(augment_brightness_delta),
        contrast_range=augment_contrast_range,
        gaussian_blur_sigma_range=augment_gaussian_blur_sigma_range,
        seed=int(seed),
    )
    image_augmentation_summary = None if image_augmentation is None else image_augmentation.to_dict()
    train_dataset = PreparedPretrainingDataset(
        train_rows,
        prepared_pretraining_metadata,
        seed=int(seed),
        cycle_variants=True,
        intensity_augmentation=image_augmentation,
    )
    validation_dataset = (
        None
        if validation_rows is None
        else PreparedPretrainingDataset(
            validation_rows,
            prepared_pretraining_metadata,
            seed=int(seed),
            cycle_variants=False,
        )
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": resolved_device == "cuda",
        # Workers must be recreated after set_epoch so deterministic augmentation
        # receives the current epoch in each worker process.
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = (
        None
        if validation_dataset is None
        else DataLoader(validation_dataset, shuffle=False, **loader_options)
    )
    model = PreparedMaskedPatchAutoencoder(config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "dapi_encoder_pretrain.pt"
    last_checkpoint_path = output_dir / "dapi_encoder_pretrain_last.pt"
    best_metric_name = "validation_loss" if validation_loader is not None else "train_loss"
    best_metric_value = float("inf")
    best_epoch = None
    history: list[dict[str, float]] = []
    started = time.time()
    print(
        "Prepared pretraining split | "
        f"train_rows={len(train_rows):,} | "
        f"validation_rows={0 if validation_rows is None else len(validation_rows):,} | "
        f"variants={train_dataset.arrays.variant_count} | "
        f"batch_size={int(batch_size)} | device={resolved_device}",
        flush=True,
    )
    for epoch in range(int(epochs)):
        train_dataset.set_epoch(epoch)
        epoch_started = time.time()
        train_loss = _run_prepared_pretraining_epoch(
            model,
            train_loader,
            device=resolved_device,
            mask_fraction=float(mask_fraction),
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
            desc=f"pretrain epoch {epoch + 1}",
        )
        row: dict[str, float] = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
        }
        if validation_loader is not None:
            validation_loss = _run_prepared_pretraining_epoch(
                model,
                validation_loader,
                device=resolved_device,
                mask_fraction=float(mask_fraction),
                optimizer=None,
                gradient_clip_norm=None,
                desc=f"pretrain validation {epoch + 1}",
            )
            row["validation_loss"] = float(validation_loss)
        row["epoch_minutes"] = float((time.time() - epoch_started) / 60.0)
        row["elapsed_minutes"] = float((time.time() - started) / 60.0)
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "pretrain_history.partial.csv", index=False)
        print("Pretraining epoch metrics: " + json.dumps(row, sort_keys=True), flush=True)
        metadata = {
            "epochs_completed": int(epoch + 1),
            "train_row_count": int(len(train_rows)),
            "validation_row_count": int(0 if validation_rows is None else len(validation_rows)),
            "prepared_pretraining_metadata": str(Path(prepared_pretraining_metadata)),
            "prepared_pretraining_summary": prepared_pretraining_summary(prepared_pretraining_metadata),
            "training_manifest_summary": training_manifest_summary,
            "mask_fraction": float(mask_fraction),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "image_augmentation": image_augmentation_summary,
            "gradient_clip_norm": (
                None if gradient_clip_norm is None else float(gradient_clip_norm)
            ),
            "seed": int(seed),
        }
        selection_value = float(row[best_metric_name])
        if selection_value < best_metric_value:
            best_metric_value = selection_value
            best_epoch = int(epoch + 1)
            best_metadata = dict(metadata)
            best_metadata.update(
                {
                    "best_epoch": int(best_epoch),
                    "selection_metric": best_metric_name,
                    "selection_value": float(best_metric_value),
                }
            )
            torch.save(_prepared_pretrain_payload(model, metadata=best_metadata), best_checkpoint_path)
        torch.save(_prepared_pretrain_payload(model, metadata=metadata), last_checkpoint_path)
    history_csv = output_dir / "pretrain_history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False)
    summary = {
        "checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "history_csv": str(history_csv),
        "row_count": int(len(rows)),
        "train_row_count": int(len(train_rows)),
        "validation_row_count": int(0 if validation_rows is None else len(validation_rows)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "mask_fraction": float(mask_fraction),
        "device": resolved_device,
        "model_config": asdict(config),
        "prepared_pretraining_summary": prepared_pretraining_summary(prepared_pretraining_metadata),
        "training_manifest_summary": training_manifest_summary,
        "image_augmentation": image_augmentation_summary,
        "gradient_clip_norm": None if gradient_clip_norm is None else float(gradient_clip_norm),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "selection_metric": best_metric_name,
        "selection_value": None if best_epoch is None else float(best_metric_value),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_validation_loss": history[-1].get("validation_loss") if history else None,
        "duration_minutes": float((time.time() - started) / 60.0),
    }
    (output_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def pretrain_image_encoders(
    rows: pd.DataFrame,
    *,
    sources: dict[str, SourceSpec],
    output_dir: Path,
    config: ModelConfig | None = None,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    mask_fraction: float = 0.30,
    validation_fraction: float = 0.0,
    device: str = "cuda",
    seed: int = 0,
    num_workers: int = 0,
    augment_intensity_probability: float = 0.0,
    augment_brightness_delta: float = 0.0,
    augment_contrast_range: tuple[float, float] = (1.0, 1.0),
    augment_gaussian_blur_sigma_range: tuple[float, float] = (0.0, 0.0),
    gradient_clip_norm: float | None = None,
    reference_pixel_size_um: float | None = None,
) -> dict[str, object]:
    config = config or ModelConfig(use_fine_branch=False)
    rows = prepare_supervised_manifest(rows)
    resolved_device = "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when provided.")
    train_rows, validation_rows = _split_pretraining_rows(
        rows,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
    )
    image_augmentation = make_intensity_augmentation_config(
        probability=float(augment_intensity_probability),
        brightness_delta=float(augment_brightness_delta),
        contrast_range=augment_contrast_range,
        gaussian_blur_sigma_range=augment_gaussian_blur_sigma_range,
        seed=int(seed),
    )
    image_augmentation_summary = None if image_augmentation is None else image_augmentation.to_dict()
    dataset = CellCropDataset(
        train_rows,
        sources,
        config,
        reference_pixel_size_um=reference_pixel_size_um,
        intensity_augmentation=image_augmentation,
    )
    validation_dataset = (
        None
        if validation_rows is None
        else CellCropDataset(
            validation_rows,
            sources,
            config,
            reference_pixel_size_um=reference_pixel_size_um,
        )
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=_collate,
        num_workers=int(num_workers),
        generator=generator,
    )
    validation_loader = (
        None
        if validation_dataset is None
        else DataLoader(
            validation_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=_collate,
            num_workers=int(num_workers),
        )
    )
    model = MaskedCropAutoencoder(config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(int(epochs)):
        dataset.set_epoch(epoch)
        epoch_started = time.time()
        total_loss = 0.0
        total_n = 0
        model.train()
        for batch in loader:
            batch.pop("index")
            targets = {key: value.to(resolved_device) for key, value in batch.items() if key.endswith("_image")}
            masked_batch = {}
            masks = {}
            for key, image in targets.items():
                if key not in {"local_image", "context_image"}:
                    continue
                masked_batch[key], masks[key] = _mask_images(image, mask_fraction)
            outputs = model(masked_batch)
            losses = []
            for key, reconstruction in outputs.items():
                target = targets[key]
                mask = masks[key]
                losses.append(F.mse_loss(reconstruction[mask], target[mask]) if bool(mask.any()) else F.mse_loss(reconstruction, target))
            loss = sum(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
            optimizer.step()
            batch_n = int(next(iter(targets.values())).shape[0])
            total_loss += float(loss.detach().cpu()) * batch_n
            total_n += batch_n
        row = {
            "epoch": float(epoch + 1),
            "train_loss": total_loss / max(total_n, 1),
            "epoch_minutes": float((time.time() - epoch_started) / 60.0),
            "elapsed_minutes": float((time.time() - started) / 60.0),
        }
        if validation_loader is not None:
            validation_loss = 0.0
            validation_n = 0
            model.eval()
            with torch.no_grad():
                for batch in validation_loader:
                    batch.pop("index")
                    targets = {key: value.to(resolved_device) for key, value in batch.items() if key.endswith("_image")}
                    masked_batch = {}
                    masks = {}
                    for key, image in targets.items():
                        if key not in {"local_image", "context_image"}:
                            continue
                        masked_batch[key], masks[key] = _mask_images(image, mask_fraction)
                    outputs = model(masked_batch)
                    losses = []
                    for key, reconstruction in outputs.items():
                        target = targets[key]
                        mask = masks[key]
                        losses.append(F.mse_loss(reconstruction[mask], target[mask]) if bool(mask.any()) else F.mse_loss(reconstruction, target))
                    loss = sum(losses)
                    batch_n = int(next(iter(targets.values())).shape[0])
                    validation_loss += float(loss.detach().cpu()) * batch_n
                    validation_n += batch_n
            row["validation_loss"] = validation_loss / max(validation_n, 1)
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "pretrain_history.partial.csv", index=False)
    checkpoint_path = output_dir / "dapi_encoder_pretrain.pt"
    metadata = {
        "epochs": int(epochs),
        "row_count": int(len(rows)),
        "train_row_count": int(len(train_rows)),
        "validation_row_count": int(0 if validation_rows is None else len(validation_rows)),
        "mask_fraction": float(mask_fraction),
        "image_augmentation": image_augmentation_summary,
        "gradient_clip_norm": None if gradient_clip_norm is None else float(gradient_clip_norm),
        "reference_pixel_size_um": None if reference_pixel_size_um is None else float(reference_pixel_size_um),
        "seed": int(seed),
    }
    torch.save(_pretrain_payload(model, metadata=metadata), checkpoint_path)
    history_csv = output_dir / "pretrain_history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_csv),
        "row_count": int(len(rows)),
        "train_row_count": int(len(train_rows)),
        "validation_row_count": int(0 if validation_rows is None else len(validation_rows)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "device": resolved_device,
        "model_config": asdict(config),
        "image_augmentation": image_augmentation_summary,
        "gradient_clip_norm": None if gradient_clip_norm is None else float(gradient_clip_norm),
        "reference_pixel_size_um": None if reference_pixel_size_um is None else float(reference_pixel_size_um),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_validation_loss": history[-1].get("validation_loss") if history else None,
    }
    (output_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_pretrained_encoders(model: nn.Module, pretrained_checkpoint: Path, *, strict: bool = False) -> dict[str, object]:
    payload = torch.load(Path(pretrained_checkpoint), map_location="cpu")
    encoder_state = payload.get("encoder_state_dict")
    if not isinstance(encoder_state, dict):
        raise ValueError(f"{pretrained_checkpoint} does not contain encoder_state_dict.")
    loaded: dict[str, object] = {}
    if _is_legacy_shared_encoder_state(encoder_state):
        shared_encoder = getattr(model, "shared_encoder", None)
        if isinstance(shared_encoder, LegacyDapiVisionTransformer):
            loaded["shared_encoder"] = _load_branch_state(
                shared_encoder, encoder_state, strict=bool(strict)
            )
            loaded["shared_encoder"]["source_format"] = "legacy_shared_vit_exact"
        for name in ("local_encoder", "context_encoder"):
            module = getattr(model, name, None)
            if module is None:
                continue
            if isinstance(module, LegacyDapiVisionTransformer):
                state = encoder_state
                source_format = "legacy_shared_vit_exact_copy"
            else:
                state = _legacy_shared_encoder_to_branch_state(encoder_state, module)
                source_format = "legacy_shared_vit"
            loaded[name] = _load_branch_state(module, state, strict=bool(strict))
            loaded[name]["source_format"] = source_format
    else:
        for name in ("local_encoder", "context_encoder"):
            module = getattr(model, name, None)
            state = encoder_state.get(name)
            if module is None or state is None:
                continue
            loaded[name] = _load_branch_state(module, state, strict=bool(strict))
            loaded[name]["source_format"] = "pretrain_v1"
    if not loaded:
        raise ValueError(f"No compatible encoders were loaded from {pretrained_checkpoint}.")
    return loaded


def materialize_pretrained_encoder_checkpoint(
    *,
    source_checkpoint: Path,
    output_dir: Path,
    config: ModelConfig | None = None,
    source_history_csv: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Write a current-format pretraining checkpoint from an existing encoder checkpoint."""

    config = config or ModelConfig(use_fine_branch=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = MaskedCropAutoencoder(config)
    loaded = load_pretrained_encoders(model, Path(source_checkpoint), strict=False)
    checkpoint_path = output_dir / "dapi_encoder_pretrain.pt"
    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "materialized_only": True,
            "source_checkpoint": str(Path(source_checkpoint)),
            "loaded": loaded,
        }
    )
    torch.save(_pretrain_payload(model, metadata=payload_metadata), checkpoint_path)

    history_csv = output_dir / "pretrain_history.csv"
    final_pretrain_loss = None
    final_validation_loss = None
    if source_history_csv is not None and Path(source_history_csv).exists():
        history = pd.read_csv(source_history_csv)
        keep_columns = [
            column
            for column in (
                "epoch",
                "train_loss",
                "validation_loss",
                "used_image_count",
                "validation_used_image_count",
            )
            if column in history.columns
        ]
        history = history[keep_columns].copy() if keep_columns else history.copy()
        history.to_csv(history_csv, index=False)
        if "train_loss" in history.columns and not history.empty:
            final_pretrain_loss = float(history["train_loss"].iloc[-1])
        if "validation_loss" in history.columns and not history.empty:
            final_validation_loss = float(history["validation_loss"].iloc[-1])
    else:
        pd.DataFrame([{"materialized_only": True}]).to_csv(history_csv, index=False)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_csv),
        "source_checkpoint": str(Path(source_checkpoint)),
        "source_history_csv": None if source_history_csv is None else str(Path(source_history_csv)),
        "materialized_only": True,
        "loaded": loaded,
        "device": "cpu",
        "model_config": asdict(config),
        "final_pretrain_loss": final_pretrain_loss,
        "final_validation_loss": final_validation_loss,
    }
    (output_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
