from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
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
from crypt_villus_vit.prepared import PreparedPairedPretrainingDataset
from crypt_villus_vit.prepared import prepared_pretraining_summary


class ProjectionHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = int(matrix.shape[0])
    if matrix.ndim != 2 or int(matrix.shape[1]) != size:
        raise ValueError("Covariance matrix must be square.")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


class PairedScaleTeacherStudentPretrainer(nn.Module):
    """Masked paired-scale feature prediction with an EMA teacher and VICReg guards."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        readout: str = "center_4x",
        projection_hidden_dim: int = 512,
        projection_dim: int = 256,
        cross_scale_weight: float = 0.50,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.04,
    ):
        super().__init__()
        if not 0.0 <= float(cross_scale_weight) <= 1.0:
            raise ValueError("cross_scale_weight must be in [0, 1].")
        self.config = config
        self.readout = str(readout)
        self.cross_scale_weight = float(cross_scale_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.student_encoder = LegacyDapiVisionTransformer(config)
        self.student_projector = ProjectionHead(
            input_dim=int(config.embed_dim),
            hidden_dim=int(projection_hidden_dim),
            output_dim=int(projection_dim),
        )
        self.student_predictor = ProjectionHead(
            input_dim=int(projection_dim),
            hidden_dim=int(projection_hidden_dim),
            output_dim=int(projection_dim),
        )
        self.teacher_encoder = deepcopy(self.student_encoder)
        self.teacher_projector = deepcopy(self.student_projector)
        for parameter in self.teacher_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.teacher_projector.parameters():
            parameter.requires_grad = False
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(config.embed_dim)))
        nn.init.normal_(self.mask_token, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_projector.eval()
        return self

    def student_parameters(self):
        yield from self.student_encoder.parameters()
        yield from self.student_projector.parameters()
        yield from self.student_predictor.parameters()
        yield self.mask_token

    @staticmethod
    def _stable_self_attention(
        layer: nn.TransformerEncoderLayer, normalized_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Equivalent explicit MHA path that avoids unstable fused CUDA SDPA."""
        attention = layer.self_attn
        if not bool(attention._qkv_same_embed_dim):
            raise ValueError("Stable student attention requires a shared Q/K/V embedding size.")
        projected = F.linear(
            normalized_tokens, attention.in_proj_weight, attention.in_proj_bias
        )
        projected = torch.nan_to_num(
            projected, nan=0.0, posinf=100.0, neginf=-100.0
        ).clamp(min=-100.0, max=100.0)
        query, key, value = projected.chunk(3, dim=-1)
        batch_size, token_count, embed_dim = query.shape
        head_count = int(attention.num_heads)
        head_dim = int(embed_dim) // head_count

        def split_heads(values: torch.Tensor) -> torch.Tensor:
            return values.reshape(batch_size, token_count, head_count, head_dim).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(head_dim))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        logits = logits.clamp(min=-50.0, max=0.0)
        weights = logits.exp()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if float(attention.dropout) > 0.0:
            weights = F.dropout(
                weights, p=float(attention.dropout), training=bool(layer.training)
            )
        attended = torch.matmul(weights, value)
        attended = torch.nan_to_num(
            attended, nan=0.0, posinf=100.0, neginf=-100.0
        ).clamp(min=-100.0, max=100.0)
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, embed_dim)
        attended = F.linear(
            attended, attention.out_proj.weight, attention.out_proj.bias
        )
        attended = torch.nan_to_num(
            attended, nan=0.0, posinf=100.0, neginf=-100.0
        ).clamp(min=-100.0, max=100.0)
        return layer.dropout1(attended)

    @staticmethod
    def _stable_layer_norm(layer: nn.LayerNorm, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1, keepdim=True)
        centered = values - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + float(layer.eps))
        if layer.elementwise_affine:
            normalized = normalized * layer.weight
            if layer.bias is not None:
                normalized = normalized + layer.bias
        # A correctly computed D-dimensional LayerNorm vector is naturally
        # O(sqrt(D)); this wide guard is inactive for valid values and contains
        # rare CUDA arithmetic outliers before they contaminate attention.
        return torch.nan_to_num(
            normalized, nan=0.0, posinf=100.0, neginf=-100.0
        ).clamp(min=-100.0, max=100.0)

    def _student_readout(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.readout == "cls":
            values = encoded[:, 0]
        else:
            patch_tokens = encoded[:, 1:]
            if self.readout == "mean":
                values = patch_tokens.mean(dim=1)
            else:
                token_count = int(patch_tokens.shape[1])
                grid_size = int(math.isqrt(token_count))
                if grid_size * grid_size != token_count:
                    raise ValueError(
                        f"Center readout requires a square token grid; got {token_count} tokens."
                    )
                if self.readout == "center":
                    window_size = 1 if grid_size % 2 else 2
                elif self.readout.startswith("center_") and self.readout.endswith("x"):
                    window_size = int(
                        self.readout.removeprefix("center_").removesuffix("x")
                    )
                else:
                    raise ValueError(f"Unsupported student readout: {self.readout}.")
                start = (grid_size - window_size) // 2
                grid = patch_tokens.reshape(
                    patch_tokens.shape[0], grid_size, grid_size, patch_tokens.shape[2]
                )
                values = grid[
                    :, start : start + window_size, start : start + window_size, :
                ].mean(dim=(1, 2))
        return self._stable_layer_norm(self.student_encoder.norm, values)

    def _student_features(
        self,
        images: torch.Tensor,
        *,
        scale_id: int,
        mask_fraction: float,
        mask_generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_tokens = self.student_encoder.patch_embed(images).flatten(2).transpose(1, 2)
        if not bool(torch.isfinite(patch_tokens).all()):
            raise FloatingPointError(
                f"Non-finite student patch tokens at scale {scale_id}; "
                f"input_range=({float(images.min())}, {float(images.max())})."
            )
        batch_size, patch_count, _ = patch_tokens.shape
        mask = torch.rand(
            (batch_size, patch_count),
            device=images.device,
            generator=mask_generator,
        ) < float(mask_fraction)
        empty_rows = ~mask.any(dim=1)
        if bool(empty_rows.any()):
            mask[empty_rows, 0] = True
        full_rows = mask.all(dim=1)
        if bool(full_rows.any()):
            mask[full_rows, 0] = False
        masked_tokens = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.expand(batch_size, patch_count, -1),
            patch_tokens,
        )
        cls_tokens = self.student_encoder.cls_token.expand(batch_size, -1, -1)
        scale_ids = torch.full(
            (batch_size,), int(scale_id), dtype=torch.long, device=images.device
        )
        scale_embedding = self.student_encoder.scale_embeddings[scale_ids].unsqueeze(1)
        tokens = torch.cat([cls_tokens, masked_tokens], dim=1)
        tokens = tokens + self.student_encoder.pos_embed[:, : tokens.shape[1]] + scale_embedding
        if not bool(torch.isfinite(tokens).all()):
            raise FloatingPointError(f"Non-finite assembled student tokens at scale {scale_id}.")
        encoded = tokens
        for layer_index, layer in enumerate(self.student_encoder.encoder.layers):
            layer_input_max = float(encoded.detach().abs().max())
            normalized_attention_input = self._stable_layer_norm(layer.norm1, encoded)
            if not bool(torch.isfinite(normalized_attention_input).all()):
                raise FloatingPointError(
                    f"Non-finite student attention norm at scale {scale_id}, layer {layer_index}; "
                    f"layer_input_max_abs={layer_input_max}."
                )
            attention_output = self._stable_self_attention(
                layer, normalized_attention_input
            )
            if not bool(torch.isfinite(attention_output).all()):
                raise FloatingPointError(
                    f"Non-finite student self-attention at scale {scale_id}, layer {layer_index}; "
                    f"layer_input_max_abs={layer_input_max}, "
                    f"normalized_input_max_abs={float(normalized_attention_input.detach().abs().max())}."
                )
            encoded = encoded + attention_output
            normalized_feedforward_input = self._stable_layer_norm(layer.norm2, encoded)
            if not bool(torch.isfinite(normalized_feedforward_input).all()):
                raise FloatingPointError(
                    f"Non-finite student feed-forward norm at scale {scale_id}, layer {layer_index}; "
                    f"residual_max_abs={float(encoded.detach().abs().max())}."
                )
            feedforward_output = layer._ff_block(normalized_feedforward_input)
            feedforward_output = torch.nan_to_num(
                feedforward_output, nan=0.0, posinf=100.0, neginf=-100.0
            ).clamp(min=-100.0, max=100.0)
            if not bool(torch.isfinite(feedforward_output).all()):
                raise FloatingPointError(
                    f"Non-finite student feed-forward block at scale {scale_id}, layer {layer_index}; "
                    f"normalized_input_max_abs={float(normalized_feedforward_input.detach().abs().max())}."
                )
            encoded = encoded + feedforward_output
        if self.student_encoder.encoder.norm is not None:
            encoded = self._stable_layer_norm(self.student_encoder.encoder.norm, encoded)
        features = self._student_readout(encoded)
        if not bool(torch.isfinite(features).all()):
            raise FloatingPointError(f"Non-finite student readout at scale {scale_id}.")
        return features, mask

    @staticmethod
    def _alignment(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # A practical floor keeps the cosine gradient bounded if a predictor
        # briefly approaches the origin early in training.
        prediction = F.normalize(prediction, dim=1, eps=1e-4)
        target = F.normalize(target.detach(), dim=1, eps=1e-4)
        return (2.0 - 2.0 * (prediction * target).sum(dim=1)).mean()

    @staticmethod
    def _variance_loss(features: torch.Tensor) -> torch.Tensor:
        if int(features.shape[0]) <= 1:
            return features.new_zeros(())
        standard_deviation = torch.sqrt(features.var(dim=0, unbiased=True) + 1e-4)
        return F.relu(1.0 - standard_deviation).mean()

    @staticmethod
    def _covariance_loss(features: torch.Tensor) -> torch.Tensor:
        batch_size, dimension = features.shape
        if int(batch_size) <= 1:
            return features.new_zeros(())
        centered = features - features.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / float(batch_size - 1)
        return _off_diagonal(covariance).pow(2).sum() / float(dimension)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        mask_fraction: float,
        mask_generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        local_student_features, local_mask = self._student_features(
            batch["local_student_image"],
            scale_id=0,
            mask_fraction=float(mask_fraction),
            mask_generator=mask_generator,
        )
        context_student_features, context_mask = self._student_features(
            batch["context_student_image"],
            scale_id=1,
            mask_fraction=float(mask_fraction),
            mask_generator=mask_generator,
        )
        local_student_projection = self.student_projector(local_student_features)
        context_student_projection = self.student_projector(context_student_features)
        local_prediction = self.student_predictor(local_student_projection)
        context_prediction = self.student_predictor(context_student_projection)
        with torch.no_grad():
            batch_size = int(batch["local_teacher_image"].shape[0])
            device = batch["local_teacher_image"].device
            local_scale_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            context_scale_ids = torch.ones(batch_size, dtype=torch.long, device=device)
            local_teacher_features = self.teacher_encoder.encode_image(
                batch["local_teacher_image"], local_scale_ids, readout=self.readout
            )
            context_teacher_features = self.teacher_encoder.encode_image(
                batch["context_teacher_image"], context_scale_ids, readout=self.readout
            )
            local_teacher_projection = self.teacher_projector(local_teacher_features)
            context_teacher_projection = self.teacher_projector(context_teacher_features)

        same_scale_loss = 0.5 * (
            self._alignment(local_prediction, local_teacher_projection)
            + self._alignment(context_prediction, context_teacher_projection)
        )
        cross_scale_loss = 0.5 * (
            self._alignment(local_prediction, context_teacher_projection)
            + self._alignment(context_prediction, local_teacher_projection)
        )
        alignment_loss = (
            (1.0 - self.cross_scale_weight) * same_scale_loss
            + self.cross_scale_weight * cross_scale_loss
        )
        variance_loss = 0.5 * (
            self._variance_loss(local_student_projection)
            + self._variance_loss(context_student_projection)
        )
        covariance_loss = 0.5 * (
            self._covariance_loss(local_student_projection)
            + self._covariance_loss(context_student_projection)
        )
        loss = (
            alignment_loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
        )
        feature_std = 0.5 * (
            local_student_projection.std(dim=0, unbiased=False).mean()
            + context_student_projection.std(dim=0, unbiased=False).mean()
        )
        prediction_norm_min = torch.minimum(
            local_prediction.norm(dim=1).min(), context_prediction.norm(dim=1).min()
        )
        teacher_norm_min = torch.minimum(
            local_teacher_projection.norm(dim=1).min(),
            context_teacher_projection.norm(dim=1).min(),
        )
        return {
            "loss": loss,
            "alignment_loss": alignment_loss,
            "same_scale_loss": same_scale_loss,
            "cross_scale_loss": cross_scale_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
            "feature_std": feature_std,
            "prediction_norm_min": prediction_norm_min,
            "teacher_norm_min": teacher_norm_min,
            "mask_fraction": 0.5 * (local_mask.float().mean() + context_mask.float().mean()),
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        resolved_momentum = float(momentum)
        for teacher, student in zip(
            self.teacher_encoder.parameters(), self.student_encoder.parameters(), strict=True
        ):
            teacher.data.mul_(resolved_momentum).add_(student.data, alpha=1.0 - resolved_momentum)
        for teacher, student in zip(
            self.teacher_projector.parameters(), self.student_projector.parameters(), strict=True
        ):
            teacher.data.mul_(resolved_momentum).add_(student.data, alpha=1.0 - resolved_momentum)


def split_paired_pretraining_rows(
    rows: pd.DataFrame,
    *,
    validation_fraction: float,
    group_level: str = "section_id",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    table = rows.reset_index(drop=True).copy()
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    for column in ("source_group", group_level):
        if column not in table:
            raise ValueError(f"Pretraining rows require `{column}` for grouped validation.")
    rng = np.random.default_rng(int(seed))
    validation_mask = np.zeros(len(table), dtype=bool)
    selected: dict[str, list[str]] = {}
    for source_group, source_rows in table.groupby("source_group", sort=True, dropna=False):
        groups = np.asarray(
            sorted(source_rows[str(group_level)].astype(str).unique()), dtype=object
        )
        if len(groups) < 2:
            continue
        count = max(1, int(round(len(groups) * float(validation_fraction))))
        count = min(count, len(groups) - 1)
        held_out = sorted(rng.choice(groups, size=count, replace=False).tolist())
        selected[str(source_group)] = held_out
        validation_mask[source_rows.index.to_numpy()] = (
            source_rows[str(group_level)].astype(str).isin(held_out).to_numpy()
        )
    train_rows = table.loc[~validation_mask].reset_index(drop=True)
    validation_rows = table.loc[validation_mask].reset_index(drop=True)
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Grouped pretraining split produced an empty partition.")
    return train_rows, validation_rows, selected


def _scheduled_learning_rate(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(base_learning_rate) * float(step + 1) / float(warmup_steps)
    denominator = max(total_steps - warmup_steps - 1, 1)
    progress = min(max(float(step - warmup_steps) / float(denominator), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(minimum_learning_rate) + (
        float(base_learning_rate) - float(minimum_learning_rate)
    ) * cosine


def _scheduled_teacher_momentum(
    *, step: int, total_steps: int, base_momentum: float
) -> float:
    progress = min(max(float(step) / float(max(total_steps - 1, 1)), 0.0), 1.0)
    return 1.0 - (1.0 - float(base_momentum)) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _checkpoint_payload(
    model: PairedScaleTeacherStudentPretrainer,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "crypt-villus-vit/paired-teacher-student-pretrain-v1",
        "model_config": asdict(model.config),
        "encoder_state_dict": model.teacher_encoder.state_dict(),
        "student_encoder_state_dict": model.student_encoder.state_dict(),
        "pretrainer_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "metadata": dict(metadata),
    }


def _run_epoch(
    model: PairedScaleTeacherStudentPretrainer,
    loader: DataLoader,
    *,
    device: str,
    mask_fraction: float,
    mask_generator: torch.Generator,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_value: float | None,
    gradient_clip_norm: float | None,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    learning_rate: float,
    minimum_learning_rate: float,
    teacher_momentum: float,
    desc: str,
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    totals: dict[str, float] = {}
    total_pairs = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=desc, unit="batch")
        for batch_index, batch in enumerate(progress):
            batch.pop("index")
            tensor_batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            current_lr = float(learning_rate)
            current_momentum = float(teacher_momentum)
            if is_train:
                current_lr = _scheduled_learning_rate(
                    step=global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    base_learning_rate=float(learning_rate),
                    minimum_learning_rate=float(minimum_learning_rate),
                )
                current_momentum = _scheduled_teacher_momentum(
                    step=global_step,
                    total_steps=total_steps,
                    base_momentum=float(teacher_momentum),
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = current_lr
            outputs = model(
                tensor_batch,
                mask_fraction=float(mask_fraction),
                mask_generator=mask_generator,
            )
            loss = outputs["loss"]
            if is_train:
                if not bool(torch.isfinite(loss)):
                    diagnostics: dict[str, Any] = {
                        key: float(value.detach().cpu()) for key, value in outputs.items()
                    }
                    diagnostics["batch_index"] = int(batch_index)
                    diagnostics["global_step"] = int(global_step)
                    diagnostics["learning_rate"] = current_lr
                    diagnostics["teacher_momentum"] = current_momentum
                    diagnostics["nonfinite_inputs"] = [
                        key
                        for key, value in tensor_batch.items()
                        if not bool(torch.isfinite(value).all())
                    ]
                    diagnostics["nonfinite_parameters"] = [
                        name
                        for name, parameter in model.named_parameters()
                        if not bool(torch.isfinite(parameter).all())
                    ][:20]
                    raise FloatingPointError(
                        "Teacher-student pretraining loss became non-finite: "
                        + json.dumps(diagnostics, sort_keys=True)
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                student_parameters = list(model.student_parameters())
                if gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_value_(
                        student_parameters,
                        clip_value=float(gradient_clip_value),
                        foreach=False,
                    )
                if gradient_clip_norm is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        student_parameters,
                        max_norm=float(gradient_clip_norm),
                        foreach=False,
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        diagnostics = {
                            key: float(value.detach().cpu()) for key, value in outputs.items()
                        }
                        diagnostics.update(
                            {
                                "batch_index": int(batch_index),
                                "global_step": int(global_step),
                                "learning_rate": current_lr,
                                "teacher_momentum": current_momentum,
                            }
                        )
                        raise FloatingPointError(
                            "Teacher-student gradient norm became non-finite: "
                            + json.dumps(diagnostics, sort_keys=True)
                        )
                optimizer.step()
                model.update_teacher(current_momentum)
                global_step += 1
            batch_size = int(next(iter(tensor_batch.values())).shape[0])
            for key, value in outputs.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            totals["learning_rate"] = totals.get("learning_rate", 0.0) + current_lr * batch_size
            totals["teacher_momentum"] = (
                totals.get("teacher_momentum", 0.0) + current_momentum * batch_size
            )
            total_pairs += batch_size
            if batch_index % 10 == 0 or batch_index + 1 == len(loader):
                progress.set_postfix(
                    loss=f"{float(outputs['loss'].detach()):.3f}",
                    align=f"{float(outputs['alignment_loss'].detach()):.3f}",
                    std=f"{float(outputs['feature_std'].detach()):.3f}",
                    lr=f"{current_lr:.2e}",
                )
    metrics = {key: value / max(total_pairs, 1) for key, value in totals.items()}
    metrics["pair_count"] = float(total_pairs)
    return metrics, global_step


def pretrain_paired_scale_teacher_student(
    rows: pd.DataFrame,
    *,
    prepared_pretraining_metadata: Path,
    output_dir: Path,
    config: ModelConfig | None = None,
    epochs: int = 35,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    minimum_learning_rate: float = 1e-6,
    warmup_epochs: int = 3,
    weight_decay: float = 0.04,
    mask_fraction: float = 0.40,
    validation_fraction: float = 0.10,
    validation_group_level: str = "section_id",
    readout: str = "center_4x",
    projection_hidden_dim: int = 512,
    projection_dim: int = 256,
    cross_scale_weight: float = 0.50,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
    teacher_momentum: float = 0.996,
    device: str = "cuda",
    seed: int = 7,
    num_workers: int = 0,
    augment_intensity_probability: float = 0.80,
    augment_brightness_delta: float = 0.10,
    augment_contrast_range: tuple[float, float] = (0.70, 1.30),
    augment_gaussian_blur_sigma_range: tuple[float, float] = (0.0, 1.5),
    gradient_clip_value: float | None = 1.0,
    gradient_clip_norm: float | None = 1.0,
    resume: bool = True,
) -> dict[str, Any]:
    config = config or ModelConfig(use_fine_branch=False)
    resolved_device = "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        # The fused SDPA path can intermittently return non-finite activations
        # for the masked norm-first transformer on Ampere. The math kernel is
        # slower but stable and leaves the architecture/objective unchanged.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    train_rows, validation_rows, validation_groups = split_paired_pretraining_rows(
        rows,
        validation_fraction=float(validation_fraction),
        group_level=str(validation_group_level),
        seed=int(seed),
    )
    augmentation = make_intensity_augmentation_config(
        probability=float(augment_intensity_probability),
        brightness_delta=float(augment_brightness_delta),
        contrast_range=augment_contrast_range,
        gaussian_blur_sigma_range=augment_gaussian_blur_sigma_range,
        seed=int(seed),
    )
    train_dataset = PreparedPairedPretrainingDataset(
        train_rows,
        prepared_pretraining_metadata,
        seed=int(seed),
        cycle_student_variants=True,
        student_intensity_augmentation=augmentation,
    )
    validation_dataset = PreparedPairedPretrainingDataset(
        validation_rows,
        prepared_pretraining_metadata,
        seed=int(seed),
        cycle_student_variants=False,
    )
    loader_options = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": resolved_device == "cuda",
        "persistent_workers": False,
    }
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=shuffle_generator, **loader_options
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    model = PairedScaleTeacherStudentPretrainer(
        config,
        readout=str(readout),
        projection_hidden_dim=int(projection_hidden_dim),
        projection_dim=int(projection_dim),
        cross_scale_weight=float(cross_scale_weight),
        variance_weight=float(variance_weight),
        covariance_weight=float(covariance_weight),
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        list(model.student_parameters()),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        foreach=False,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "dapi_encoder_pretrain.pt"
    last_checkpoint = output_dir / "dapi_encoder_pretrain_last.pt"
    history_path = output_dir / "pretrain_history.csv"
    history: list[dict[str, float]] = []
    starting_epoch = 0
    best_epoch = None
    best_validation_loss = float("inf")
    global_step = 0
    if bool(resume) and last_checkpoint.exists():
        payload = torch.load(last_checkpoint, map_location="cpu")
        if dict(payload.get("model_config", {})) != asdict(config):
            raise ValueError("Resume checkpoint model configuration does not match.")
        model.load_state_dict(payload["pretrainer_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(resolved_device)
        starting_epoch = int(payload.get("epoch", 0))
        global_step = starting_epoch * len(train_loader)
        metadata = dict(payload.get("metadata", {}))
        prior_best = metadata.get("best_validation_loss")
        if prior_best is not None:
            best_validation_loss = float(prior_best)
        prior_best_epoch = metadata.get("best_epoch")
        if prior_best_epoch is not None:
            best_epoch = int(prior_best_epoch)
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict(orient="records")
        print(f"Resuming paired-scale pretraining from epoch {starting_epoch}.", flush=True)
    total_steps = int(epochs) * len(train_loader)
    warmup_steps = int(warmup_epochs) * len(train_loader)
    summary_base = {
        "objective": "masked_paired_scale_ema_feature_prediction",
        "readout": str(readout),
        "train_pair_count": len(train_dataset),
        "validation_pair_count": len(validation_dataset),
        "validation_groups": validation_groups,
        "prepared_pretraining_metadata": str(Path(prepared_pretraining_metadata)),
        "prepared_pretraining_summary": prepared_pretraining_summary(
            prepared_pretraining_metadata
        ),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "minimum_learning_rate": float(minimum_learning_rate),
        "warmup_epochs": int(warmup_epochs),
        "weight_decay": float(weight_decay),
        "mask_fraction": float(mask_fraction),
        "projection_hidden_dim": int(projection_hidden_dim),
        "projection_dim": int(projection_dim),
        "cross_scale_weight": float(cross_scale_weight),
        "variance_weight": float(variance_weight),
        "covariance_weight": float(covariance_weight),
        "teacher_momentum": float(teacher_momentum),
        "augmentation": None if augmentation is None else augmentation.to_dict(),
        "gradient_clip_norm": (
            None if gradient_clip_norm is None else float(gradient_clip_norm)
        ),
        "gradient_clip_value": (
            None if gradient_clip_value is None else float(gradient_clip_value)
        ),
        "seed": int(seed),
        "device": resolved_device,
        "attention_backend": "math" if resolved_device == "cuda" else "cpu_default",
        "optimizer_foreach": False,
        "model_config": asdict(config),
    }
    print(
        "Paired-scale pretraining split | "
        f"train_pairs={len(train_dataset):,} | validation_pairs={len(validation_dataset):,} | "
        f"batch_size={int(batch_size)} | device={resolved_device} | "
        f"validation_groups={validation_groups}",
        flush=True,
    )
    started = time.time()
    for epoch in range(starting_epoch, int(epochs)):
        train_dataset.set_epoch(epoch)
        epoch_started = time.time()
        train_mask_generator = torch.Generator(device=resolved_device)
        train_mask_generator.manual_seed(int(seed) * 1_000_003 + epoch)
        train_metrics, global_step = _run_epoch(
            model,
            train_loader,
            device=resolved_device,
            mask_fraction=float(mask_fraction),
            mask_generator=train_mask_generator,
            optimizer=optimizer,
            gradient_clip_value=gradient_clip_value,
            gradient_clip_norm=gradient_clip_norm,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            learning_rate=float(learning_rate),
            minimum_learning_rate=float(minimum_learning_rate),
            teacher_momentum=float(teacher_momentum),
            desc=f"paired pretrain epoch {epoch + 1}",
        )
        validation_mask_generator = torch.Generator(device=resolved_device)
        validation_mask_generator.manual_seed(int(seed) * 2_000_003 + 17)
        validation_metrics, _ = _run_epoch(
            model,
            validation_loader,
            device=resolved_device,
            mask_fraction=float(mask_fraction),
            mask_generator=validation_mask_generator,
            optimizer=None,
            gradient_clip_value=None,
            gradient_clip_norm=None,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            learning_rate=float(learning_rate),
            minimum_learning_rate=float(minimum_learning_rate),
            teacher_momentum=float(teacher_momentum),
            desc=f"paired validation {epoch + 1}",
        )
        row: dict[str, float] = {"epoch": float(epoch + 1)}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        row["epoch_minutes"] = (time.time() - epoch_started) / 60.0
        row["elapsed_minutes"] = (time.time() - started) / 60.0
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print("Paired pretraining epoch metrics: " + json.dumps(row, sort_keys=True), flush=True)
        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = int(epoch + 1)
            metadata = {
                **summary_base,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "epochs_completed": int(epoch + 1),
                "exported_encoder": "ema_teacher",
            }
            torch.save(
                _checkpoint_payload(
                    model, optimizer, epoch=int(epoch + 1), metadata=metadata
                ),
                best_checkpoint,
            )
        metadata = {
            **summary_base,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_completed": int(epoch + 1),
            "exported_encoder": "ema_teacher",
        }
        torch.save(
            _checkpoint_payload(model, optimizer, epoch=int(epoch + 1), metadata=metadata),
            last_checkpoint,
        )
    summary = {
        **summary_base,
        "checkpoint_path": str(best_checkpoint),
        "last_checkpoint_path": str(last_checkpoint),
        "history_csv": str(history_path),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": int(epochs),
        "duration_minutes_this_invocation": (time.time() - started) / 60.0,
        "final_metrics": history[-1] if history else None,
    }
    (output_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
