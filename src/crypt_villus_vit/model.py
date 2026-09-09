from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    local_crop_px: int = 512
    context_crop_px: int = 2048
    fine_crop_px: int = 128
    input_size_px: int = 256
    fine_input_size_px: int = 128
    patch_size_px: int = 16
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_local_branch: bool = True
    use_context_branch: bool = True
    use_fine_branch: bool = True
    local_readout: str = "mean"
    context_readout: str = "mean"
    encoder_architecture: str = "separate_patch"
    retain_scale_embeddings: bool = False
    fusion_architecture: str = "concat"
    fine_gate_init: float = -3.0


class PatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        readout: str = "mean",
        use_scale_embeddings: bool = False,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        token_count = (image_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        self.scale_embeddings = (
            nn.Parameter(torch.zeros(2, embed_dim)) if bool(use_scale_embeddings) else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * float(mlp_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.readout = str(readout)

    def _readout(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return encoded.mean(dim=1)
        token_count = int(encoded.shape[1])
        grid_size = int(math.isqrt(token_count))
        if grid_size * grid_size != token_count:
            raise ValueError(f"Center readout requires a square token grid; got {token_count} tokens.")
        if self.readout == "center":
            window_size = 1 if grid_size % 2 else 2
        elif self.readout.startswith("center_") and self.readout.endswith("x"):
            window_size = int(self.readout.removeprefix("center_").removesuffix("x"))
        else:
            raise ValueError("Patch readout must be `mean`, `center`, or `center_<N>x`.")
        if window_size <= 0 or window_size > grid_size:
            raise ValueError(f"Invalid center readout window size: {window_size}.")
        start = (grid_size - window_size) // 2
        tokens = encoded.reshape(encoded.shape[0], grid_size, grid_size, encoded.shape[2])
        center_tokens = tokens[:, start : start + window_size, start : start + window_size, :]
        return center_tokens.mean(dim=(1, 2))

    def forward(self, image: torch.Tensor, scale_id: int | None = None) -> torch.Tensor:
        tokens = self.patch(image).flatten(2).transpose(1, 2)
        tokens = tokens + self.positional_embedding[:, : tokens.shape[1]]
        if self.scale_embeddings is not None:
            if scale_id is None or int(scale_id) not in {0, 1}:
                raise ValueError("A scale-aware PatchEncoder requires scale_id 0 or 1.")
            tokens = tokens + self.scale_embeddings[int(scale_id)].reshape(1, 1, -1)
        encoded = self.encoder(tokens)
        return self.norm(self._readout(encoded))


class FineEncoder(nn.Module):
    def __init__(self, *, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.output_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class LegacyDapiVisionTransformer(nn.Module):
    """Original shared local/context DAPI ViT used by the legacy checkpoint."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.input_size_px % config.patch_size_px != 0:
            raise ValueError("input_size_px must be divisible by patch_size_px")
        self.input_size_px = int(config.input_size_px)
        self.patch_size_px = int(config.patch_size_px)
        self.embed_dim = int(config.embed_dim)
        self.patch_embed = nn.Conv2d(
            1,
            int(config.embed_dim),
            kernel_size=int(config.patch_size_px),
            stride=int(config.patch_size_px),
        )
        patch_count = (int(config.input_size_px) // int(config.patch_size_px)) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(config.embed_dim)))
        self.scale_embeddings = nn.Parameter(torch.zeros(2, int(config.embed_dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + patch_count, int(config.embed_dim)))
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.embed_dim),
            nhead=int(config.num_heads),
            dim_feedforward=int(config.embed_dim * float(config.mlp_ratio)),
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config.depth))
        self.norm = nn.LayerNorm(int(config.embed_dim))

    def encode_tokens(self, images: torch.Tensor, scale_ids: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        batch_size = int(patch_tokens.shape[0])
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        scale_embedding = self.scale_embeddings[scale_ids.long()].unsqueeze(1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]] + scale_embedding
        return self.encoder(tokens)

    def encode_image(
        self,
        images: torch.Tensor,
        scale_ids: torch.Tensor,
        *,
        readout: str = "cls",
    ) -> torch.Tensor:
        encoded = self.encode_tokens(images, scale_ids)
        return self.readout_encoded(encoded, readout=readout)

    def readout_encoded(self, encoded: torch.Tensor, *, readout: str = "cls") -> torch.Tensor:
        readout = str(readout)
        if readout == "cls":
            return self.norm(encoded[:, 0])
        patch_tokens = encoded[:, 1:]
        if readout == "mean":
            return self.norm(patch_tokens.mean(dim=1))
        token_count = int(patch_tokens.shape[1])
        grid_size = int(math.isqrt(token_count))
        if grid_size * grid_size != token_count:
            raise ValueError(f"Center readout requires a square token grid; got {token_count} tokens.")
        if readout == "center":
            window_size = 1 if grid_size % 2 else 2
        elif readout.startswith("center_") and readout.endswith("x"):
            window_size = int(readout.removeprefix("center_").removesuffix("x"))
        else:
            raise ValueError("Scale-aware readout must be `cls`, `mean`, `center`, or `center_<N>x`.")
        if window_size <= 0 or window_size > grid_size:
            raise ValueError(f"Invalid center readout window size: {window_size}.")
        start = (grid_size - window_size) // 2
        tokens = patch_tokens.reshape(
            patch_tokens.shape[0], grid_size, grid_size, patch_tokens.shape[2]
        )
        center_tokens = tokens[:, start : start + window_size, start : start + window_size, :]
        return self.norm(center_tokens.mean(dim=(1, 2)))


class LegacyConvEncoder(nn.Module):
    """Fine CNN branch with the exact module layout used by the original checkpoint."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(int(in_channels), int(base_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(base_channels)),
            nn.GELU(),
            nn.Conv2d(int(base_channels), int(base_channels), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(base_channels)),
            nn.GELU(),
            nn.Conv2d(int(base_channels), int(base_channels * 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(base_channels * 2)),
            nn.GELU(),
            nn.Conv2d(int(base_channels * 2), int(base_channels * 2), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(base_channels * 2)),
            nn.GELU(),
            nn.Conv2d(int(base_channels * 2), int(base_channels * 4), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(base_channels * 4)),
            nn.GELU(),
            nn.Conv2d(int(base_channels * 4), int(base_channels * 4), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(base_channels * 4)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output_dim = int(base_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).flatten(1)


class LegacyGraphSAGELayer(nn.Module):
    def __init__(self, *, hidden_dim: int, edge_feature_dim: int, dropout: float):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(int(hidden_dim + edge_feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(int(hidden_dim * 2), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(node_features)
        source_index = edge_index[0]
        target_index = edge_index[1]
        source_features = node_features[source_index]
        messages = self.message_mlp(torch.cat([source_features, edge_features], dim=1))
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, target_index, messages)
        degree = torch.bincount(target_index, minlength=node_features.shape[0]).float().to(node_features.device)
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(1)
        updated = self.update_mlp(torch.cat([node_features, aggregated], dim=1))
        return self.norm(node_features + updated)


class LegacyOrdinalRegressionHead(nn.Module):
    def __init__(self, *, hidden_dim: int, prediction_mode: str = "scalar", ordinal_bin_count: int = 32):
        super().__init__()
        self.prediction_mode = str(prediction_mode)
        self.ordinal_bin_count = int(ordinal_bin_count)
        self.scalar_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        continuous_prediction = torch.sigmoid(self.scalar_head(hidden)).squeeze(1)
        predicted_bins = torch.clamp(
            torch.floor(continuous_prediction * float(self.ordinal_bin_count)),
            0,
            self.ordinal_bin_count - 1,
        ).long()
        return {
            "continuous_prediction": continuous_prediction,
            "predicted_bins": predicted_bins,
            "predicted_offsets": continuous_prediction * float(self.ordinal_bin_count) - predicted_bins.float(),
        }


class LegacyAxisFractionPredictor(nn.Module):
    """Compatibility model for checkpoints that store `config` plus `model_state_dict`."""

    def __init__(self, config: ModelConfig, *, legacy_config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.legacy_config = dict(legacy_config)
        self.primary_task_name = str(legacy_config.get("supervised_target_column", "target_axis"))
        self.aux_task_name = str(
            legacy_config.get("supervised_aux_target_column", "epithelial_distance_clipped_1p0")
        )
        graph_hidden_dim = int(legacy_config.get("graph_hidden_dim", config.embed_dim))
        graph_dropout = float(legacy_config.get("graph_dropout", 0.1))
        gnn_layers = int(legacy_config.get("gnn_layers", 3))
        self.encoder = LegacyDapiVisionTransformer(config)
        self.fine_encoder = LegacyConvEncoder(in_channels=1, base_channels=32) if config.use_fine_branch else None
        node_scalar_dim = 4
        input_hidden_dim = int(config.embed_dim * 2 + node_scalar_dim)
        if self.fine_encoder is not None:
            input_hidden_dim += int(self.fine_encoder.output_dim)
        self.node_input = nn.Sequential(
            nn.Linear(input_hidden_dim, graph_hidden_dim),
            nn.GELU(),
            nn.Dropout(graph_dropout),
            nn.Linear(graph_hidden_dim, graph_hidden_dim),
        )
        self.gnn_layers = nn.ModuleList(
            [
                LegacyGraphSAGELayer(hidden_dim=graph_hidden_dim, edge_feature_dim=5, dropout=graph_dropout)
                for _ in range(gnn_layers)
            ]
        )
        self.post_mlp = nn.Sequential(
            nn.LayerNorm(graph_hidden_dim),
            nn.Linear(graph_hidden_dim, graph_hidden_dim),
            nn.GELU(),
            nn.Dropout(graph_dropout),
        )
        self.target_heads = nn.ModuleDict(
            {
                self.primary_task_name: LegacyOrdinalRegressionHead(
                    hidden_dim=graph_hidden_dim,
                    prediction_mode=str(legacy_config.get("prediction_mode", "scalar")),
                    ordinal_bin_count=int(legacy_config.get("ordinal_bin_count", 32)),
                ),
                self.aux_task_name: LegacyOrdinalRegressionHead(
                    hidden_dim=graph_hidden_dim,
                    prediction_mode=str(legacy_config.get("prediction_mode", "scalar")),
                    ordinal_bin_count=int(legacy_config.get("ordinal_bin_count", 32)),
                ),
            }
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        local_image = batch.get("local_image", batch.get("small_images"))
        context_image = batch.get("context_image", batch.get("large_images"))
        if local_image is None or context_image is None:
            raise KeyError("Legacy model requires `local_image`/`context_image` or `small_images`/`large_images`.")
        device = local_image.device
        batch_size = int(local_image.shape[0])
        scale_small = torch.zeros(batch_size, dtype=torch.long, device=device)
        scale_large = torch.ones(batch_size, dtype=torch.long, device=device)
        parts = [
            self.encoder.encode_image(local_image, scale_small),
            self.encoder.encode_image(context_image, scale_large),
        ]
        if self.fine_encoder is not None:
            fine_image = batch.get("fine_image", batch.get("fine_images"))
            if fine_image is None:
                raise KeyError("Legacy model checkpoint expects a fine image branch.")
            parts.append(self.fine_encoder(fine_image))
        node_scalar_features = batch.get("node_scalar_features")
        if node_scalar_features is None:
            node_scalar_features = torch.zeros(batch_size, 4, dtype=local_image.dtype, device=device)
        parts.append(node_scalar_features)
        hidden = self.node_input(torch.cat(parts, dim=1))
        edge_index = batch.get("edge_index")
        edge_features = batch.get("edge_features")
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        if edge_features is None:
            edge_features = torch.empty((0, 5), dtype=local_image.dtype, device=device)
        for layer in self.gnn_layers:
            hidden = layer(hidden, edge_index, edge_features)
        hidden = self.post_mlp(hidden)
        task_outputs = {task_name: head(hidden) for task_name, head in self.target_heads.items()}
        primary = task_outputs[self.primary_task_name]["continuous_prediction"]
        auxiliary = task_outputs[self.aux_task_name]["continuous_prediction"]
        return {
            "predicted_axis_coordinate": primary,
            "predicted_epithelial_distance_clipped_1p0": auxiliary,
            "task_outputs": task_outputs,
        }


class MultitaskDapiVit(nn.Module):
    """Two-scale DAPI ViT with optional fine-CNN branch and two scalar heads."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not (config.use_local_branch or config.use_context_branch or config.use_fine_branch):
            raise ValueError("At least one image branch must be enabled.")
        self.config = config
        encoder_architecture = str(config.encoder_architecture).strip().lower()
        if encoder_architecture not in {"separate_patch", "shared_scale_aware", "separate_scale_aware"}:
            raise ValueError(
                "encoder_architecture must be `separate_patch`, `shared_scale_aware`, "
                "or `separate_scale_aware`."
            )
        self.encoder_architecture = encoder_architecture
        self.shared_encoder = None
        self.local_encoder = None
        self.context_encoder = None
        if encoder_architecture == "shared_scale_aware":
            if config.use_local_branch or config.use_context_branch:
                self.shared_encoder = LegacyDapiVisionTransformer(config)
        elif encoder_architecture == "separate_scale_aware":
            self.local_encoder = LegacyDapiVisionTransformer(config) if config.use_local_branch else None
            self.context_encoder = LegacyDapiVisionTransformer(config) if config.use_context_branch else None
        else:
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
                    use_scale_embeddings=config.retain_scale_embeddings,
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
                    use_scale_embeddings=config.retain_scale_embeddings,
                )
                if config.use_context_branch
                else None
            )
        self.fine_encoder = FineEncoder(output_dim=config.embed_dim // 2) if config.use_fine_branch else None
        fusion_architecture = str(config.fusion_architecture).strip().lower()
        if fusion_architecture not in {"concat", "gated_fine"}:
            raise ValueError("fusion_architecture must be `concat` or `gated_fine`.")
        if fusion_architecture == "gated_fine" and self.fine_encoder is None:
            raise ValueError("gated_fine fusion requires the fine branch.")
        self.fusion_architecture = fusion_architecture
        self.fine_gate = (
            nn.Parameter(torch.tensor(float(config.fine_gate_init)))
            if fusion_architecture == "gated_fine"
            else None
        )
        hidden_in = (
            (config.embed_dim if config.use_local_branch else 0)
            + (config.embed_dim if config.use_context_branch else 0)
            + (config.embed_dim // 2 if self.fine_encoder is not None else 0)
        )
        hidden = config.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_in),
            nn.Linear(hidden_in, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.axis_head = nn.Linear(hidden, 1)
        self.epithelial_head = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        parts = []
        if self.config.use_local_branch:
            if "local_image" not in batch:
                raise KeyError("Model was configured with a local branch, but batch has no `local_image`.")
            if self.shared_encoder is not None:
                scale_ids = torch.zeros(
                    batch["local_image"].shape[0], dtype=torch.long, device=batch["local_image"].device
                )
                parts.append(
                    self.shared_encoder.encode_image(
                        batch["local_image"], scale_ids, readout=self.config.local_readout
                    )
                )
            elif self.encoder_architecture == "separate_scale_aware":
                scale_ids = torch.zeros(
                    batch["local_image"].shape[0], dtype=torch.long, device=batch["local_image"].device
                )
                parts.append(
                    self.local_encoder.encode_image(
                        batch["local_image"], scale_ids, readout=self.config.local_readout
                    )
                )
            else:
                parts.append(
                    self.local_encoder(
                        batch["local_image"],
                        scale_id=0 if self.config.retain_scale_embeddings else None,
                    )
                )
        if self.config.use_context_branch:
            if "context_image" not in batch:
                raise KeyError("Model was configured with a context branch, but batch has no `context_image`.")
            if self.shared_encoder is not None:
                scale_ids = torch.ones(
                    batch["context_image"].shape[0], dtype=torch.long, device=batch["context_image"].device
                )
                parts.append(
                    self.shared_encoder.encode_image(
                        batch["context_image"], scale_ids, readout=self.config.context_readout
                    )
                )
            elif self.encoder_architecture == "separate_scale_aware":
                scale_ids = torch.ones(
                    batch["context_image"].shape[0], dtype=torch.long, device=batch["context_image"].device
                )
                parts.append(
                    self.context_encoder.encode_image(
                        batch["context_image"], scale_ids, readout=self.config.context_readout
                    )
                )
            else:
                parts.append(
                    self.context_encoder(
                        batch["context_image"],
                        scale_id=1 if self.config.retain_scale_embeddings else None,
                    )
                )
        if self.fine_encoder is not None:
            if "fine_image" not in batch:
                raise KeyError("Model was configured with a fine branch, but batch has no `fine_image`.")
            fine_features = self.fine_encoder(batch["fine_image"])
            if self.fine_gate is not None:
                fine_features = fine_features * torch.sigmoid(self.fine_gate)
            parts.append(fine_features)
        hidden = self.head(torch.cat(parts, dim=1))
        axis_logit = self.axis_head(hidden).squeeze(1)
        epithelial_logit = self.epithelial_head(hidden).squeeze(1)
        axis = torch.sigmoid(axis_logit)
        epithelial = torch.sigmoid(epithelial_logit)
        return {
            "axis_logit": axis_logit,
            "epithelial_logit": epithelial_logit,
            "predicted_axis_coordinate": axis,
            "predicted_epithelial_distance_clipped_1p0": epithelial,
        }


def checkpoint_payload(model: MultitaskDapiVit, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "format": "tissueframe-peyer-representation/v1" if hasattr(model, "classifier") else "crypt-villus-vit/v1",
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }


def save_model_checkpoint(model: MultitaskDapiVit, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata=metadata), path)


def _model_config_from_legacy_config(config: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        local_crop_px=int(config.get("local_if_crop_fullres_px", 512)),
        context_crop_px=int(config.get("context_if_crop_fullres_px", 2048)),
        fine_crop_px=int(config.get("fine_if_crop_fullres_px", 128)),
        input_size_px=int(config.get("vit_input_size_px", 256)),
        fine_input_size_px=int(config.get("fine_model_input_size_px", 128)),
        patch_size_px=int(config.get("vit_patch_size_px", 16)),
        embed_dim=int(config.get("vit_embed_dim", 256)),
        depth=int(config.get("vit_depth", 6)),
        num_heads=int(config.get("vit_num_heads", 8)),
        mlp_ratio=float(config.get("vit_mlp_ratio", 4.0)),
        dropout=float(config.get("vit_dropout", 0.0)),
        use_local_branch=True,
        use_context_branch=True,
        use_fine_branch=bool(config.get("use_fine_branch", int(config.get("fine_if_crop_fullres_px", 0)) > 0)),
        local_readout=str(config.get("local_readout", "mean")),
        context_readout=str(config.get("context_readout", "mean")),
    )


def load_model_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if "model_config" in payload:
        config = ModelConfig(**payload["model_config"])
        if payload.get("format") == "tissueframe-peyer-representation/v1":
            from .peyer import PeyerClassifier
            model: nn.Module = PeyerClassifier(config)
        else:
            model = MultitaskDapiVit(config)
    elif "config" in payload:
        config = _model_config_from_legacy_config(dict(payload["config"]))
        model = LegacyAxisFractionPredictor(config, legacy_config=dict(payload["config"]))
    else:
        raise ValueError(
            "Checkpoint does not contain `model_config` or a supported `config` payload."
        )
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if state is None:
        raise ValueError("Checkpoint does not contain `model_state_dict` or `state_dict`.")
    model.load_state_dict(state, strict=bool(strict))
    model.to(torch.device(device))
    model.eval()
    return model, payload
