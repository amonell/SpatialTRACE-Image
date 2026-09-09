from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crypt_villus_vit.cli import main
from crypt_villus_vit.augmentation import IntensityAugmentationConfig
from crypt_villus_vit.augmentation import apply_intensity_parameters
from crypt_villus_vit.augmentation import sample_intensity_parameters
from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.model import MultitaskDapiVit
from crypt_villus_vit.model import save_model_checkpoint
from crypt_villus_vit.prepared import PreparedCellCropDataset
from crypt_villus_vit.train import _apply_freeze_mode
from crypt_villus_vit.train import train_model


def test_freeze_mode_heads_only_trains_heads():
    model = MultitaskDapiVit(ModelConfig(embed_dim=64, depth=2, num_heads=4, input_size_px=64, fine_input_size_px=32))
    summary = _apply_freeze_mode(model, "heads")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert summary["freeze_mode"] == "heads"
    assert trainable
    assert all(name.startswith(("head.", "axis_head.", "epithelial_head.")) for name in trainable)


def test_freeze_mode_last1_trains_heads_and_last_transformer_blocks():
    model = MultitaskDapiVit(ModelConfig(embed_dim=64, depth=2, num_heads=4, input_size_px=64, fine_input_size_px=32))
    summary = _apply_freeze_mode(model, "last1")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert summary["freeze_mode"] == "last1"
    assert any(name.startswith("local_encoder.encoder.layers.1.") for name in trainable)
    assert any(name.startswith("context_encoder.encoder.layers.1.") for name in trainable)
    assert any(name.startswith("head.") for name in trainable)
    assert not any(name.startswith("local_encoder.encoder.layers.0.") for name in trainable)
    assert not any(name.startswith("context_encoder.encoder.layers.0.") for name in trainable)


def test_train_writes_history_and_summary(tmp_path: Path):
    image = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
    image_path = tmp_path / "demo.npy"
    np.save(image_path, image)
    source_manifest = tmp_path / "sources.csv"
    source_manifest.write_text(f"source_id,image_path,pixel_size_um\nif:test,{image_path},0.325\n")
    supervised_manifest = tmp_path / "supervised.csv"
    pd.DataFrame(
        {
            "source_id": ["if:test"] * 4,
            "centroid_x_fullres_px": [64, 96, 128, 160],
            "centroid_y_fullres_px": [64, 96, 128, 160],
            "target_axis": [0.1, 0.3, 0.6, 0.9],
            "epithelial_distance_clipped_1p0": [0.9, 0.7, 0.4, 0.2],
        }
    ).to_csv(supervised_manifest, index=False)
    output_dir = tmp_path / "train"
    main(
        [
            "train",
            "--source-manifest",
            str(source_manifest),
            "--supervised-manifest",
            str(supervised_manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--validation-fraction",
            "0.25",
            "--seed",
            "7",
            "--device",
            "cpu",
            "--input-size-px",
            "64",
            "--context-crop-px",
            "128",
            "--local-crop-px",
            "64",
            "--fine-crop-px",
            "32",
            "--fine-input-size-px",
            "32",
            "--embed-dim",
            "64",
            "--depth",
            "1",
            "--num-heads",
            "4",
        ]
    )
    assert (output_dir / "crypt_villus_vit_model.pt").exists()
    assert (output_dir / "history.csv").exists()
    assert (output_dir / "training_summary.json").exists()
    history = pd.read_csv(output_dir / "history.csv")
    assert {"train_loss", "validation_loss"}.issubset(history.columns)


def test_binary_train_and_predict_exports_probability_columns(tmp_path: Path):
    image = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
    image_path = tmp_path / "demo.npy"
    np.save(image_path, image)
    source_manifest = tmp_path / "sources.csv"
    source_manifest.write_text(f"source_id,image_path,pixel_size_um\nif:test,{image_path},0.325\n")
    supervised_manifest = tmp_path / "binary_supervised.csv"
    pd.DataFrame(
        {
            "source_id": ["if:test"] * 6,
            "centroid_x_fullres_px": [48, 80, 112, 144, 176, 208],
            "centroid_y_fullres_px": [48, 80, 112, 144, 176, 208],
            "peyer_label": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "train", "train", "validation", "validation"],
        }
    ).to_csv(supervised_manifest, index=False)
    output_dir = tmp_path / "binary_train"
    main(
        [
            "train",
            "--source-manifest",
            str(source_manifest),
            "--supervised-manifest",
            str(supervised_manifest),
            "--output-dir",
            str(output_dir),
            "--task-type",
            "binary_classification",
            "--target-column",
            "peyer_label",
            "--prediction-column",
            "peyer_probability",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--input-size-px",
            "64",
            "--context-crop-px",
            "128",
            "--local-crop-px",
            "64",
            "--fine-crop-px",
            "32",
            "--fine-input-size-px",
            "32",
            "--embed-dim",
            "64",
            "--depth",
            "1",
            "--num-heads",
            "4",
        ]
    )
    summary = __import__("json").loads((output_dir / "training_summary.json").read_text())
    assert summary["task_type"] == "binary_classification"
    assert summary["target_column"] == "peyer_label"

    cells_csv = tmp_path / "cells.csv"
    pd.read_csv(supervised_manifest).drop(columns=["peyer_label", "split"]).to_csv(cells_csv, index=False)
    prediction_dir = tmp_path / "binary_predictions"
    main(
        [
            "predict-cells",
            "--source-manifest",
            str(source_manifest),
            "--cells-csv",
            str(cells_csv),
            "--checkpoint",
            str(output_dir / "best_crypt_villus_vit_model.pt"),
            "--output-dir",
            str(prediction_dir),
            "--device",
            "cpu",
        ]
    )
    predictions = pd.read_csv(prediction_dir / "predictions.csv")
    assert {"peyer_probability", "peyer_logit", "peyer_class"}.issubset(predictions.columns)
    assert "predicted_gate_name" not in predictions.columns
    predict_summary = __import__("json").loads((prediction_dir / "summary.json").read_text())
    assert predict_summary["task_type"] == "binary_classification"
    assert predict_summary["gate_percentages_csv"] is None


def test_pretrain_writes_encoder_checkpoint(tmp_path: Path):
    image = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
    image_path = tmp_path / "demo.npy"
    np.save(image_path, image)
    source_manifest = tmp_path / "sources.csv"
    source_manifest.write_text(f"source_id,image_path,pixel_size_um\nif:test,{image_path},0.325\n")
    supervised_manifest = tmp_path / "supervised.csv"
    pd.DataFrame(
        {
            "source_id": ["if:test"] * 3,
            "centroid_x_fullres_px": [64, 96, 128],
            "centroid_y_fullres_px": [64, 96, 128],
            "target_axis": [0.1, 0.5, 0.9],
            "epithelial_distance_clipped_1p0": [0.8, 0.5, 0.2],
        }
    ).to_csv(supervised_manifest, index=False)
    output_dir = tmp_path / "pretrain"
    main(
        [
            "pretrain",
            "--source-manifest",
            str(source_manifest),
            "--supervised-manifest",
            str(supervised_manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--input-size-px",
            "64",
            "--context-crop-px",
            "128",
            "--local-crop-px",
            "64",
            "--fine-crop-px",
            "32",
            "--fine-input-size-px",
            "32",
            "--embed-dim",
            "64",
            "--depth",
            "1",
            "--num-heads",
            "4",
        ]
    )
    assert (output_dir / "dapi_encoder_pretrain.pt").exists()
    assert (output_dir / "pretrain_history.csv").exists()
    assert (output_dir / "pretrain_summary.json").exists()


def test_pretrain_from_prepared_shards(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    shard_dir = prepared_dir / "pretraining_shards"
    shard_dir.mkdir(parents=True)
    images = np.random.default_rng(9).integers(
        0,
        255,
        size=(6, 2, 64, 64),
        dtype=np.uint8,
    )
    shard_path = shard_dir / "pretraining_shard_00000.npy"
    np.save(shard_path, images)
    manifest = prepared_dir / "pretraining_manifest.csv"
    pd.DataFrame(
        {
            "prepared_index": np.arange(6),
            "scale_id": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "train", "train", "validation", "validation"],
        }
    ).to_csv(manifest, index=False)
    metadata = prepared_dir / "pretraining_metadata.json"
    metadata.write_text(
        __import__("json").dumps(
            {
                "manifest_path": str(manifest),
                "completed_row_count": 6,
                "rows_per_shard": 6,
                "variant_count": 2,
                "image_dtype": "uint8",
                "shard_paths": [str(shard_path)],
            }
        )
    )
    output_dir = tmp_path / "pretrain_prepared"
    main(
        [
            "pretrain",
            "--prepared-pretraining-metadata",
            str(metadata),
            "--supervised-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--input-size-px",
            "64",
            "--patch-size-px",
            "16",
            "--embed-dim",
            "32",
            "--depth",
            "1",
            "--num-heads",
            "4",
            "--augment-intensity-probability",
            "1.0",
            "--augment-contrast-range",
            "0.6",
            "1.5",
            "--augment-gaussian-blur-sigma-range",
            "0.1",
            "2.5",
            "--gradient-clip-norm",
            "1.0",
        ]
    )
    checkpoint = torch.load(output_dir / "dapi_encoder_pretrain.pt", map_location="cpu")
    assert checkpoint["format"] == "crypt-villus-vit/pretrain-v1"
    assert "patch_embed.weight" in checkpoint["encoder_state_dict"]
    assert checkpoint["metadata"]["image_augmentation"]["gaussian_blur_sigma_range"] == [0.1, 2.5]
    assert checkpoint["metadata"]["gradient_clip_norm"] == 1.0
    history = pd.read_csv(output_dir / "pretrain_history.csv")
    assert {"train_loss", "validation_loss", "elapsed_minutes"}.issubset(history.columns)


def test_pretrain_materializes_legacy_shared_encoder(tmp_path: Path):
    embed_dim = 32
    patch_count = 16
    legacy_state = {
        "patch_embed.weight": torch.randn(embed_dim, 1, 16, 16),
        "patch_embed.bias": torch.randn(embed_dim),
        "pos_embed": torch.randn(1, patch_count + 1, embed_dim),
    }
    layer = torch.nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=4,
        dim_feedforward=embed_dim * 4,
        batch_first=True,
        norm_first=True,
    )
    encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
    for key, value in encoder.state_dict().items():
        legacy_state[f"encoder.{key}"] = value
    norm = torch.nn.LayerNorm(embed_dim)
    for key, value in norm.state_dict().items():
        legacy_state[f"norm.{key}"] = value

    checkpoint = tmp_path / "legacy_pretrain.pt"
    torch.save({"encoder_state_dict": legacy_state}, checkpoint)
    source_manifest = tmp_path / "sources.csv"
    supervised_manifest = tmp_path / "supervised.csv"
    source_manifest.write_text("source_id,image_path\nsample_000,demo.npy\n")
    supervised_manifest.write_text("source_id,centroid_x_fullres_px,centroid_y_fullres_px,target_axis\n")
    output_dir = tmp_path / "pretrain"
    main(
        [
            "pretrain",
            "--source-manifest",
            str(source_manifest),
            "--supervised-manifest",
            str(supervised_manifest),
            "--output-dir",
            str(output_dir),
            "--materialize-only",
            "--initial-checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--input-size-px",
            "64",
            "--patch-size-px",
            "16",
            "--embed-dim",
            str(embed_dim),
            "--depth",
            "1",
            "--num-heads",
            "4",
        ]
    )
    materialized = torch.load(output_dir / "dapi_encoder_pretrain.pt", map_location="cpu")
    local_state = materialized["encoder_state_dict"]["local_encoder"]
    assert torch.equal(local_state["patch.weight"], legacy_state["patch_embed.weight"])
    assert torch.equal(local_state["positional_embedding"], legacy_state["pos_embed"][:, 1:, :])
    assert (output_dir / "pretrain_summary.json").exists()


def test_train_from_prepared_supervised_shards(tmp_path: Path):
    shard_dir = tmp_path / "prepared" / "xenium_supervised_shards_v1"
    shard_dir.mkdir(parents=True)
    local = np.random.default_rng(1).integers(0, 255, size=(4, 1, 64, 64), dtype=np.uint8)
    context = np.random.default_rng(2).integers(0, 255, size=(4, 1, 64, 64), dtype=np.uint8)
    fine = np.random.default_rng(3).integers(0, 255, size=(4, 1, 32, 32), dtype=np.uint8)
    local_path = shard_dir / "local_shard_00000.npy"
    context_path = shard_dir / "context_shard_00000.npy"
    fine_path = shard_dir / "fine_shard_00000.npy"
    np.save(local_path, local)
    np.save(context_path, context)
    np.save(fine_path, fine)
    manifest = tmp_path / "prepared" / "xenium_supervised_prepared_sharded_v1_manifest.csv"
    pd.DataFrame(
        {
            "source_id": ["sample_000"] * 4,
            "prepared_index": [0, 1, 2, 3],
            "target_axis": [0.1, 0.3, 0.6, 0.9],
            "epithelial_distance_clipped_1p0": [0.9, 0.7, 0.4, 0.2],
            "split": ["train", "train", "train", "validation"],
        }
    ).to_csv(manifest, index=False)
    metadata = tmp_path / "prepared" / "xenium_supervised_prepared_sharded_v1_metadata.json"
    metadata.write_text(
        __import__("json").dumps(
            {
                "manifest_path": str(manifest),
                "completed_row_count": 4,
                "total_row_count": 4,
                "rows_per_shard": 4,
                "image_dtype": "uint8",
                "local_shard_paths": [str(local_path)],
                "context_shard_paths": [str(context_path)],
                "fine_shard_paths": [str(fine_path)],
            }
        )
    )
    image_path = tmp_path / "unused.npy"
    np.save(image_path, np.zeros((8, 8), dtype=np.float32))
    source_manifest = tmp_path / "sources.csv"
    source_manifest.write_text(f"source_id,image_path,pixel_size_um\nsample_000,{image_path},0.325\n")
    output_dir = tmp_path / "train_prepared"
    main(
        [
            "train",
            "--source-manifest",
            str(source_manifest),
            "--supervised-manifest",
            str(manifest),
            "--prepared-supervised-metadata",
            str(metadata),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--input-size-px",
            "64",
            "--context-crop-px",
            "128",
            "--local-crop-px",
            "64",
            "--fine-crop-px",
            "32",
            "--fine-input-size-px",
            "32",
            "--embed-dim",
            "64",
            "--depth",
            "1",
            "--num-heads",
            "4",
            "--augment-intensity-probability",
            "1.0",
            "--augment-contrast-range",
            "0.6",
            "1.5",
            "--augment-gaussian-blur-sigma-range",
            "0.1",
            "2.5",
            "--gradient-clip-norm",
            "1.0",
        ]
    )
    summary = __import__("json").loads((output_dir / "training_summary.json").read_text())
    assert summary["prepared_supervised_summary"]["row_count"] == 4
    assert summary["intensity_augmentation"]["gaussian_blur_sigma_range"] == [0.1, 2.5]
    assert summary["gradient_clip_norm"] == 1.0
    assert summary["batching_strategy"] == "global_shuffle"
    assert (output_dir / "crypt_villus_vit_model.pt").exists()


def test_prepared_dataset_applies_intensity_augmentation(tmp_path: Path):
    shard_dir = tmp_path / "prepared" / "xenium_supervised_shards_v1"
    shard_dir.mkdir(parents=True)
    local = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(1, 1, 64, 64)
    context = np.flip(local, axis=-1).copy()
    fine = np.linspace(0, 255, 32 * 32, dtype=np.uint8).reshape(1, 1, 32, 32)
    local_path = shard_dir / "local_shard_00000.npy"
    context_path = shard_dir / "context_shard_00000.npy"
    fine_path = shard_dir / "fine_shard_00000.npy"
    np.save(local_path, local)
    np.save(context_path, context)
    np.save(fine_path, fine)
    manifest = tmp_path / "prepared" / "manifest.csv"
    pd.DataFrame(
        {
            "source_id": ["sample_000"],
            "prepared_index": [0],
            "target_axis": [0.5],
            "epithelial_distance_clipped_1p0": [0.5],
        }
    ).to_csv(manifest, index=False)
    metadata = tmp_path / "prepared" / "metadata.json"
    metadata.write_text(
        __import__("json").dumps(
            {
                "manifest_path": str(manifest),
                "completed_row_count": 1,
                "total_row_count": 1,
                "rows_per_shard": 1,
                "image_dtype": "uint8",
                "local_shard_paths": [str(local_path)],
                "context_shard_paths": [str(context_path)],
                "fine_shard_paths": [str(fine_path)],
            }
        )
    )
    config = ModelConfig(local_crop_px=64, context_crop_px=64, fine_crop_px=32, input_size_px=64, fine_input_size_px=32)
    augmentation = IntensityAugmentationConfig(
        probability=1.0,
        brightness_delta=0.0,
        contrast_range=(0.5, 0.5),
        seed=7,
    )
    dataset = PreparedCellCropDataset(
        pd.read_csv(manifest),
        metadata,
        config,
        intensity_augmentation=augmentation,
    )
    observed = dataset[0]["local_image"]
    original = local[0].astype(np.float32) / 255.0
    expected = np.clip((original - original.mean()) * 0.5 + original.mean(), 0.0, 1.0).astype(np.float32)
    np.testing.assert_allclose(observed, expected, atol=1e-6)


def test_gaussian_blur_reduces_high_frequency_variation():
    checkerboard = (np.indices((64, 64)).sum(axis=0) % 2).astype(np.float32)[None]
    blurred = apply_intensity_parameters(
        checkerboard,
        contrast=1.0,
        brightness=0.0,
        gaussian_blur_sigma=2.0,
    )
    assert blurred.shape == checkerboard.shape
    assert float(blurred.std()) < float(checkerboard.std()) * 0.05
    assert np.isclose(float(blurred.mean()), float(checkerboard.mean()), atol=1e-4)


def test_blur_contrast_sampling_is_deterministic_and_epoch_specific():
    config = IntensityAugmentationConfig(
        probability=1.0,
        brightness_delta=0.0,
        contrast_range=(0.6, 1.5),
        gaussian_blur_sigma_range=(0.1, 2.5),
        seed=42,
    )
    first = sample_intensity_parameters(config, sample_index=17, epoch=3)
    repeated = sample_intensity_parameters(config, sample_index=17, epoch=3)
    next_epoch = sample_intensity_parameters(config, sample_index=17, epoch=4)
    assert first == repeated
    assert first != next_epoch
    assert first is not None
    contrast, brightness, blur_sigma = first
    assert 0.6 <= contrast <= 1.5
    assert brightness == 0.0
    assert 0.1 <= blur_sigma <= 2.5


def test_train_from_prepared_shards_keeps_rows_with_missing_epithelial_labels(tmp_path: Path):
    shard_dir = tmp_path / "prepared" / "xenium_supervised_shards_v1"
    shard_dir.mkdir(parents=True)
    local = np.random.default_rng(4).integers(0, 255, size=(6, 1, 64, 64), dtype=np.uint8)
    context = np.random.default_rng(5).integers(0, 255, size=(6, 1, 64, 64), dtype=np.uint8)
    fine = np.random.default_rng(6).integers(0, 255, size=(6, 1, 32, 32), dtype=np.uint8)
    local_path = shard_dir / "local_shard_00000.npy"
    context_path = shard_dir / "context_shard_00000.npy"
    fine_path = shard_dir / "fine_shard_00000.npy"
    np.save(local_path, local)
    np.save(context_path, context)
    np.save(fine_path, fine)

    manifest = tmp_path / "prepared" / "xenium_supervised_prepared_sharded_v1_manifest.csv"
    pd.DataFrame(
        {
            "source_id": ["if:test", "if:test", "if:test", "xenium:test", "xenium:test", "xenium:test"],
            "prepared_index": [0, 1, 2, 3, 4, 5],
            "target_axis": [0.1, 0.2, 0.3, 0.6, 0.8, 0.9],
            "epithelial_distance_clipped_1p0": [np.nan, np.nan, np.nan, 0.6, 0.4, 0.2],
            "split": ["train", "train", "validation", "train", "train", "validation"],
        }
    ).to_csv(manifest, index=False)
    metadata = tmp_path / "prepared" / "xenium_supervised_prepared_sharded_v1_metadata.json"
    metadata.write_text(
        __import__("json").dumps(
            {
                "manifest_path": str(manifest),
                "completed_row_count": 6,
                "total_row_count": 6,
                "rows_per_shard": 6,
                "image_dtype": "uint8",
                "local_shard_paths": [str(local_path)],
                "context_shard_paths": [str(context_path)],
                "fine_shard_paths": [str(fine_path)],
            }
        )
    )
    image_path = tmp_path / "unused.npy"
    np.save(image_path, np.zeros((8, 8), dtype=np.float32))
    source_manifest = tmp_path / "sources.csv"
    source_manifest.write_text(
        f"source_id,image_path,pixel_size_um\nif:test,{image_path},0.325\nxenium:test,{image_path},0.2125\n"
    )
    output_dir = tmp_path / "train_prepared_partial"
    main(
        [
            "train",
            "--source-manifest",
            str(source_manifest),
            "--supervised-manifest",
            str(manifest),
            "--prepared-supervised-metadata",
            str(metadata),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--input-size-px",
            "64",
            "--context-crop-px",
            "128",
            "--local-crop-px",
            "64",
            "--fine-crop-px",
            "32",
            "--fine-input-size-px",
            "32",
            "--embed-dim",
            "64",
            "--depth",
            "1",
            "--num-heads",
            "4",
        ]
    )
    summary = __import__("json").loads((output_dir / "training_summary.json").read_text())
    history = pd.read_csv(output_dir / "history.csv")
    assert summary["train_row_count"] == 4
    assert summary["validation_row_count"] == 2
    assert history.loc[0, "train_axis_scored_n"] == 4
    assert history.loc[0, "train_epithelial_scored_n"] == 2
    assert history.loc[0, "validation_axis_scored_n"] == 2
    assert history.loc[0, "validation_epithelial_scored_n"] == 1


def test_train_from_prepared_shards_records_initial_checkpoint(tmp_path: Path):
    shard_dir = tmp_path / "prepared" / "xenium_supervised_shards_v1"
    shard_dir.mkdir(parents=True)
    local = np.random.default_rng(7).integers(0, 255, size=(2, 1, 32, 32), dtype=np.uint8)
    context = np.random.default_rng(8).integers(0, 255, size=(2, 1, 32, 32), dtype=np.uint8)
    fine = np.random.default_rng(9).integers(0, 255, size=(2, 1, 16, 16), dtype=np.uint8)
    local_path = shard_dir / "local_shard_00000.npy"
    context_path = shard_dir / "context_shard_00000.npy"
    fine_path = shard_dir / "fine_shard_00000.npy"
    np.save(local_path, local)
    np.save(context_path, context)
    np.save(fine_path, fine)
    manifest = tmp_path / "prepared" / "xenium_supervised_prepared_sharded_v1_manifest.csv"
    rows = pd.DataFrame(
        {
            "source_id": ["xenium:test", "xenium:test"],
            "prepared_index": [0, 1],
            "target_axis": [0.2, 0.8],
            "epithelial_distance_clipped_1p0": [0.7, 0.3],
            "split": ["train", "validation"],
        }
    )
    rows.to_csv(manifest, index=False)
    metadata = tmp_path / "prepared" / "xenium_supervised_prepared_sharded_v1_metadata.json"
    metadata.write_text(
        __import__("json").dumps(
            {
                "manifest_path": str(manifest),
                "completed_row_count": 2,
                "total_row_count": 2,
                "rows_per_shard": 2,
                "image_dtype": "uint8",
                "local_shard_paths": [str(local_path)],
                "context_shard_paths": [str(context_path)],
                "fine_shard_paths": [str(fine_path)],
            }
        )
    )
    config = ModelConfig(
        local_crop_px=32,
        context_crop_px=32,
        fine_crop_px=16,
        input_size_px=32,
        fine_input_size_px=16,
        patch_size_px=16,
        embed_dim=32,
        depth=1,
        num_heads=4,
    )
    initial_checkpoint = tmp_path / "initial.pt"
    save_model_checkpoint(
        MultitaskDapiVit(config),
        initial_checkpoint,
        metadata={"checkpoint_role": "unit-test-initialization"},
    )
    output_dir = tmp_path / "train_from_initial"
    summary = train_model(
        rows,
        sources={},
        output_dir=output_dir,
        config=config,
        epochs=1,
        batch_size=1,
        device="cpu",
        prepared_supervised_metadata=metadata,
        initial_checkpoint=initial_checkpoint,
    )
    assert summary["initial_checkpoint"] == str(initial_checkpoint)
    assert summary["initial_checkpoint_loaded"]["format"] == "crypt-villus-vit/v1"
    assert summary["initial_checkpoint_loaded"]["source_metadata"]["checkpoint_role"] == "unit-test-initialization"
    assert summary["pretrained_checkpoint"] is None
