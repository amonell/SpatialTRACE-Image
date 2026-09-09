from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.prepared import PreparedPairedPretrainingDataset
from crypt_villus_vit.representation_pretrain import PairedScaleTeacherStudentPretrainer
from crypt_villus_vit.representation_pretrain import pretrain_paired_scale_teacher_student
from crypt_villus_vit.representation_pretrain import split_paired_pretraining_rows


def _prepared_pairs(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    images = np.random.default_rng(3).integers(
        0, 255, size=(8, 2, 32, 32), dtype=np.uint8
    )
    shard = prepared / "shard.npy"
    np.save(shard, images)
    rows = pd.DataFrame(
        {
            "source_id": ["if:a"] * 4 + ["if:b"] * 4,
            "source_group": ["if"] * 8,
            "section_id": ["a"] * 4 + ["b"] * 4,
            "center_id": ["a0", "a0", "a1", "a1", "b0", "b0", "b1", "b1"],
            "scale_id": [0, 1] * 4,
            "prepared_index": np.arange(8),
        }
    )
    manifest = prepared / "manifest.csv"
    rows.to_csv(manifest, index=False)
    metadata = prepared / "metadata.json"
    metadata.write_text(
        __import__("json").dumps(
            {
                "manifest_path": str(manifest),
                "completed_row_count": 8,
                "rows_per_shard": 8,
                "variant_count": 2,
                "image_dtype": "uint8",
                "shard_paths": [str(shard)],
            }
        )
    )
    return rows, metadata


def _config() -> ModelConfig:
    return ModelConfig(
        input_size_px=32,
        patch_size_px=8,
        embed_dim=32,
        depth=1,
        num_heads=4,
        use_fine_branch=False,
    )


def test_paired_dataset_returns_matched_scale_views(tmp_path: Path):
    rows, metadata = _prepared_pairs(tmp_path)
    dataset = PreparedPairedPretrainingDataset(rows, metadata, seed=7)
    assert len(dataset) == 4
    item = dataset[0]
    assert item["local_student_image"].shape == (1, 32, 32)
    assert item["context_student_image"].shape == (1, 32, 32)
    assert item["local_teacher_image"].shape == (1, 32, 32)
    assert item["context_teacher_image"].shape == (1, 32, 32)


def test_teacher_student_objective_is_finite_and_updates_teacher():
    model = PairedScaleTeacherStudentPretrainer(
        _config(), readout="center_2x", projection_hidden_dim=64, projection_dim=32
    )
    batch = {
        key: torch.rand(4, 1, 32, 32)
        for key in (
            "local_student_image",
            "context_student_image",
            "local_teacher_image",
            "context_teacher_image",
        )
    }
    output = model(batch, mask_fraction=0.4)
    assert all(bool(torch.isfinite(value)) for value in output.values())
    before = next(model.teacher_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.student_encoder.parameters()).add_(1.0)
    model.update_teacher(0.5)
    assert not torch.equal(before, next(model.teacher_encoder.parameters()))


def test_explicit_student_attention_matches_torch_attention():
    torch.manual_seed(11)
    model = PairedScaleTeacherStudentPretrainer(
        _config(), readout="center_2x", projection_hidden_dim=64, projection_dim=32
    ).eval()
    layer = model.student_encoder.encoder.layers[0]
    tokens = torch.randn(3, 17, 32)
    normalized = layer.norm1(tokens)
    expected = layer._sa_block(normalized, None, None, is_causal=False)
    observed = model._stable_self_attention(layer, normalized)
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_explicit_student_layer_norm_matches_torch_layer_norm():
    torch.manual_seed(12)
    model = PairedScaleTeacherStudentPretrainer(
        _config(), readout="center_2x", projection_hidden_dim=64, projection_dim=32
    ).eval()
    layer_norm = model.student_encoder.encoder.layers[0].norm1
    values = torch.randn(3, 17, 32)
    torch.testing.assert_close(
        model._stable_layer_norm(layer_norm, values),
        layer_norm(values),
        rtol=1e-5,
        atol=1e-6,
    )


def test_grouped_pair_split_keeps_centers_together(tmp_path: Path):
    rows, _ = _prepared_pairs(tmp_path)
    train, validation, groups = split_paired_pretraining_rows(
        rows, validation_fraction=0.5, seed=7
    )
    assert groups
    assert set(train["center_id"]).isdisjoint(set(validation["center_id"]))


def test_tiny_paired_pretraining_writes_transfer_checkpoint(tmp_path: Path):
    rows, metadata = _prepared_pairs(tmp_path)
    output = tmp_path / "output"
    summary = pretrain_paired_scale_teacher_student(
        rows,
        prepared_pretraining_metadata=metadata,
        output_dir=output,
        config=_config(),
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        minimum_learning_rate=1e-5,
        warmup_epochs=0,
        validation_fraction=0.5,
        readout="center_2x",
        projection_hidden_dim=64,
        projection_dim=32,
        device="cpu",
        augment_intensity_probability=0.0,
    )
    payload = torch.load(summary["checkpoint_path"], map_location="cpu")
    assert payload["format"] == "crypt-villus-vit/paired-teacher-student-pretrain-v1"
    assert "scale_embeddings" in payload["encoder_state_dict"]
    assert summary["best_epoch"] == 1
