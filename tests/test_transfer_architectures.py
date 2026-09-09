from dataclasses import replace
from pathlib import Path

import pytest
import torch

from crypt_villus_vit.model import LegacyDapiVisionTransformer
from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.model import MultitaskDapiVit
from crypt_villus_vit.pretrain import load_pretrained_encoders


def _config() -> ModelConfig:
    return ModelConfig(
        input_size_px=32,
        fine_input_size_px=16,
        patch_size_px=8,
        embed_dim=32,
        depth=1,
        num_heads=4,
        local_readout="cls",
        context_readout="cls",
    )


@pytest.mark.parametrize(
    ("encoder_architecture", "readout", "use_fine_branch", "fusion_architecture"),
    [
        ("separate_patch", "center_2x", True, "concat"),
        ("shared_scale_aware", "cls", True, "concat"),
        ("shared_scale_aware", "center_2x", True, "concat"),
        ("separate_scale_aware", "cls", True, "concat"),
        ("shared_scale_aware", "cls", False, "concat"),
        ("shared_scale_aware", "cls", True, "gated_fine"),
    ],
)
def test_transfer_architecture_forward_shapes(
    encoder_architecture: str,
    readout: str,
    use_fine_branch: bool,
    fusion_architecture: str,
):
    config = replace(
        _config(),
        encoder_architecture=encoder_architecture,
        local_readout=readout,
        context_readout=readout,
        use_fine_branch=use_fine_branch,
        fusion_architecture=fusion_architecture,
    )
    model = MultitaskDapiVit(config)
    batch = {
        "local_image": torch.rand(2, 1, 32, 32),
        "context_image": torch.rand(2, 1, 32, 32),
    }
    if use_fine_branch:
        batch["fine_image"] = torch.rand(2, 1, 16, 16)
    outputs = model(batch)
    assert outputs["predicted_axis_coordinate"].shape == (2,)
    assert outputs["predicted_epithelial_distance_clipped_1p0"].shape == (2,)


@pytest.mark.parametrize("encoder_architecture", ["shared_scale_aware", "separate_scale_aware"])
def test_legacy_pretraining_load_keeps_cls_and_scale_embeddings(
    tmp_path: Path,
    encoder_architecture: str,
):
    config = replace(_config(), encoder_architecture=encoder_architecture)
    source = LegacyDapiVisionTransformer(config)
    with torch.no_grad():
        source.cls_token.normal_()
        source.scale_embeddings.normal_()
    checkpoint = tmp_path / "legacy_pretraining.pt"
    torch.save({"encoder_state_dict": source.state_dict()}, checkpoint)

    model = MultitaskDapiVit(config)
    loaded = load_pretrained_encoders(model, checkpoint)
    encoders = (
        [model.shared_encoder]
        if encoder_architecture == "shared_scale_aware"
        else [model.local_encoder, model.context_encoder]
    )
    for encoder in encoders:
        assert torch.equal(encoder.cls_token, source.cls_token)
        assert torch.equal(encoder.scale_embeddings, source.scale_embeddings)
    assert loaded


def test_separate_patch_can_retain_legacy_scale_embeddings(tmp_path: Path):
    source = LegacyDapiVisionTransformer(_config())
    with torch.no_grad():
        source.scale_embeddings.normal_()
    checkpoint = tmp_path / "legacy_pretraining.pt"
    torch.save({"encoder_state_dict": source.state_dict()}, checkpoint)
    config = replace(
        _config(),
        encoder_architecture="separate_patch",
        retain_scale_embeddings=True,
        local_readout="center_2x",
        context_readout="center_2x",
    )
    model = MultitaskDapiVit(config)
    load_pretrained_encoders(model, checkpoint)
    assert torch.equal(model.local_encoder.scale_embeddings, source.scale_embeddings)
    assert torch.equal(model.context_encoder.scale_embeddings, source.scale_embeddings)


def test_default_architecture_does_not_add_transfer_only_state_keys():
    state_keys = set(MultitaskDapiVit(_config()).state_dict())
    assert not any(key.startswith("shared_encoder.") for key in state_keys)
    assert "fine_gate" not in state_keys


@pytest.mark.parametrize(
    ("use_local_branch", "use_context_branch", "batch_key"),
    [(True, False, "local_image"), (False, True, "context_image")],
)
def test_shared_scale_aware_supports_single_scale_ablations(
    use_local_branch: bool,
    use_context_branch: bool,
    batch_key: str,
):
    config = replace(
        _config(),
        encoder_architecture="shared_scale_aware",
        use_local_branch=use_local_branch,
        use_context_branch=use_context_branch,
        use_fine_branch=False,
        local_readout="center_2x",
        context_readout="center_2x",
    )
    model = MultitaskDapiVit(config)
    outputs = model({batch_key: torch.rand(2, 1, 32, 32)})
    assert outputs["predicted_axis_coordinate"].shape == (2,)
    assert model.shared_encoder is not None
