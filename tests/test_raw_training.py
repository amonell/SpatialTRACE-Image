import importlib.util
import json
import multiprocessing
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
from torch.utils.data import DataLoader

from crypt_villus_vit.augmentation import IntensityAugmentationConfig
from crypt_villus_vit.cli import main
from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.predict import CellCropDataset, SourceGroupedBatchSampler, _collate
from crypt_villus_vit.raw_training import (
    RamCropCache, RawTrainingDataset, close_training_loader, make_raw_training_loader, warm_ram_cache,
)
from crypt_villus_vit.sources import SourceSpec
from crypt_villus_vit.train import _set_dataset_epoch, train_model

RUST = pytest.param("rust", marks=pytest.mark.skipif(
    importlib.util.find_spec("spatialtrace_image_rust") is None, reason="Optional Rust extra not installed"))
PROTOCOL = "direct_pyramid_crop_normalize_resize_uint8_v1"


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    sources = {}
    for i, sid in enumerate(("a", "b")):
        path = tmp_path / f"{sid}.tif"
        tifffile.imwrite(path, np.random.default_rng(i).integers(0, 65535, (97, 113), dtype=np.uint16),
                         tile=(32, 32), compression="deflate", photometric="minisblack")
        sources[sid] = SourceSpec(sid, path, pixel_size_um=(.325, .2125)[i])
    rows = pd.DataFrame(dict(source_id=["b", "a"] * 6,
                             centroid_x_fullres_px=[1, 105, 40, 52, 19, 38, 44, 55, 61, 66, 70, 0],
                             centroid_y_fullres_px=[2, 90, 45, 71, 22, 31, 50, 55, 60, 66, 70, 0],
                             target_axis=[0, 1] * 6, epithelial_distance_clipped_1p0=np.linspace(0, 1, 12),
                             split=["train"] * 8 + ["validation"] * 2 + ["test"] * 2))
    config = ModelConfig(local_crop_px=32, context_crop_px=64, fine_crop_px=16,
                         input_size_px=32, fine_input_size_px=16, patch_size_px=8,
                         embed_dim=16, depth=1, num_heads=2, dropout=.2,
                         encoder_architecture="shared_scale_aware", retain_scale_embeddings=True,
                         local_readout="center_4x", context_readout="center_4x")
    return rows, sources, config


@pytest.mark.parametrize("backend", ["cached", RUST])
@pytest.mark.parametrize("workers", [0, 2])
def test_three_epochs_identical_order_pixels_augmentation_and_rng(fixture, backend, workers):
    rows, sources, config = fixture
    aug = IntensityAugmentationConfig(probability=1, brightness_delta=.15,
                                      contrast_range=(.8, 1.2), gaussian_blur_sigma_range=(.1, .4), seed=7)
    options = dict(reference_pixel_size_um=.325, input_protocol=PROTOCOL)
    reference = DataLoader(CellCropDataset(rows, sources, config, intensity_augmentation=aug, **options),
                           batch_sampler=SourceGroupedBatchSampler(rows, batch_size=3, shuffle=True, seed=4),
                           collate_fn=_collate)
    cache = RamCropCache(len(rows), config, 1)
    loader = make_raw_training_loader(rows, sources, config, batch_size=3, seed=4, shuffle=True,
                                     num_workers=workers, device="cpu", crop_backend=backend,
                                     tile_cache_mib=1, intensity_augmentation=aug, cache=cache, **options)
    ref, actual, before_augmentation = [], [], []
    try:
        for source, results in [(reference, ref), (loader, actual)]:
            torch.manual_seed(37)
            for epoch in range(3):
                _set_dataset_epoch(source, epoch)
                results.append((list(source), torch.rand(5)))
        for (expected_batches, expected_rng), (batches, rng) in zip(ref, actual):
            assert torch.equal(expected_rng, rng)
            for expected, batch in zip(expected_batches, batches):
                assert expected.keys() == batch.keys()
                for key in batch:
                    assert torch.equal(expected[key], batch[key]), key
        for epoch_batches, _ in actual:
            before_augmentation.append(next(item["local_image"][i] for item in epoch_batches
                                             for i, index in enumerate(item["index"]) if index == 0))
        assert not torch.equal(before_augmentation[0], before_augmentation[1])
        assert int(cache.ready.sum()) == len(rows)
    finally:
        close_training_loader(reference)
        close_training_loader(loader)


@pytest.mark.parametrize("backend,warm,cache_mib,task", [
    ("cached", True, 1, "axis_regression"),
    ("cached", False, 1, "axis_regression"),
    ("cached", True, 0, "binary_classification"),
])
def test_training_matches_reference_weights_and_selection(tmp_path, fixture, backend, warm, cache_mib, task):
    rows, sources, config = fixture
    settings = dict(sources=sources, config=config, epochs=3, batch_size=2, device="cpu", seed=41,
                    reference_pixel_size_um=.325, task_type=task,
                    augment_intensity_probability=1, augment_brightness_delta=.1,
                    augment_contrast_range=(.85, 1.15), augment_gaussian_blur_sigma_range=(.1, .4))
    original = train_model(rows, output_dir=tmp_path / "reference", **settings)
    fast = train_model(rows, output_dir=tmp_path / "fast", crop_backend=backend, num_workers=2,
                       ram_crop_cache_mib=cache_mib, warm_crop_cache=warm, **settings)
    assert fast["best_epoch"] == original["best_epoch"]
    assert fast["selection_value"] == original["selection_value"]
    assert fast["train_row_count"] == 8 and fast["validation_row_count"] == 2
    assert fast["data_loading"]["ram_crop_cache_rows"] == (10 if cache_mib else 0)
    left = torch.load(original["checkpoint_path"], weights_only=True)["model_state_dict"]
    right = torch.load(fast["checkpoint_path"], weights_only=True)["model_state_dict"]
    assert all(torch.equal(left[key], right[key]) for key in left)
    a, b = pd.read_csv(original["history_csv"]), pd.read_csv(fast["history_csv"])
    columns = [name for name in a if not name.endswith("_seconds")]
    pd.testing.assert_frame_equal(a[columns], b[columns], check_exact=True)
    assert not list((tmp_path / "fast").glob("*.npy"))


def test_warm_cache_bounds_and_no_redecode(fixture, monkeypatch):
    rows, sources, config = fixture
    # Enough for only three rows; test partial-cache fallback too.
    cache = RamCropCache(len(rows), config, 0.007)
    assert 0 < cache.capacity < len(rows)
    assert cache.allocated_bytes <= 0.007 * 1024**2
    options = dict(input_protocol=PROTOCOL, reference_pixel_size_um=.325, crop_backend="cached", tile_cache_mib=1)
    state = torch.get_rng_state().clone()
    warm_ram_cache(cache, rows.iloc[:8], rows.iloc[8:10], sources, config, num_workers=0, **options)
    assert torch.equal(state, torch.get_rng_state())
    dataset = RawTrainingDataset(rows, sources, config, cache=cache, **options)
    def fail(_):
        raise RuntimeError("cache miss")
    monkeypatch.setattr(CellCropDataset, "__getitem__", lambda self, index: fail(index))
    assert dataset[0]["local_image"].shape == (1, 32, 32)
    with pytest.raises(RuntimeError, match="cache miss"):
        dataset[len(rows) - 1]
    dataset.close()


def test_raw_training_cli(tmp_path):
    main(["create-demo", "--output-dir", str(tmp_path / "demo")])
    main(["train", "--source-manifest", str(tmp_path / "demo/sources.csv"),
          "--supervised-manifest", str(tmp_path / "demo/labels.csv"), "--output-dir", str(tmp_path / "fit"),
          "--crop-backend", "cached", "--ram-crop-cache-mib", "1", "--num-workers", "2",
          "--epochs", "2", "--device", "cpu", "--input-size-px", "32", "--fine-input-size-px", "16",
          "--local-crop-px", "32", "--context-crop-px", "64", "--fine-crop-px", "16",
          "--patch-size-px", "8", "--embed-dim", "16", "--depth", "1", "--num-heads", "2"])
    provenance = json.loads((tmp_path / "fit/run_provenance.json").read_text())
    assert provenance["configuration"]["ram_crop_cache_mib"] == 1


@pytest.mark.parametrize("settings", [dict(crop_backend="cached", prepared_supervised_metadata="unused"),
                                    dict(ram_crop_cache_mib=1), dict(crop_backend="cached", num_workers=-1),
                                    dict(crop_backend="cached", input_protocol="legacy_float_v1")])
def test_reject_incompatible_modes(tmp_path, fixture, settings):
    rows, sources, config = fixture
    with pytest.raises(ValueError):
        train_model(rows, sources=sources, config=config, output_dir=tmp_path / "bad", **settings)
    assert not (tmp_path / "bad").exists()


def test_failure_shuts_down_persistent_workers(tmp_path, fixture, monkeypatch):
    import crypt_villus_vit.train as training
    rows, sources, config = fixture
    before = {p.pid for p in multiprocessing.active_children()}
    def fail(model, loader, **kwargs):
        next(iter(loader))
        raise RuntimeError("deliberate epoch failure")
    monkeypatch.setattr(training, "_run_epoch", fail)
    with pytest.raises(RuntimeError, match="deliberate epoch failure"):
        training.train_model(rows, sources=sources, config=config, output_dir=tmp_path / "failed",
                             crop_backend="cached", num_workers=2, device="cpu", reference_pixel_size_um=.325)
    assert {p.pid for p in multiprocessing.active_children()} == before


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shared-memory guard")
def test_insufficient_shared_memory_fails_before_allocation(fixture, monkeypatch):
    import crypt_villus_vit.raw_training as raw
    from types import SimpleNamespace
    _, _, config = fixture
    monkeypatch.setattr(raw.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="/dev/shm"):
        RamCropCache(100, config, 1)
