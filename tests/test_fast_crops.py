import importlib.util

import numpy as np
import pandas as pd
import pytest
import tifffile

from crypt_villus_vit.fast_crops import CachedImagePyramid
from crypt_villus_vit.model import ModelConfig, MultitaskDapiVit, save_model_checkpoint
from crypt_villus_vit.predict import (
    CellCropDataset,
    SourceGroupedBatchSampler,
    predict_cells,
)
from crypt_villus_vit.production_crops import ImagePyramid, normalize_resize_quantize
from crypt_villus_vit.sources import SourceSpec

BACKENDS = ["cached", pytest.param("rust", marks=pytest.mark.skipif(
    importlib.util.find_spec("spatialtrace_image_rust") is None, reason="Optional Rust extra not installed"
))]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cache_bytes", [0, 128, 8192, 1024**2])
def test_tiled_pyramid_pixel_parity_and_cache_bound(tmp_path, backend, cache_bytes):
    rng = np.random.default_rng(43)
    image = rng.integers(0, 65536, (161, 193), dtype=np.uint16)
    path = tmp_path / "pyramid.ome.tif"
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(image, subifds=1, tile=(32, 32), compression="deflate",
                     metadata={"axes": "YX"}, photometric="minisblack")
        writer.write(image[::2, ::2], subfiletype=1, tile=(32, 32),
                     compression="deflate", photometric="minisblack")
    source = SourceSpec("a", path)
    ref = ImagePyramid(source)
    fast = CachedImagePyramid(source, backend=backend, cache_bytes=cache_bytes)
    for x, y in [(0, 0), (192.9, 160.9), (32.5, 33.5), (85.7, 82.1), (85.7, 82.1)]:
        for native, output in [(31, 32), (64, 32), (197, 48), (17, 16)]:
            np.testing.assert_array_equal(ref.crop(x, y, native, output), fast.crop(x, y, native, output))
            assert fast.cached_bytes <= cache_bytes
    if cache_bytes == 1024**2:
        assert fast.hits > 0 and fast.misses > 0
    fast.close()
    ref.close()
    assert fast.cached_bytes == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_channels_and_z_projection(tmp_path, backend):
    image = np.random.default_rng(3).integers(0, 5000, (2, 3, 64, 96), dtype=np.uint16)
    path = tmp_path / "channels.ome.tif"
    tifffile.imwrite(path, image, ome=True, metadata={"axes": "CZYX"},
                     tile=(32, 32), photometric="minisblack")
    ref = ImagePyramid(SourceSpec("a", path, channel_index=1))
    fast = CachedImagePyramid(SourceSpec("a", path, channel_index=1),
                              backend=backend, cache_bytes=1024**2)
    for x, y in [(10, 12), (55.1, 45), (95, 63)]:
        np.testing.assert_array_equal(ref.crop(x, y, 40, 24), fast.crop(x, y, 40, 24))
    fast.close()
    ref.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_numpy_and_special_value_normalization(tmp_path, backend):
    rng = np.random.default_rng(71)
    source = tmp_path / "image.npy"
    np.save(source, np.zeros((20, 20), dtype=np.float32))
    fast = CachedImagePyramid(SourceSpec("a", source), backend=backend, cache_bytes=1024)
    arrays = [rng.normal(size=(77, 53)).astype(np.float32), np.zeros((31, 31), np.float32),
              np.full((9, 9), 65535, np.float32), np.array([[np.nan, np.inf, -np.inf]], np.float32),
              np.array([[0., np.nan, 1., np.inf, -np.inf]], np.float32)]
    for exponent in (-12, -6, 6, 12):
        arrays.append((rng.normal(size=(37, 49)) * 10.**exponent).astype(np.float32))
    for _ in range(20):
        arrays.append(rng.integers(0, 65536, (97, 113)).astype(np.float32))
    for crop in arrays:
        for output in (16, 64):
            np.testing.assert_array_equal(normalize_resize_quantize(crop, output), fast.normalize_crop(crop, output))
    fast.close()


def test_spatial_order_visits_every_row_once():
    rows = pd.DataFrame({"source_id": ["b", "a", "a", "b", "a"],
                         "centroid_x_fullres_px": [3000, 0, 2100, 0, 50],
                         "centroid_y_fullres_px": [100, 0, 0, 100, 60]})
    sampler = SourceGroupedBatchSampler(rows, batch_size=2, shuffle=False, spatial_order=True)
    batches = list(sampler)
    assert [i for batch in batches for i in batch] == [1, 4, 2, 3, 0]
    assert all(rows.iloc[batch].source_id.nunique() == 1 for batch in batches)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("task_type", ["axis_regression", "binary_classification"])
def test_predictions_keep_row_identity_and_values(tmp_path, backend, task_type):
    import torch
    torch.set_num_threads(1)
    image = np.random.default_rng(3).integers(0, 6000, (96, 96), dtype=np.uint16)
    path = tmp_path / "raw.tif"
    tifffile.imwrite(path, image, tile=(32, 32), photometric="minisblack")
    sources = {"a": SourceSpec("a", path, pixel_size_um=.325)}
    rows = pd.DataFrame({"cell_id": ["c", "a", "b"], "source_id": ["a"] * 3,
                         "centroid_x_fullres_px": [83, 1, 43],
                         "centroid_y_fullres_px": [89, 2, 45]})
    config = ModelConfig(local_crop_px=32, context_crop_px=64, fine_crop_px=16,
                         input_size_px=32, fine_input_size_px=16, patch_size_px=8,
                         embed_dim=16, depth=1, num_heads=2)
    checkpoint = tmp_path / "model.pt"
    save_model_checkpoint(MultitaskDapiVit(config), checkpoint, metadata={
        "input_protocol": "direct_pyramid_crop_normalize_resize_uint8_v1",
        "reference_pixel_size_um": .325,
        "task_type": task_type,
    })
    for name, workers in [("reference", 0), (backend, 0), (backend + "_workers", 2)]:
        predict_cells(rows, sources=sources, checkpoint_path=checkpoint, output_dir=tmp_path / name,
                      batch_size=2, device="cpu", num_workers=workers,
                      crop_backend=backend if name != "reference" else "reference")
    ref = pd.read_csv(tmp_path / "reference" / "predictions.csv")
    for name in (backend, backend + "_workers"):
        result = pd.read_csv(tmp_path / name / "predictions.csv")
        assert result.cell_id.tolist() == rows.cell_id.tolist()
        columns = (["predicted_axis_coordinate", "predicted_epithelial_distance_clipped_1p0"]
                   if task_type == "axis_regression" else ["peyer_probability", "peyer_logit"])
        for column in columns:
            np.testing.assert_allclose(result[column], ref[column], atol=1e-6)


def test_optimized_backend_rejects_legacy_protocol():
    with pytest.raises(ValueError, match="production input protocol"):
        CellCropDataset(pd.DataFrame(), {}, ModelConfig(), crop_backend="cached")


@pytest.mark.parametrize("backend", BACKENDS)
def test_switching_sources_preserves_calibration_and_values(tmp_path, backend):
    rng = np.random.default_rng(1)
    sources = {}
    for i, source_id in enumerate(("a", "b")):
        path = tmp_path / f"{source_id}.npy"
        np.save(path, rng.integers(0, 5000, (61, 73), dtype=np.uint16))
        sources[source_id] = SourceSpec(source_id, path, pixel_size_um=(.325, .2125)[i])
    rows = pd.DataFrame({"source_id": ["a", "b", "a"], "centroid_x_fullres_px": [30.1] * 3,
                         "centroid_y_fullres_px": [35.2] * 3})
    config = ModelConfig(local_crop_px=32, context_crop_px=64, fine_crop_px=16,
                         input_size_px=32, fine_input_size_px=16)
    options = {"reference_pixel_size_um": .325,
               "input_protocol": "direct_pyramid_crop_normalize_resize_uint8_v1"}
    ref = CellCropDataset(rows, sources, config, **options)
    fast = CellCropDataset(rows, sources, config, crop_backend=backend, **options)
    for i in range(3):
        original, actual = ref[i], fast[i]
        for key in ("local_image", "context_image", "fine_image"):
            np.testing.assert_array_equal(original[key], actual[key])
        assert len(fast.extractor.backends) == 1
    for dataset in (ref, fast):
        for image in dataset.extractor.backends.values():
            image.close()
