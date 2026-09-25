import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

from crypt_villus_vit.cli import build_parser, main
from crypt_villus_vit.model import ModelConfig
from crypt_villus_vit.predict import CellCropDataset
from crypt_villus_vit.prepare_crops import INPUT_PROTOCOL, ShardWriter, prepare_crops
from crypt_villus_vit.prepared import PreparedCellCropDataset, PreparedPairedPretrainingDataset
from crypt_villus_vit.prepared import validate_prepared_crop_configuration
from crypt_villus_vit.sources import load_source_manifest


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(13)
    records = []
    for source, calibration in [("b", .325), ("a", .2125)]:
        path = tmp_path / f"{source}.ome.tif"
        image = rng.integers(0, 6000, (145, 163), dtype=np.uint16)
        with tifffile.TiffWriter(path, ome=True) as writer:
            writer.write(image, subifds=1, tile=(32, 32), compression="deflate",
                         metadata={"axes": "YX"}, photometric="minisblack")
            writer.write(image[::2, ::2], subfiletype=1, tile=(32, 32),
                         compression="deflate", photometric="minisblack")
        records.append(dict(source_id=source, image_path=path.name,
                            pixel_size_um=calibration, section_id=source))
    pd.DataFrame(records).to_csv(tmp_path / "sources.csv", index=False)
    pd.DataFrame(dict(cell_id=["z", "y", "x", "w", "v"], source_id=["b", "a", "b", "a", "b"],
                      centroid_x_fullres_px=[160.5, 0, 65.1, 44, 97],
                      centroid_y_fullres_px=[143, 0, 50.2, 41, 88],
                      target_axis=[.1, .2, .3, .4, .5],
                      epithelial_distance_clipped_1p0=[.8, .7, .6, .5, .4],
                      split=["train", "validation", "train", "validation", "train"])
                 ).to_csv(tmp_path / "cells.csv", index=False)
    return tmp_path


def arguments(root, command, destination, *extra):
    return [command, "--source-manifest", str(root / "sources.csv"),
            "--cells-csv", str(root / "cells.csv"), "--output-dir", str(root / destination),
            "--local-crop-px", "32", "--context-crop-px", "96", "--input-size-px", "32",
            "--rows-per-shard", "3", "--batch-size", "2", "--num-workers", "0", *extra]


BACKENDS = ["cached", pytest.param("rust", marks=pytest.mark.skipif(
    importlib.util.find_spec("spatialtrace_image_rust") is None, reason="Optional Rust extra not installed"))]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("supervised", [False, True])
def test_shards_match_original_crops_and_keep_row_identity(inputs, backend, workers, supervised):
    command = "prepare-supervised" if supervised else "prepare-pretraining"
    extra = ["--fine-crop-px", "16", "--fine-input-size-px", "16"] if supervised else []
    args = build_parser().parse_args(arguments(inputs, command, "prepared", *extra,
             "--crop-backend", backend, "--num-workers", str(workers)))
    metadata = prepare_crops(args, supervised=supervised)
    config = ModelConfig(local_crop_px=32, context_crop_px=96, input_size_px=32,
                         fine_crop_px=16, fine_input_size_px=16, use_fine_branch=supervised)
    original_rows = pd.read_csv(inputs / "cells.csv")
    dataset = CellCropDataset(original_rows, load_source_manifest(inputs / "sources.csv"), config,
                              reference_pixel_size_um=.325, input_protocol=INPUT_PROTOCOL)
    originals = [dataset[i] for i in range(len(dataset))]
    try:
        root = inputs / "prepared"
        manifest = pd.read_csv(root / "manifest.csv")
        if supervised:
            pd.testing.assert_frame_equal(manifest.drop(columns="prepared_index"), original_rows)
            prepared = PreparedCellCropDataset(manifest, root / "metadata.json", config)
            for i, original in enumerate(originals):
                for branch in ("local", "context", "fine"):
                    np.testing.assert_array_equal(prepared[i][f"{branch}_image"], original[f"{branch}_image"])
            expected_groups = {branch: [item[f"{branch}_image"] for item in originals]
                               for branch in ("local", "context", "fine")}
        else:
            assert manifest.center_id.tolist() == [f"{sid}:{i}" for i, sid in enumerate(original_rows.source_id)
                                                    for _ in range(2)]
            assert manifest.prepared_index.tolist() == list(range(10))
            assert manifest.scale_id.tolist() == [0, 1] * 5
            prepared = PreparedPairedPretrainingDataset(manifest, root / "metadata.json")
            assert len(prepared) == 5
            expected_groups = {"pairs": [item[f"{branch}_image"] for item in originals
                                         for branch in ("local", "context")]}
            for i, pair in prepared.pairs.iterrows():
                for scale, branch in enumerate(("local", "context")):
                    np.testing.assert_array_equal(prepared[i][f"{branch}_teacher_image"],
                                                  originals[int(pair[scale]) // 2][f"{branch}_image"])
        for branch, images in expected_groups.items():
            paths = metadata["shard_paths" if branch == "pairs" else f"{branch}_shard_paths"]
            for shard, filename in enumerate(paths):
                expected = np.rint(np.stack(images[shard * 3:(shard + 1) * 3]) * 255).astype(np.uint8)
                actual = np.load(root / filename)
                np.testing.assert_array_equal(actual, expected)
                # Same .npy bytes as the former sequential np.save writer, including the last shard.
                oracle = inputs / "oracle.npy"
                np.save(oracle, expected)
                assert (root / filename).read_bytes() == oracle.read_bytes()
        assert not list(root.glob("*.partial"))
    finally:
        for reader in dataset.extractor.backends.values():
            reader.close()


@pytest.mark.parametrize("extra,match", [
    (["--rows-per-shard", "0"], "positive"),
    (["--num-workers", "-1"], "nonnegative"),
    (["--reference-pixel-size-um", "nan"], "finite"),
    (["--batch-size", "0"], "positive"),
])
def test_invalid_settings_leave_no_output(inputs, extra, match):
    args = build_parser().parse_args(arguments(inputs, "prepare-pretraining", "bad", *extra))
    with pytest.raises(ValueError, match=match):
        prepare_crops(args)
    assert not (inputs / "bad").exists()


@pytest.mark.parametrize("case,match", [("empty", "no rows"), ("test", "test rows"),
                                       ("unknown", "Unknown source"), ("nan", "finite")])
def test_invalid_rows(inputs, case, match):
    rows = pd.read_csv(inputs / "cells.csv")
    if case == "empty":
        rows = rows.iloc[:0]
    elif case == "test":
        rows.loc[0, "split"] = " Test "
    elif case == "unknown":
        rows.loc[0, "source_id"] = "missing"
    else:
        rows.loc[0, "centroid_x_fullres_px"] = np.nan
    rows.to_csv(inputs / "cells.csv", index=False)
    args = build_parser().parse_args(arguments(inputs, "prepare-pretraining", "bad"))
    with pytest.raises(ValueError, match=match):
        prepare_crops(args)
    assert not (inputs / "bad").exists()


def test_cli_provenance_no_overwrite_and_no_fine(inputs):
    command = arguments(inputs, "prepare-supervised", "supervised", "--no-fine-branch")
    main(command)
    root = inputs / "supervised"
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["fine_shard_paths"] == []
    assert json.loads((root / "run_provenance.json").read_text())["configuration"]["command"] == "prepare-supervised"
    with pytest.raises(FileExistsError):
        main(command)


def test_shard_writer_lru_and_incomplete_protection(tmp_path):
    writer = ShardWriter(tmp_path, "crop_", 11, 3, (1, 2, 2), max_open=1)
    order = np.array([10, 0, 6, 1, 9, 5, 2, 4, 8, 7, 3])
    for index in order:
        writer.write(np.array([index]), np.full((1, 1, 2, 2), index, np.uint8))
        assert len(writer.maps) <= 1
    with pytest.raises(ValueError, match="Duplicate"):
        writer.write(np.array([0]), np.zeros((1, 1, 2, 2), np.uint8))
    paths = writer.finish()
    assert np.concatenate([np.load(tmp_path / path) for path in paths])[:, 0, 0, 0].tolist() == list(range(11))
    incomplete = ShardWriter(tmp_path, "incomplete_", 2, 2, (1, 2, 2))
    incomplete.write(np.array([0]), np.zeros((1, 1, 2, 2), np.uint8))
    with pytest.raises(ValueError, match="incomplete"):
        incomplete.finish()
    incomplete.close()
    assert not (tmp_path / "incomplete_00000.npy").exists()


def test_generated_shards_train_and_predict(inputs):
    main(arguments(inputs, "prepare-supervised", "training_crops", "--fine-crop-px", "16",
                   "--fine-input-size-px", "16", "--num-workers", "2"))
    main(["train", "--source-manifest", str(inputs / "sources.csv"),
          "--supervised-manifest", str(inputs / "training_crops/manifest.csv"),
          "--prepared-supervised-metadata", str(inputs / "training_crops/metadata.json"),
          "--output-dir", str(inputs / "fit"), "--device", "cpu", "--epochs", "1", "--batch-size", "2",
          "--local-crop-px", "32", "--context-crop-px", "96", "--fine-crop-px", "16",
          "--input-size-px", "32", "--fine-input-size-px", "16", "--patch-size-px", "8",
          "--embed-dim", "16", "--depth", "1", "--num-heads", "2"])
    main(["predict-cells", "--source-manifest", str(inputs / "sources.csv"),
          "--cells-csv", str(inputs / "cells.csv"), "--checkpoint", str(inputs / "fit/best_crypt_villus_vit_model.pt"),
          "--output-dir", str(inputs / "predictions"), "--device", "cpu", "--no-scatter"])
    assert len(pd.read_csv(inputs / "predictions/predictions.csv")) == 5


def test_prepared_configuration_must_match_training(inputs):
    main(arguments(inputs, "prepare-supervised", "prepared"))
    path = inputs / "prepared/metadata.json"
    config = ModelConfig(local_crop_px=32, context_crop_px=96, input_size_px=32)
    assert validate_prepared_crop_configuration(path, config, input_protocol=INPUT_PROTOCOL,
                                                reference_pixel_size_um=None) == .325
    with pytest.raises(ValueError, match="input_size_px"):
        validate_prepared_crop_configuration(path, ModelConfig(local_crop_px=32, context_crop_px=96),
                                            input_protocol=INPUT_PROTOCOL, reference_pixel_size_um=.325)
    with pytest.raises(ValueError, match="reference_pixel_size_um"):
        validate_prepared_crop_configuration(path, config, input_protocol=INPUT_PROTOCOL,
                                            reference_pixel_size_um=.2125)


def test_generated_shards_pretrain(inputs):
    main(arguments(inputs, "prepare-pretraining", "paired"))
    main(["pretrain-representation", "--prepared-metadata", str(inputs / "paired/metadata.json"),
          "--output-dir", str(inputs / "representation"), "--epochs", "1", "--batch-size", "2",
          "--input-size-px", "32", "--patch-size-px", "8", "--embed-dim", "16", "--depth", "1",
          "--num-heads", "2", "--device", "cpu"])
    assert (inputs / "representation/dapi_encoder_pretrain.pt").is_file()
