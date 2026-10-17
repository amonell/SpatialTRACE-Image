#!/usr/bin/env python3
"""Create compact Figure 1 inputs from frozen model outputs and raw images.

Only deterministic cropping, subsetting, and descriptive aggregation occur here.
No model fitting, checkpoint selection, inference, or ablation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from paper_paths import Path, lock_record, rendering_script
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import tifffile
import zarr

PEYER_LIMITS_UM = (3507.95, 4404.70, 5359.04, 6255.79)
RARi_LIMITS_PX = (17000, 23000, 48000, 52500)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def crop_tiff_pixels(path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        height, width = tif.series[0].levels[0].shape[-2:]
        x0, x1, y0, y1 = map(int, window)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Crop window {window} is outside image bounds {(width, height)}")
        store = tif.series[0].levels[0].aszarr()
        try:
            root = zarr.open(store, mode="r")
            image = root["0"] if hasattr(root, "keys") and "0" in root else root
            crop = np.asarray(image[y0:y1, x0:x1])
        finally:
            store.close()
    return crop


def crop_tiff_um(
    path: Path,
    limits_um: tuple[float, float, float, float],
    pixel_size_um: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    window = (
        int(np.floor(limits_um[0] / pixel_size_um)),
        int(np.ceil(limits_um[1] / pixel_size_um)),
        int(np.floor(limits_um[2] / pixel_size_um)),
        int(np.ceil(limits_um[3] / pixel_size_um)),
    )
    return crop_tiff_pixels(path, window), window


def subset_xy(
    cells: pd.DataFrame,
    limits_um: tuple[float, float, float, float],
    *,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    x0, x1, y0, y1 = limits_um
    return cells[
        cells[x_column].between(x0, x1)
        & cells[y_column].between(y0, y1)
    ].copy()


def protect(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite:\n" + "\n".join(existing))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-input-dir", type=Path, required=True)
    parser.add_argument("--day90-dapi", type=Path, required=True)
    parser.add_argument("--peyer-predictions", type=Path, required=True)
    parser.add_argument("--vit-cells-manifest", type=Path, required=True)
    parser.add_argument("--panel-d-region-lock", type=Path, required=True)
    parser.add_argument("--rari-predictions", type=Path, required=True)
    parser.add_argument("--frozen-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    graph_dir = args.graph_input_dir.expanduser().resolve()
    day90_dapi = require(args.day90_dapi)
    peyer_path = require(args.peyer_predictions)
    vit_cells_path = require(args.vit_cells_manifest)
    region_lock_path = require(args.panel_d_region_lock)
    rari_path = require(args.rari_predictions)
    registry_path = require(args.frozen_registry)
    out = args.output_dir.expanduser().resolve()
    outputs = {
        "day90_training_dapi": out / "day90_training_dapi.npy",
        "day90_inference_dapi": out / "day90_inference_dapi.npy",
        "day90_training_graph_cells": out / "day90_training_graph_cells.parquet",
        "day90_inference_vit_cells": out / "day90_inference_vit_cells.csv",
        "day90_peyer_dapi": out / "day90_peyer_dapi.npy",
        "peyer_cells": out / "peyer_region_cells.parquet",
        "rari_zoom": out / "rari_zoom_predictions.parquet",
        "manifest": out / "figure1_input_manifest.json",
    }
    protect(list(outputs.values()), args.overwrite)

    registry = json.loads(registry_path.read_text())
    region_lock = json.loads(region_lock_path.read_text())
    peyer_model = registry["models"]["peyer_graph_classifier"]
    xenium_model = registry["models"]["xenium_image_coordinate_model"]
    if_model = registry["models"]["if_image_coordinate_model"]
    if sha256(peyer_path) != peyer_model["predictions"]["sha256"]:
        raise RuntimeError("Peyer's patch predictions do not match the frozen registry.")
    if sha256(rari_path) != if_model["whole_tissue_predictions"][1]["sha256"]:
        raise RuntimeError("RARi whole-tissue predictions do not match the frozen registry.")

    section_spec = region_lock["source_section"]
    if Path(section_spec["dapi_path"]).resolve() != day90_dapi:
        raise RuntimeError("The DAPI input does not match the locked Figure 1d source section.")
    pixel_size_um = float(section_spec["pixel_size_um"])
    training_region = region_lock["regions"]["paired_training_example"]
    inference_region = region_lock["regions"]["unseen_image_example"]
    training_window = tuple(map(int, training_region["pixel_window_xyxy"]))
    inference_window = tuple(map(int, inference_region["pixel_window_xyxy"]))
    training_limits = tuple(map(float, training_region["limits_um_xyxy"]))
    inference_limits = tuple(map(float, inference_region["limits_um_xyxy"]))

    np.save(outputs["day90_training_dapi"], crop_tiff_pixels(day90_dapi, training_window))
    np.save(outputs["day90_inference_dapi"], crop_tiff_pixels(day90_dapi, inference_window))
    peyer_crop, peyer_window = crop_tiff_um(day90_dapi, PEYER_LIMITS_UM, pixel_size_um)
    np.save(outputs["day90_peyer_dapi"], peyer_crop)

    section_cells_path = require(graph_dir / "section_cells.parquet")
    section_cells = pd.read_parquet(section_cells_path)
    training_graph = subset_xy(
        section_cells[section_cells["batch"].astype(str).eq(str(section_spec["batch"]))],
        training_limits,
        x_column="x",
        y_column="y",
    )
    if training_graph.empty:
        raise RuntimeError("Locked Figure 1d training region contains no graph-coordinate cells.")
    training_graph.to_parquet(outputs["day90_training_graph_cells"], index=False)

    vit_cells = pd.read_csv(vit_cells_path)
    inference_vit = subset_xy(
        vit_cells[vit_cells["source_id"].astype(str).eq(str(section_spec["vit_source_id"]))],
        tuple(map(float, inference_window)),
        x_column="centroid_x_fullres_px",
        y_column="centroid_y_fullres_px",
    )
    if inference_vit.empty:
        raise RuntimeError("Locked Figure 1d inference region contains no ViT-manifest cells.")
    inference_vit.to_csv(outputs["day90_inference_vit_cells"], index=False)

    peyer = pd.read_csv(peyer_path, usecols=["global_index", "batch", "x_spatial", "y_spatial", "gat_peyer_probability", "gat_peyer_class", "peyer_manual_label", "split"])
    peyer = peyer[
        peyer.batch.astype(str).eq("day90_SI_r2")
        & peyer.x_spatial.between(PEYER_LIMITS_UM[0], PEYER_LIMITS_UM[1])
        & peyer.y_spatial.between(PEYER_LIMITS_UM[2], PEYER_LIMITS_UM[3])
    ].copy()
    if peyer.empty:
        raise RuntimeError("The prespecified held-out Peyer's patch crop contains no cells.")
    peyer.to_parquet(outputs["peyer_cells"], index=False)

    rari = pd.read_parquet(rari_path)
    rx0, rx1, ry0, ry1 = RARi_LIMITS_PX
    rari_zoom = rari[rari.centroid_x_fullres_px.between(rx0, rx1) & rari.centroid_y_fullres_px.between(ry0, ry1)].copy()
    rari_zoom.to_parquet(outputs["rari_zoom"], index=False)

    graph_manifest = require(graph_dir / "graph_component_inputs_manifest.json")
    critical_inputs = [
        registry_path,
        graph_manifest,
        section_cells_path,
        peyer_path,
        vit_cells_path,
        region_lock_path,
        rari_path,
        day90_dapi,
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "operation": "deterministic Figure 1 image cropping and table subsetting",
        "no_model_inference": True,
        "frozen_checkpoints": {
            "xenium_representation_pretraining_v2": xenium_model["checkpoint"],
            "if_production_v2": if_model["checkpoint"],
            "peyer_graph_classifier": peyer_model["checkpoint"],
        },
        "crop_definitions": {
            "day90_training_limits_um": training_limits,
            "day90_training_pixel_window": training_window,
            "day90_inference_limits_um": inference_limits,
            "day90_inference_pixel_window": inference_window,
            "day90_peyer_limits_um": PEYER_LIMITS_UM,
            "day90_peyer_pixel_window": peyer_window,
            "rari_pixel_window": RARi_LIMITS_PX,
        },
        "inputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in critical_inputs
        ],
        "outputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for key, path in outputs.items() if key != "manifest"],
        "counts": {
            "figure1d_training_graph_cells": len(training_graph),
            "figure1d_inference_vit_cells": len(inference_vit),
            "peyer_crop_cells": len(peyer),
            "rari_crop_cells": len(rari_zoom),
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    print(outputs["manifest"])


if __name__ == "__main__":
    main()
