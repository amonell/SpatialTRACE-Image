#!/usr/bin/env python3
"""Extract the exact IF region imported into the manually assembled Figure 3a."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, protect_outputs, require_file, write_provenance  # noqa: E402

EXPECTED_NUCLEI_TABLE = "ef4f043cf53e69e0fee0e93fae7ed0a803756638ebe7c408aca2dc48fc3b8cd9"
EXPECTED_PRETRAINING_MANIFEST = "33721684b3b2619e253d1bd294926cea7fef7bd72b70e45dbdc5cfe90acb1339"
EXPECTED_VSI = "978bd4fc3aa45b30a2686fbf70c42bf51d53994971b861ff404299274b20cc11"
EXPECTED_METADATA = "edf5fa06ac0619a60542a138400ed41017193ef84605e2f199ac12d7b583d9bf"

SOURCE_ID = "if:cont1"
SECTION_ID = "cont1"
CONDITION = "Cont1"
NUCLEUS_LABEL_ID = 352082
EXPECTED_CENTER_X = 15709.439790575916
EXPECTED_CENTER_Y = 22534.952879581157
EXPECTED_PIXEL_SIZE_UM = 0.3250059901547313
CROP_SIZES = {"overview": 4096, "context": 2048, "local": 512, "fine": 128}


def tree_record(path: Path, role: str) -> dict[str, object]:
    """Hash a microscopy sidecar directory as ordered relative paths plus bytes."""
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    total_bytes = 0
    files = [item for item in sorted(resolved.rglob("*")) if item.is_file()]
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                total_bytes += len(chunk)
                digest.update(chunk)
    return {
        "role": role,
        "path": str(resolved),
        "sha256_tree": digest.hexdigest(),
        "bytes": total_bytes,
        "file_count": len(files),
    }


def centered_bounds(center_x: float, center_y: float, crop_px: int) -> dict[str, int]:
    half = int(crop_px // 2)
    top = int(np.floor(center_y)) - half
    left = int(np.floor(center_x)) - half
    return {"top": top, "bottom": top + crop_px, "left": left, "right": left + crop_px}


def mask_crop(mask: np.ndarray, bounds: dict[str, int], label_id: int) -> np.ndarray:
    height = bounds["bottom"] - bounds["top"]
    width = bounds["right"] - bounds["left"]
    result = np.zeros((height, width), dtype=bool)
    source_top = max(bounds["top"], 0)
    source_bottom = min(bounds["bottom"], int(mask.shape[0]))
    source_left = max(bounds["left"], 0)
    source_right = min(bounds["right"], int(mask.shape[1]))
    destination_top = source_top - bounds["top"]
    destination_left = source_left - bounds["left"]
    result[
        destination_top : destination_top + source_bottom - source_top,
        destination_left : destination_left + source_right - source_left,
    ] = np.asarray(mask[source_top:source_bottom, source_left:source_right]) == int(label_id)
    return result


def load_pretraining_inputs(
    manifest_path: Path,
    shard_dir: Path,
    center_id: str,
    rows_per_shard: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    manifest = pd.read_csv(manifest_path)
    selected = manifest.loc[manifest["center_id"].astype(str).eq(center_id)].copy()
    if set(selected["scale_name"].astype(str)) != {"local", "context"} or len(selected) != 2:
        raise RuntimeError(f"Expected exactly one local and one context pretraining row for {center_id}")
    inputs: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for row in selected.sort_values("scale_id").itertuples(index=False):
        prepared_index = int(row.prepared_index)
        shard_index, offset = divmod(prepared_index, rows_per_shard)
        shard_path = require_file(shard_dir / f"pretraining_shard_{shard_index:05d}.npy")
        shard = np.load(shard_path, mmap_mode="r")
        if tuple(shard.shape[1:]) != (8, 256, 256):
            raise RuntimeError(f"Unexpected pretraining shard shape {shard.shape}: {shard_path}")
        scale_name = str(row.scale_name)
        inputs[f"{scale_name}_model_input"] = np.asarray(shard[offset, 0], dtype=np.uint8).copy()
        records.append(file_record(shard_path, f"{scale_name}_pretraining_shard"))
        rows.append(
            {
                "row_id": str(row.row_id),
                "scale_name": scale_name,
                "scale_id": int(row.scale_id),
                "prepared_index": prepared_index,
                "variant_index": 0,
                "native_crop_fullres_px": int(row.native_crop_fullres_px),
                "crop_width_um": float(row.crop_width_um),
                "model_input_size_px": int(row.model_input_size_px),
                "holdout": bool(row.holdout),
                "target_available": bool(row.target_available),
            }
        )
    return inputs, records, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-pipeline-src", type=Path, required=True)
    parser.add_argument("--nuclei-table", type=Path, required=True)
    parser.add_argument("--pretraining-manifest", type=Path, required=True)
    parser.add_argument("--pretraining-shard-dir", type=Path, required=True)
    parser.add_argument("--vsi", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--nuclei-mask", type=Path, required=True)
    parser.add_argument("--tissue-metadata", type=Path, required=True)
    parser.add_argument("--nucleus-label-id", type=int, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.nucleus_label_id != NUCLEUS_LABEL_ID:
        raise RuntimeError(f"Figure 3a is locked to nucleus label {NUCLEUS_LABEL_ID}")
    pipeline_src = args.if_pipeline_src.expanduser().resolve()
    if not pipeline_src.is_dir():
        raise FileNotFoundError(pipeline_src)
    sys.path.insert(0, str(pipeline_src))
    from if_processing_pipeline.spatial_if_multiscale_dapi import (  # noqa: PLC0415
        DapiCropManager,
        DapiSourceSpec,
        IfXeniumMultiscaleVitConfig,
    )

    nuclei_path = require_file(args.nuclei_table, EXPECTED_NUCLEI_TABLE)
    pretraining_manifest = require_file(args.pretraining_manifest, EXPECTED_PRETRAINING_MANIFEST)
    vsi_path = require_file(args.vsi, EXPECTED_VSI)
    mask_path = require_file(args.nuclei_mask)
    tissue_metadata = require_file(args.tissue_metadata, EXPECTED_METADATA)
    sidecar_dir = args.sidecar_dir.expanduser().resolve()

    nuclei = pd.read_csv(nuclei_path)
    selected = nuclei.loc[nuclei["object_label_id"].astype(int).eq(NUCLEUS_LABEL_ID)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one row for nucleus label {NUCLEUS_LABEL_ID}")
    row = selected.iloc[0]
    center_x = float(row["center_x_fullres_px"])
    center_y = float(row["center_y_fullres_px"])
    pixel_size_um = float(row["pixel_size_um"])
    if str(row["source_id"]) != SOURCE_ID or str(row["section_id"]) != SECTION_ID:
        raise RuntimeError("The selected nucleus did not resolve to the locked Cont1 source")
    if not np.isclose(center_x, EXPECTED_CENTER_X, atol=1e-9) or not np.isclose(center_y, EXPECTED_CENTER_Y, atol=1e-9):
        raise RuntimeError("The selected nucleus coordinates differ from the historical Figure 3a record")
    if not np.isclose(pixel_size_um, EXPECTED_PIXEL_SIZE_UM, atol=1e-12):
        raise RuntimeError("The selected nucleus pixel size differs from the historical Figure 3a record")

    source = DapiSourceSpec(
        source_id=SOURCE_ID,
        source_group="if",
        section_id=SECTION_ID,
        sample_family=SECTION_ID,
        condition=CONDITION,
        image_kind="vsi",
        image_path=vsi_path,
        sidecar_dir=sidecar_dir,
        nuclei_mask_path=mask_path,
        label_path=None,
        metadata_path=tissue_metadata,
        pixel_size_um=pixel_size_um,
        holdout=False,
    )
    backend = DapiCropManager([source], config=IfXeniumMultiscaleVitConfig()).backend(SOURCE_ID)
    label_mask = np.load(mask_path, mmap_mode="r")
    arrays: dict[str, np.ndarray] = {}
    crop_records: dict[str, dict[str, object]] = {}
    for name, crop_px in CROP_SIZES.items():
        image = backend.load_native_crop(
            center_x_fullres_px=center_x,
            center_y_fullres_px=center_y,
            crop_fullres_px=crop_px,
        )
        bounds = centered_bounds(center_x, center_y, crop_px)
        arrays[name] = np.rint(np.clip(np.asarray(image), 0.0, 1.0) * 255.0).astype(np.uint8)
        arrays[f"{name}_target_mask"] = mask_crop(label_mask, bounds, NUCLEUS_LABEL_ID)
        crop_records[name] = {
            "crop_size_fullres_px": crop_px,
            "field_width_um": crop_px * pixel_size_um,
            "bounds_fullres_px": bounds,
        }

    center_id = f"{SOURCE_ID}:nucleus:{NUCLEUS_LABEL_ID}"
    prepared_inputs, prepared_records, pretraining_rows = load_pretraining_inputs(
        pretraining_manifest,
        args.pretraining_shard_dir.expanduser().resolve(),
        center_id,
        args.rows_per_shard,
    )
    arrays.update(prepared_inputs)

    out = args.output_dir.expanduser().resolve()
    output_npz = out / "panel_A_exact_historical_region.npz"
    output_json = out / "panel_A_exact_historical_region.json"
    provenance = out / "panel_A_exact_historical_region_provenance.json"
    overwritten = protect_outputs([output_npz, output_json, provenance], args.overwrite)
    np.savez_compressed(output_npz, **arrays)
    payload = {
        "lineage": "exact_region_imported_into_manual_figure3a",
        "source_group": "if",
        "source_id": SOURCE_ID,
        "section_id": SECTION_ID,
        "condition": CONDITION,
        "center_id": center_id,
        "nucleus_label_id": NUCLEUS_LABEL_ID,
        "center_x_fullres_px": center_x,
        "center_y_fullres_px": center_y,
        "nucleus_area_px": int(row["nucleus_area_px"]),
        "pixel_size_um": pixel_size_um,
        "crop_records": crop_records,
        "pretraining_rows": pretraining_rows,
        "used_in_multimodal_pretraining": True,
        "used_in_xenium_supervised_coordinate_training": False,
        "selection_note": "Recovered from the original out-of-repository scripts and metadata; not newly selected.",
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    source_module = pipeline_src / "if_processing_pipeline/spatial_if_multiscale_dapi.py"
    input_records = [
        file_record(nuclei_path, "if_nuclei_table"),
        file_record(pretraining_manifest, "multimodal_pretraining_manifest"),
        file_record(vsi_path, "cont1_vsi_header"),
        tree_record(sidecar_dir, "cont1_vsi_sidecar_tree"),
        file_record(mask_path, "cont1_nuclei_segmentation"),
        file_record(tissue_metadata, "cont1_tissue_metadata"),
        file_record(source_module, "microscopy_crop_implementation"),
        *prepared_records,
    ]
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=input_records,
        outputs=[output_npz, output_json],
        extra={"panel": "Figure 3a exact microscopy source region", **payload},
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
