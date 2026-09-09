from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def _decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])
    return np.asarray(values)


def _read_obs_column(obs: h5py.Group, column: str) -> np.ndarray:
    if column not in obs:
        raise KeyError(f"Missing obs column `{column}`.")
    node = obs[column]
    if isinstance(node, h5py.Dataset):
        return _decode_array(node[()])
    if isinstance(node, h5py.Group) and {"codes", "categories"}.issubset(node.keys()):
        codes = np.asarray(node["codes"][()])
        categories = _decode_array(node["categories"][()])
        out = np.empty(len(codes), dtype=object)
        valid = codes >= 0
        out[valid] = categories[codes[valid]]
        out[~valid] = ""
        return out
    raise TypeError(f"Unsupported h5ad obs storage for `{column}`.")


def _read_obsm_matrix(obsm: h5py.Group, key: str) -> np.ndarray:
    if key not in obsm:
        raise KeyError(f"Missing obsm key `{key}`.")
    node = obsm[key]
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[()])
    raise TypeError(f"Unsupported h5ad obsm storage for `{key}`.")


def _match_image(batch_value: str, image_paths: list[Path]) -> Path | None:
    token = str(batch_value)
    exact = [path for path in image_paths if token in str(path)]
    if exact:
        return sorted(exact, key=lambda path: (len(str(path)), str(path)))[0]
    compact_token = token.replace("_", "").replace("-", "").lower()
    compact_matches = [
        path
        for path in image_paths
        if compact_token in str(path).replace("_", "").replace("-", "").lower()
    ]
    if compact_matches:
        return sorted(compact_matches, key=lambda path: (len(str(path)), str(path)))[0]
    return None


def build_xenium_manifests(
    *,
    h5ad_path: Path,
    output_dir: Path,
    image_root: Path,
    image_glob: str = "**/xenium_output/morphology_mip.ome.tif",
    batch_column: str = "batch",
    spatial_key: str = "X_spatial",
    spatial_units: str = "pixels",
    axis_column: str = "crypt_villi_axis",
    epithelial_column: str = "epithelial_distance_clipped",
    epithelial_clip_max: float | None = None,
    pixel_size_um: float | None = None,
    channel_index: int = 0,
    max_cells_per_source: int | None = None,
    max_train_cells_per_source: int | None = None,
    max_validation_cells_per_source: int | None = None,
    validation_fraction: float = 0.20,
    validation_mode: str = "section",
    seed: int = 0,
) -> dict[str, object]:
    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    image_root = Path(image_root)
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)
    if not image_root.exists():
        raise FileNotFoundError(image_root)
    image_paths = sorted(image_root.glob(str(image_glob)))
    if not image_paths:
        raise FileNotFoundError(f"No image files matched `{image_glob}` below {image_root}.")

    with h5py.File(h5ad_path, "r") as handle:
        obs = handle["obs"]
        batches = _read_obs_column(obs, batch_column).astype(str)
        axis = pd.to_numeric(pd.Series(_read_obs_column(obs, axis_column)), errors="coerce").to_numpy()
        epithelial = pd.to_numeric(pd.Series(_read_obs_column(obs, epithelial_column)), errors="coerce").to_numpy()
        if epithelial_clip_max is not None:
            epithelial = np.clip(epithelial, 0.0, float(epithelial_clip_max))
        coords = _read_obsm_matrix(handle["obsm"], spatial_key)

    if coords.shape[1] < 2:
        raise ValueError(f"`{spatial_key}` must contain at least two coordinate columns.")
    spatial_units = str(spatial_units).strip().lower()
    if spatial_units not in {"pixels", "microns", "um"}:
        raise ValueError("spatial_units must be `pixels` or `microns`.")
    coordinate_pixel_size_um = None
    coords_fullres_px = np.asarray(coords[:, :2], dtype=np.float64).copy()
    if spatial_units in {"microns", "um"}:
        if pixel_size_um is None:
            raise ValueError("pixel_size_um is required when spatial_units is `microns`.")
        coordinate_pixel_size_um = float(pixel_size_um)
        if coordinate_pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive when spatial_units is `microns`.")
        coords_fullres_px = coords_fullres_px / coordinate_pixel_size_um
    finite = (
        np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, 1])
        & np.isfinite(axis)
        & np.isfinite(epithelial)
        & (pd.Series(batches).str.len().to_numpy() > 0)
    )
    rng = np.random.default_rng(int(seed))
    unique_batches = sorted(pd.Series(batches[finite]).drop_duplicates().astype(str).tolist())
    batch_to_source = {batch: f"sample_{idx:03d}" for idx, batch in enumerate(unique_batches)}
    source_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    validation_batches: set[str] = set()
    validation_indices: set[int] = set()
    mode = str(validation_mode).strip().lower()
    if mode not in {"section", "cell"}:
        raise ValueError("validation_mode must be `section` or `cell`.")
    if float(validation_fraction) > 0 and unique_batches:
        if mode == "section":
            val_count = max(1, int(round(len(unique_batches) * float(validation_fraction))))
            val_count = min(val_count, max(len(unique_batches) - 1, 1))
            validation_batches = set(rng.choice(np.asarray(unique_batches), size=val_count, replace=False).tolist())
        else:
            all_indices = np.arange(len(batches))
            eligible = all_indices[finite & np.isin(batches, unique_batches)]
            if len(eligible) > 1:
                val_count = max(1, int(round(len(eligible) * float(validation_fraction))))
                val_count = min(val_count, len(eligible) - 1)
                validation_indices = set(rng.choice(eligible, size=val_count, replace=False).tolist())

    image_link_dir = output_dir.parent / "image_links"
    image_link_dir.mkdir(parents=True, exist_ok=True)

    for batch in unique_batches:
        image_path = _match_image(batch, image_paths)
        if image_path is None:
            missing_images.append(batch)
            continue
        source_id = batch_to_source[batch]
        image_link = image_link_dir / f"{source_id}.ome.tif"
        if image_link.exists() or image_link.is_symlink():
            image_link.unlink()
        os.symlink(image_path, image_link)
        source_rows.append(
            {
                "source_id": source_id,
                "image_path": str(Path("..") / "image_links" / image_link.name),
                "condition": "",
                "section_id": source_id,
                "pixel_size_um": "" if pixel_size_um is None else float(pixel_size_um),
                "channel_index": int(channel_index),
            }
        )
    available_sources = {row["source_id"] for row in source_rows}
    cell_rows: list[dict[str, Any]] = []
    all_cell_rows: list[dict[str, Any]] = []
    for batch in unique_batches:
        source_id = batch_to_source[batch]
        if source_id not in available_sources:
            continue
        all_indices = np.flatnonzero(finite & (batches == batch))
        if mode == "section":
            split_name = "validation" if batch in validation_batches else "train"
            splits = np.repeat(split_name, len(all_indices))
        else:
            splits = np.asarray(["validation" if int(idx) in validation_indices else "train" for idx in all_indices])
        selected_indices = all_indices
        selected_splits = splits
        legacy_cap = None if max_cells_per_source is None else int(max_cells_per_source)
        train_cap = legacy_cap if max_train_cells_per_source is None else int(max_train_cells_per_source)
        validation_cap = (
            legacy_cap if max_validation_cells_per_source is None else int(max_validation_cells_per_source)
        )
        keep: list[int] = []
        for split_name, cap in (("train", train_cap), ("validation", validation_cap)):
            split_positions = np.flatnonzero(splits == split_name)
            if cap is not None and cap > 0 and len(split_positions) > cap:
                split_positions = np.sort(rng.choice(split_positions, size=cap, replace=False))
            keep.extend(split_positions.tolist())
        if keep:
            keep = sorted(keep)
            selected_indices = all_indices[keep]
            selected_splits = splits[keep]
        row_ids = {int(idx): f"{source_id}_cell_{local_index:07d}" for local_index, idx in enumerate(all_indices)}
        for idx, split_name in zip(all_indices, splits, strict=True):
            all_cell_rows.append(
                {
                    "source_id": source_id,
                    "centroid_x_fullres_px": float(coords_fullres_px[idx, 0]),
                    "centroid_y_fullres_px": float(coords_fullres_px[idx, 1]),
                    "target_axis": float(axis[idx]),
                    "epithelial_distance_clipped_1p0": float(epithelial[idx]),
                    "split": str(split_name),
                    "row_id": row_ids[int(idx)],
                }
            )
        for idx, split_name in zip(selected_indices, selected_splits, strict=True):
            cell_rows.append(
                {
                    "source_id": source_id,
                    "centroid_x_fullres_px": float(coords_fullres_px[idx, 0]),
                    "centroid_y_fullres_px": float(coords_fullres_px[idx, 1]),
                    "target_axis": float(axis[idx]),
                    "epithelial_distance_clipped_1p0": float(epithelial[idx]),
                    "split": str(split_name),
                    "row_id": row_ids[int(idx)],
                }
            )
    if not source_rows:
        raise ValueError("No source images could be matched to h5ad batches.")
    if not cell_rows:
        raise ValueError("No finite supervised cell rows could be exported.")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = output_dir / "source_manifest.csv"
    supervised_manifest = output_dir / "supervised_cells.csv"
    all_supervised_manifest = output_dir / "all_supervised_cells.csv"
    cells_manifest = output_dir / "cells.csv"
    pd.DataFrame(source_rows).to_csv(source_manifest, index=False)
    supervised = pd.DataFrame(cell_rows)
    supervised.to_csv(supervised_manifest, index=False)
    all_supervised = pd.DataFrame(all_cell_rows)
    all_supervised.to_csv(all_supervised_manifest, index=False)
    supervised.drop(columns=["target_axis", "epithelial_distance_clipped_1p0"]).to_csv(cells_manifest, index=False)
    summary = {
        "h5ad_path": str(h5ad_path),
        "image_root": str(image_root),
        "image_glob": str(image_glob),
        "image_link_dir": str(image_link_dir),
        "source_manifest": str(source_manifest),
        "supervised_manifest": str(supervised_manifest),
        "all_supervised_manifest": str(all_supervised_manifest),
        "cells_manifest": str(cells_manifest),
        "source_count": int(len(source_rows)),
        "cell_count": int(len(supervised)),
        "all_cell_count": int(len(all_supervised)),
        "validation_mode": mode,
        "validation_fraction": float(validation_fraction),
        "validation_cell_count": int((supervised["split"] == "validation").sum()),
        "all_validation_cell_count": int((all_supervised["split"] == "validation").sum()),
        "validation_source_ids": sorted(
            {
                str(value)
                for value in all_supervised.loc[
                    all_supervised["split"].astype(str).eq("validation"), "source_id"
                ].drop_duplicates()
            }
        ),
        "max_train_cells_per_source": None if max_train_cells_per_source is None else int(max_train_cells_per_source),
        "max_validation_cells_per_source": (
            None if max_validation_cells_per_source is None else int(max_validation_cells_per_source)
        ),
        "epithelial_column": str(epithelial_column),
        "epithelial_clip_max": None if epithelial_clip_max is None else float(epithelial_clip_max),
        "spatial_key": str(spatial_key),
        "spatial_units": spatial_units,
        "coordinate_pixel_size_um": coordinate_pixel_size_um,
        "row_id_policy": "anonymized per-source sequential IDs",
        "missing_image_count": int(len(missing_images)),
    }
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(summary_path)
    return summary
