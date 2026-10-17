#!/usr/bin/env python3
"""Build compact, immutable plotting inputs from the frozen graph-model outputs.

This script performs deterministic subsetting only. It does not fit a model,
select a checkpoint, construct an evaluation split, or calculate an ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from paper_paths import Path, lock_record, rendering_script
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


BATCHES = ("day6_SI_r2", "day8_SI_Ctrl", "day30_SI_r2", "day90_SI_r2")
ZOOM_BATCH = "day90_SI_r2"
ZOOM_VILLI = ("day90_SI_r2__villus_006", "day90_SI_r2__villus_004")
GRAPH_BATCH = "day6_SI_r2"
GRAPH_CENTER = (5315.0, 2350.0)
GRAPH_ROI_WIDTH = 500.0
GRAPH_WIDTH = 62.0
N_NEIGHBORS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str | None) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected and sha256(path) != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")
    return path


def categorical(group: h5py.Group, indices: np.ndarray) -> np.ndarray:
    categories = np.asarray(group["categories"])
    categories = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in categories])
    return categories[np.asarray(group["codes"])[indices]]


def sampled(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return np.sort(np.random.default_rng(seed).choice(indices, maximum, replace=False))


def crop_limits(coords: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = np.nanmin(coords, axis=0)
    x1, y1 = np.nanmax(coords, axis=0)
    width = max(float(x1 - x0), 900.0) * 1.18
    height = max(float(y1 - y0), 700.0) * 1.18
    cx, cy = float((x0 + x1) / 2), float((y0 + y1) / 2)
    return cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2


def protect(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite:\n" + "\n".join(existing))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--crypt-predictions", type=Path, required=True)
    parser.add_argument("--epithelial-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points-per-section", type=int, default=250000)
    parser.add_argument("--expected-labels-sha256")
    parser.add_argument("--expected-crypt-sha256")
    parser.add_argument("--expected-epithelial-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    h5ad = require_hash(args.h5ad, None)
    labels_path = require_hash(args.labels, args.expected_labels_sha256)
    crypt_path = require_hash(args.crypt_predictions, args.expected_crypt_sha256)
    epithelial_path = require_hash(args.epithelial_predictions, args.expected_epithelial_sha256)
    output_dir = args.output_dir.expanduser().resolve()
    paths = {
        "sections": output_dir / "section_cells.parquet",
        "labels": output_dir / "sparse_labels.parquet",
        "zoom": output_dir / "day90_zoom_cells.parquet",
        "graph_nodes": output_dir / "graph_example_nodes.parquet",
        "graph_edges": output_dir / "graph_example_edges.tsv",
        "focus_features": output_dir / "graph_focus_scvi_features.tsv",
        "focus_expression": output_dir / "graph_focus_gene_expression.tsv",
        "manifest": output_dir / "graph_component_inputs_manifest.json",
    }
    protect(list(paths.values()), args.overwrite)

    labels = pd.read_csv(labels_path)
    labels["global_index"] = labels["global_index"].astype(np.int64)
    zoom_labels = labels[
        labels["batch"].astype(str).eq(ZOOM_BATCH)
        & labels["training_villus_id"].astype(str).isin(ZOOM_VILLI)
    ]
    if zoom_labels.empty:
        raise RuntimeError("The prespecified Day 90 zoom villi are absent from the labels.")

    crypt = np.load(crypt_path, mmap_mode="r")
    epithelial = np.load(epithelial_path, mmap_mode="r")
    section_frames: list[pd.DataFrame] = []
    with h5py.File(h5ad, "r") as handle:
        batch_group = handle["obs"]["batch"]
        batch_categories = np.asarray([
            v.decode() if isinstance(v, bytes) else str(v)
            for v in np.asarray(batch_group["categories"])
        ])
        batch_codes = np.asarray(batch_group["codes"])
        class_group = handle["obs"]["Class"]
        coords_ds = handle["obsm"]["X_spatial"]
        embed_ds = handle["obsm"]["X_mde"]
        scvi_ds = handle["obsm"]["X_scVI"]
        if len(crypt) != coords_ds.shape[0] or len(epithelial) != coords_ds.shape[0]:
            raise RuntimeError("Frozen predictions do not align to the H5AD row count.")

        batch_indices: dict[str, np.ndarray] = {}
        for batch_offset, batch in enumerate(BATCHES):
            code = int(np.flatnonzero(batch_categories == batch)[0])
            all_idx = np.flatnonzero(batch_codes == code)
            batch_indices[batch] = all_idx
            idx = sampled(all_idx, args.max_points_per_section, args.seed + batch_offset)
            coords = np.asarray(coords_ds[idx])
            emb = np.asarray(embed_ds[idx])
            section_frames.append(pd.DataFrame({
                "global_index": idx,
                "batch": batch,
                "x": coords[:, 0],
                "y": coords[:, 1],
                "embedding_x": emb[:, 0],
                "embedding_y": emb[:, 1],
                "cell_class": categorical(class_group, idx),
                "crypt_villus_prediction": np.asarray(crypt[idx], dtype=np.float32),
                "epithelial_distance_prediction": np.asarray(epithelial[idx], dtype=np.float32),
            }))

        label_idx = np.sort(labels["global_index"].unique())
        label_coords = np.asarray(coords_ds[label_idx])
        coordinate_lookup = pd.DataFrame({"global_index": label_idx, "x": label_coords[:, 0], "y": label_coords[:, 1]})
        sparse_labels = labels.merge(coordinate_lookup, on="global_index", how="left", validate="many_to_one")

        zoom_idx = np.sort(zoom_labels["global_index"].unique())
        limits = crop_limits(np.asarray(coords_ds[zoom_idx]))
        day90_idx = batch_indices[ZOOM_BATCH]
        day90_coords = np.asarray(coords_ds[day90_idx])
        in_zoom = (
            (day90_coords[:, 0] >= limits[0]) & (day90_coords[:, 0] <= limits[1])
            & (day90_coords[:, 1] >= limits[2]) & (day90_coords[:, 1] <= limits[3])
        )
        zoom_all_idx = day90_idx[in_zoom]
        zoom_all_coords = day90_coords[in_zoom]
        zoom_cells = pd.DataFrame({
            "global_index": zoom_all_idx,
            "x": zoom_all_coords[:, 0],
            "y": zoom_all_coords[:, 1],
            "crypt_villus_prediction": np.asarray(crypt[zoom_all_idx], dtype=np.float32),
            "epithelial_distance_prediction": np.asarray(epithelial[zoom_all_idx], dtype=np.float32),
        })

        graph_idx_all = batch_indices[GRAPH_BATCH]
        graph_xy_all = np.asarray(coords_ds[graph_idx_all])
        roi_half = GRAPH_ROI_WIDTH / 2
        in_roi = (
            (np.abs(graph_xy_all[:, 0] - GRAPH_CENTER[0]) <= roi_half)
            & (np.abs(graph_xy_all[:, 1] - GRAPH_CENTER[1]) <= roi_half)
        )
        roi_idx, roi_xy = graph_idx_all[in_roi], graph_xy_all[in_roi]
        density_tree = cKDTree(roi_xy)
        graph_half = GRAPH_WIDTH / 2
        counts = np.asarray([len(v) for v in density_tree.query_ball_point(roi_xy, r=graph_half, p=np.inf)])
        focus_roi = int(np.argmax(counts))
        focus_xy = roi_xy[focus_roi]
        in_graph = (np.abs(roi_xy[:, 0] - focus_xy[0]) <= graph_half) & (np.abs(roi_xy[:, 1] - focus_xy[1]) <= graph_half)
        graph_idx, graph_xy = roi_idx[in_graph], roi_xy[in_graph]
        focus = int(np.argmin(np.sum((graph_xy - focus_xy) ** 2, axis=1)))
        tree = cKDTree(graph_xy)
        _, nn = tree.query(graph_xy, k=N_NEIGHBORS + 1)
        nn = nn[:, 1:]
        graph_nodes = pd.DataFrame({
            "node": np.arange(len(graph_idx)),
            "global_index": graph_idx,
            "x": graph_xy[:, 0],
            "y": graph_xy[:, 1],
            "cell_class": categorical(class_group, graph_idx),
            "is_focus": np.arange(len(graph_idx)) == focus,
            "is_focus_neighbor": np.isin(np.arange(len(graph_idx)), nn[focus]),
        })
        graph_edges = pd.DataFrame({
            "source": np.repeat(np.arange(len(graph_idx)), N_NEIGHBORS),
            "target": nn.reshape(-1),
        })
        focus_features = np.asarray(scvi_ds[[int(graph_idx[focus])]])[0]
        focus_expression = np.asarray(handle["X"][[int(graph_idx[focus])]])[0]
        gene_key = "_index" if "_index" in handle["var"] else "gene"
        gene_names = np.asarray(handle["var"][gene_key])
        gene_names = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in gene_names])
        top_gene_idx = np.argsort(focus_expression)[-12:][::-1]

    sections = pd.concat(section_frames, ignore_index=True)
    sections.to_parquet(paths["sections"], index=False)
    sparse_labels.to_parquet(paths["labels"], index=False)
    zoom_cells.to_parquet(paths["zoom"], index=False)
    graph_nodes.to_parquet(paths["graph_nodes"], index=False)
    graph_edges.to_csv(paths["graph_edges"], sep="\t", index=False)
    pd.DataFrame({"feature": [f"scVI_{i + 1}" for i in range(len(focus_features))], "value": focus_features}).to_csv(
        paths["focus_features"], sep="\t", index=False
    )
    pd.DataFrame({"gene": gene_names[top_gene_idx], "expression": focus_expression[top_gene_idx]}).to_csv(
        paths["focus_expression"], sep="\t", index=False
    )

    input_records = []
    for role, path in (("h5ad", h5ad), ("labels", labels_path), ("crypt_predictions", crypt_path), ("epithelial_predictions", epithelial_path)):
        stat = path.stat()
        input_records.append({
            "role": role,
            "path": str(path),
            "sha256": sha256(path) if role != "h5ad" else None,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "hash_note": None if role != "h5ad" else "Large immutable H5AD identity is locked by absolute path, byte size, and nanosecond mtime.",
        })
    outputs = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for key, path in paths.items() if key != "manifest"]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "operation": "deterministic panel-specific subsetting of frozen preprocessing outputs",
        "seed": args.seed,
        "batches": list(BATCHES),
        "max_points_per_section": args.max_points_per_section,
        "zoom_batch": ZOOM_BATCH,
        "zoom_villi": list(ZOOM_VILLI),
        "zoom_limits_um": dict(zip(("x_min", "x_max", "y_min", "y_max"), limits, strict=True)),
        "graph": {"batch": GRAPH_BATCH, "roi_center_um": GRAPH_CENTER, "roi_width_um": GRAPH_ROI_WIDTH, "graph_width_um": GRAPH_WIDTH, "n_neighbors": N_NEIGHBORS, "focus_global_index": int(graph_idx[focus])},
        "inputs": input_records,
        "outputs": outputs,
        "counts": {"section_cells": len(sections), "sparse_label_rows": len(sparse_labels), "zoom_cells": len(zoom_cells), "graph_nodes": len(graph_nodes), "graph_edges": len(graph_edges)},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    print(paths["manifest"])


if __name__ == "__main__":
    main()
