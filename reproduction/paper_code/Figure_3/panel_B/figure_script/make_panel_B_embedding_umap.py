#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, protect_outputs, require_file, save_pair, write_provenance, repaired_reference  # noqa: E402

EXPECTED_SELECTION = "3f884b2fe134b3f0cb1b9f7fe58c25b72e2957b874bb120754d0eba7dfec0c61"
EXPECTED_CHECKPOINT = "d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137"
EXPECTED_SPLIT = "f707d0598722c0381c552ad9f4803174de8652e56d8adf99d64cf8bed61994e2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umap", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    umap_path = repaired_reference(args.umap, args.reference_manifest, 'figure3f_umap.csv')
    metadata_path = repaired_reference(args.metadata, args.reference_manifest, "embedding_metadata_current_references.csv")
    selection_path = require_file(args.selection, EXPECTED_SELECTION)
    checkpoint = require_file(args.checkpoint, EXPECTED_CHECKPOINT)
    split = require_file(args.split_manifest, EXPECTED_SPLIT)
    coordinates, metadata = pd.read_csv(umap_path), pd.read_csv(metadata_path)
    selection = json.loads(selection_path.read_text())
    if len(coordinates) != 30_000 or len(metadata) != 30_000 or selection["selected_cells"] != 30_000:
        raise RuntimeError("Panel B requires the frozen 30,000-cell subset")
    if set(metadata["source_id"].astype(str)) != {"sample_008"}:
        raise RuntimeError("Panel B requires only the locked sample_008 source")
    if metadata["row_id"].astype(str).nunique() != 30_000:
        raise RuntimeError("Panel B embedding metadata contains duplicate row identifiers")
    table = pd.concat([metadata.reset_index(drop=True), coordinates.reset_index(drop=True)], axis=1)
    out = args.output_dir.expanduser().resolve()
    stem, provenance = out / "panel_B_embedding_umap", out / "panel_B_embedding_umap_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), provenance], args.overwrite)
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 3.05))
    specifications = (("target_axis", "Crypt-villus coordinate", "viridis", 0, 1), ("epithelial_distance_clipped_1p0", "Epithelial-distance coordinate", "coolwarm", 0, .7))
    for axis, (column, title, cmap, vmin, vmax) in zip(axes, specifications, strict=True):
        artist = axis.scatter(table.umap_1, table.umap_2, c=table[column], s=2, linewidths=0, alpha=.78, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("UMAP 1"); axis.set_ylabel("UMAP 2")
        axis.set_xticks([]); axis.set_yticks([]); axis.set_aspect("equal", adjustable="datalim")
        figure.colorbar(artist, ax=axis, fraction=.046, pad=.02)
    figure.tight_layout()
    rendered = save_pair(figure, stem, args.dpi); plt.close(figure)
    write_provenance(provenance, script=Path(__file__), inputs=[file_record(args.reference_manifest, "current_graph_reference_release"), file_record(umap_path, "frozen_umap_coordinates"), file_record(metadata_path, "frozen_embedding_metadata"), file_record(selection_path, "deterministic_subset_record"), file_record(checkpoint, "released_production_v2_checkpoint"), file_record(split, "locked_sample008_source")], outputs=rendered, extra={"panel": "Figure 3b", "source_id": "sample_008", "cells": 30000, "selection_seed": selection["selection_seed"], "umap_refitted": False}, overwritten=overwritten)


if __name__ == "__main__":
    main()
