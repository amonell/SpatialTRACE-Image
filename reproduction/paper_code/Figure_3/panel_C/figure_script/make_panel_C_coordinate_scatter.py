#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, protect_outputs, require_file, save_pair, write_provenance, repaired_reference  # noqa: E402

EXPECTED_CHECKPOINT = "d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137"
EXPECTED_SPLIT = "f707d0598722c0381c552ad9f4803174de8652e56d8adf99d64cf8bed61994e2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--metrics-record", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prediction_path = repaired_reference(args.predictions, args.reference_manifest, "predictions_current_references.parquet")
    metrics_path = repaired_reference(args.metrics_record, args.reference_manifest, "metrics.json")
    checkpoint = require_file(args.checkpoint, EXPECTED_CHECKPOINT)
    split_path = require_file(args.split_manifest, EXPECTED_SPLIT)
    frame = pd.read_parquet(prediction_path)
    split = json.loads(split_path.read_text())
    saved = json.loads(metrics_path.read_text())
    if len(frame) != 50_000 or frame.row_id.nunique() != 50_000 or len(split["cell_ids"]) != 50_000:
        raise RuntimeError("Panel C requires all 50,000 locked sample_008 cells")
    if frame["row_id"].astype(str).tolist() != list(map(str, split["cell_ids"])):
        raise RuntimeError("Panel C prediction identities/order differ from the sample_008 lock")
    specs = (("crypt_villus", "target_axis", "predicted_axis_coordinate", "Crypt-villus coordinate", "viridis"), ("epithelial_distance", "epithelial_distance_clipped_1p0", "predicted_epithelial_distance_clipped_1p0", "Epithelial-distance coordinate", "coolwarm"))
    results = saved
    for name, reference, prediction, _, _ in specs:
        if saved[name]['n'] != len(frame) or not np.isfinite(frame[[reference,prediction]].to_numpy()).all():
            raise RuntimeError(f'Invalid frozen metric count or coordinates: {name}')
    out = args.output_dir.expanduser().resolve()
    stem, table, provenance = out / "panel_C_coordinate_scatter", out / "panel_C_metrics.tsv", out / "panel_C_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), table, provenance], args.overwrite)
    pd.DataFrame([{"coordinate": name, **values} for name, values in results.items()]).to_csv(table, sep="\t", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(7.3, 3.35))
    figure.subplots_adjust(left=.08, right=.96, bottom=.16, top=.87, wspace=.42)
    for axis, (name, reference, prediction, title, cmap) in zip(axes, specs, strict=True):
        artist = axis.hexbin(frame[reference], frame[prediction], gridsize=55, extent=(0, 1, 0, 1), mincnt=1, cmap=cmap, norm=LogNorm(), linewidths=0)
        axis.plot([0, 1], [0, 1], "--", color="#222222", linewidth=1)
        value = results[name]
        axis.text(.03, .97, f"Pearson r = {value['pearson_r']:.3f}\nMAE = {value['mae']:.3f}\nR² = {value['r2']:.3f}", transform=axis.transAxes, ha="left", va="top", fontsize=7.5, bbox={"boxstyle": "round,pad=.28", "facecolor": "white", "edgecolor": "none", "alpha": .92})
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Graph-derived reference", ylabel="Image-model prediction")
        axis.set_aspect("equal"); axis.set_title(title, fontweight="bold")
        colorbar = figure.colorbar(artist, ax=axis, fraction=.045, pad=.02); colorbar.set_label("Cell count")
    rendered = save_pair(figure, stem, args.dpi); plt.close(figure)
    write_provenance(provenance, script=Path(__file__), inputs=[file_record(args.reference_manifest, "current_graph_reference_release"), file_record(prediction_path, "sample008_production_v2_predictions"), file_record(metrics_path, "sample008_metrics"), file_record(checkpoint, "released_production_v2_checkpoint"), file_record(split_path, "locked_sample008_source")], outputs=[*rendered, table], extra={"panel": "Figure 3c", "source_id": "sample_008", "cells": 50000, "reference_lineage": "Dense coordinates predicted by the frozen graph model from transcriptomic and spatial-neighborhood features; same graph arrays as production-v2 SFT", "metrics": results}, overwritten=overwritten)


if __name__ == "__main__":
    main()
