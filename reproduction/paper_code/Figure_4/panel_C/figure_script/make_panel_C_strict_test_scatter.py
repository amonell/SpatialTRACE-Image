#!/usr/bin/env python3
"""Render the strict IF test scatterplots and their frozen metric annotations."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_4.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402
from Figure_4.release import load_release, release_file

EXPECTED_PREDICTIONS = "9c4532ce30c1ecd2c5a717688ea3e13d37e32a31388cb93a4a4e0dc5395518db"
EXPECTED_SPLIT = "7d9c9120c5ef55a2c30a2fe2b3c0e0fe3b16a111ca61ff5339029ae6e4a2ac48"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    release = load_release(args.release_manifest)
    predictions_path = release_file(args.predictions, release)
    frozen_metric_path = release_file(args.metrics, release)
    frozen_metrics = json.loads(frozen_metric_path.read_text())
    split = require_file(args.split_manifest, EXPECTED_SPLIT)
    checkpoint = release_file(args.checkpoint, release)
    rows = pd.read_parquet(predictions_path)
    if len(rows) != 4843 or rows.row_id.duplicated().any() or not rows.eligible_final_test.astype(bool).all():
        raise RuntimeError("Panel C requires 4,843 unique eligible strict-test cells")
    specs = (
        ("target_axis", "predicted_axis_coordinate", "Crypt-villus coordinate", "viridis"),
        ("epithelial_distance_clipped_1p0", "predicted_epithelial_distance_clipped_1p0", "Epithelial-distance coordinate", "coolwarm"),
    )
    results = {}
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_C_strict_IF_test_scatter"
    metric_path = out / "panel_C_metrics.json"
    provenance = out / "panel_C_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), metric_path, provenance], args.overwrite)
    figure, axes = plt.subplots(1, 2, figsize=(7.3, 3.2), dpi=args.dpi)
    for axis, (reference_key, prediction_key, title, cmap) in zip(axes, specs, strict=True):
        reference = rows[reference_key].to_numpy(float); prediction = rows[prediction_key].to_numpy(float)
        if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite values in frozen {title} predictions")
        metric_key = "Crypt-villus" if reference_key == "target_axis" else "Epithelial distance"
        result = frozen_metrics[metric_key]
        if int(result["n"]) != len(rows) or not np.isfinite([result[key] for key in ("pearson_r", "mae", "r2")]).all():
            raise RuntimeError("Frozen metric annotations do not match the evaluated panel population")
        results[title] = result
        density = axis.hexbin(reference, prediction, gridsize=45, mincnt=1, cmap=cmap, bins="log", rasterized=True)
        colorbar = figure.colorbar(density, ax=axis, fraction=0.046, pad=0.035)
        colorbar.set_label("Cell count", fontsize=8.5)
        colorbar.ax.tick_params(labelsize=8)
        axis.plot([0, 1], [0, 1], color="#555555", ls="--", lw=0.8)
        axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.set_aspect("equal")
        axis.set_xlabel("Sparse IF reference"); axis.set_ylabel("Image-model prediction")
        axis.set_title(title, fontsize=9.5)
        axis.text(0.04, 0.96, f"Pearson r = {result['pearson_r']:.3f}\nMAE = {result['mae']:.3f}\nR² = {result['r2']:.3f}", transform=axis.transAxes, ha="left", va="top", fontsize=8.5, linespacing=1.25, zorder=5, bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.96, "pad": 3})
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Train- and validation-crop-disjoint IF test set", fontsize=9.5, y=0.99)
    figure.subplots_adjust(left=0.075, right=0.955, bottom=0.17, top=0.83, wspace=0.42)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    metric_path.write_text(json.dumps(results, indent=2) + "\n")
    write_provenance(
        provenance, script=Path(__file__),
        inputs=[file_record(args.release_manifest, "repaired_IF_release"), file_record(frozen_metric_path, "frozen_strict_test_metrics"), file_record(predictions_path, "frozen_strict_test_predictions"), file_record(split, "locked_strict_test_split"), file_record(checkpoint, "released_IF_checkpoint")],
        outputs=[*rendered, metric_path],
        extra={
            "panel": "Figure 4c", "metrics": results, "inference_run": False, "strict_test_cells": 4843,
            "metrics_recomputed": False, "metrics_export": "Frozen metric records re-exported unchanged",
            "visible_metric_annotations": ["pearson_r", "mae", "r2"],
            "x_axis": "Sparse IF reference", "y_axis": "Image-model prediction",
            "plot_specification": {
                title: {"reference_column": reference_key, "prediction_column": prediction_key, "colormap": cmap}
                for reference_key, prediction_key, title, cmap in specs
            },
            "color_mapping": "Logarithmic cell count per hexagonal bin; not coordinate value",
            "hexbin_gridsize": 45,
            "all_frozen_test_cells_plotted": True,
            "reference_version": release["test_reference_version"],
            "evaluation_scope": release["scope"],
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
