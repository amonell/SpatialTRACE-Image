#!/usr/bin/env python3
"""Render Figure 2e from the frozen section-disjoint annotation-scaling outputs."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FIGURE_ROOT))
from common import file_record, protect_outputs, require_file, sha256, write_provenance  # noqa: E402

EXPECTED_COUNTS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
EXPECTED_AXES = ["crypt_villus", "epithelial_distance"]
EXPECTED_FOLDS = 8
EXPECTED_METRICS_SHA256 = "462e31be3c8c6cfe0ec7aff09f282bfe5bb685f16f151bd3484f3478e75dcf94"
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = "e9771443cdb07dc9db6437c8d8c58f5da29ed44e870e5ac440d3c8bc511aeeb0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metrics_path = require_file(args.metrics, EXPECTED_METRICS_SHA256)
    split_path = require_file(args.split_manifest, args.expected_split_sha256)
    checkpoint_path = require_file(args.checkpoint_manifest, EXPECTED_CHECKPOINT_MANIFEST_SHA256)
    frame = pd.read_csv(metrics_path, sep="\t")
    split = json.loads(split_path.read_text())
    checkpoints = pd.read_csv(checkpoint_path, sep="\t")

    expected_rows = EXPECTED_FOLDS * len(EXPECTED_COUNTS) * len(EXPECTED_AXES)
    required = {"fold_id", "test_section", "axis", "n_train_villi", "pearson_r"}
    if required - set(frame.columns):
        raise RuntimeError(f"Metrics table lacks columns: {sorted(required - set(frame.columns))}")
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} metric rows; observed {len(frame)}")
    if sorted(frame["axis"].unique()) != sorted(EXPECTED_AXES):
        raise RuntimeError("Unexpected coordinate axes in scaling metrics")
    if sorted(frame["n_train_villi"].unique()) != EXPECTED_COUNTS:
        raise RuntimeError("Unexpected training-villus counts in scaling metrics")
    if frame["fold_id"].nunique() != EXPECTED_FOLDS or frame["test_section"].nunique() != EXPECTED_FOLDS:
        raise RuntimeError("Expected eight distinct held-out-section folds")
    condition_sizes = frame.groupby(["axis", "n_train_villi", "fold_id"]).size()
    if not condition_sizes.eq(1).all():
        raise RuntimeError("Each axis/count/fold must have exactly one test metric")
    if split.get("status") != "LOCKED_BEFORE_MODEL_FITTING_OR_TEST_METRICS":
        raise RuntimeError("Scaling split was not locked before model fitting")
    if split.get("aggregation_unit") != "held-out tissue section":
        raise RuntimeError("Panel E requires held-out tissue sections as the aggregation unit")
    if split.get("training_villus_counts") != EXPECTED_COUNTS or len(split.get("folds", [])) != EXPECTED_FOLDS:
        raise RuntimeError("Split manifest does not describe the expected eight-fold design")
    for fold in split["folds"]:
        train = set(map(str, fold["training_sections"]))
        validation = str(fold["validation_section"])
        test = str(fold["test_section"])
        if validation == test or validation in train or test in train:
            raise RuntimeError(f"Section leakage in {fold['fold_id']}")

    if len(checkpoints) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} checkpoint records; observed {len(checkpoints)}")
    checkpoint_key = ["fold_id", "axis", "n_train_villi"]
    if checkpoints.duplicated(checkpoint_key).any():
        raise RuntimeError("Duplicate model records in checkpoint manifest")
    for row in checkpoints.itertuples(index=False):
        from checkpoint_lineage import validate_graph_checkpoint_reference
        validate_graph_checkpoint_reference(row.checkpoint, row.checkpoint_sha256)

    output_dir = args.output_dir.expanduser().resolve()
    output_pdf = output_dir / "panel_E_training_villus_scaling.pdf"
    output_png = output_dir / "panel_E_training_villus_scaling.png"
    provenance_path = output_dir / "panel_E_training_villus_scaling_provenance.json"
    summary_path = output_dir / "panel_E_training_villus_scaling_summary.tsv"
    overwritten = protect_outputs([output_pdf, output_png, provenance_path, summary_path], args.overwrite)

    summary = (
        frame.groupby(["axis", "n_train_villi"], as_index=False)["pearson_r"]
        .agg(mean="mean", sem="sem", minimum="min", maximum="max", held_out_sections="count")
    )
    summary.to_csv(summary_path, sep="\t", index=False)
    labels = {"crypt_villus": "Crypt-villus coordinate", "epithelial_distance": "Epithelial-distance coordinate"}
    colors = {"crypt_villus": "#1f77b4", "epithelial_distance": "#d62728"}
    plt.rcParams.update({"font.size": 8})
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    rng = np.random.default_rng(550000)
    for coordinate in EXPECTED_AXES:
        raw = frame[frame["axis"].eq(coordinate)].copy()
        for count in EXPECTED_COUNTS:
            subset_raw = raw[raw["n_train_villi"].eq(count)].sort_values("fold_id")
            jitter = rng.uniform(-0.34, 0.34, len(subset_raw))
            axis.scatter(count + jitter, subset_raw["pearson_r"], s=9, color=colors[coordinate], alpha=0.23, linewidths=0, zorder=1)
        subset = summary[summary["axis"].eq(coordinate)].sort_values("n_train_villi")
        x = subset["n_train_villi"].to_numpy()
        y = subset["mean"].to_numpy()
        sem = subset["sem"].to_numpy()
        axis.plot(x, y, marker="o", markersize=4.1, linewidth=2, color=colors[coordinate], label=labels[coordinate], zorder=3)
        axis.fill_between(x, y - sem, y + sem, color=colors[coordinate], alpha=0.16, linewidth=0, zorder=2)
    axis.set_xlabel("Training villi", fontweight="bold")
    axis.set_ylabel("Held-out-section Pearson r", fontweight="bold")
    lower = max(-1.0, min(-0.05, float(frame["pearson_r"].min()) - 0.05))
    axis.set_ylim(lower, 1.0)
    axis.set_xlim(0, 47)
    axis.set_xticks([1, 10, 20, 30, 40, 45])
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_pdf, bbox_inches="tight", dpi=args.dpi)
    figure.savefig(output_png, bbox_inches="tight", dpi=args.dpi)
    plt.close(figure)

    seeds = {
        f"{axis_name}_minimum": int(min(run["seeds"][axis_name] for fold in split["folds"] for run in fold["runs"]))
        for axis_name in EXPECTED_AXES
    }
    write_provenance(
        provenance_path,
        script=__file__,
        inputs=[
            file_record(metrics_path, "eight_fold_section_disjoint_metrics"),
            file_record(split_path, "locked_section_disjoint_scaling_split"),
            file_record(checkpoint_path, "160_model_checkpoint_manifest"),
        ],
        outputs=[output_pdf, output_png, summary_path],
        seeds=seeds | {"display_jitter": 550000},
        warnings=[
            "Error bands are mean plus or minus SEM across eight held-out tissue sections, not cells.",
            "The upstream 30-dimensional scVI representation was learned jointly across the dataset; supervised GAT fitting, validation, checkpoint selection, and testing are section-disjoint.",
        ],
        extra={
            "panel": "Figure 2e",
            "evaluation_status": "EIGHT_FOLD_SECTION_DISJOINT_SCALING",
            "split_sha256": sha256(split_path),
            "fold_count": EXPECTED_FOLDS,
            "checkpoint_count": len(checkpoints),
            "display": "individual held-out-section values and mean plus or minus SEM",
        },
        overwritten=overwritten,
    )
    print(output_pdf)
    print(output_png)


if __name__ == "__main__":
    main()
