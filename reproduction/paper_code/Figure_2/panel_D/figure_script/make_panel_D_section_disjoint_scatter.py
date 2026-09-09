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
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score


FIGURE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FIGURE_ROOT))
from common import file_record, protect_outputs, require_file, write_provenance  # noqa: E402


EXPECTED = {
    "predictions": "56b4647a7bce1956f104386c6e8bbab6941c6891dcd56495989133179cc7a660",
    "metrics": "3f66dc38c9a64905e07e23d287a136c71334f50ed94c1736574f1e099b8b275f",
    "split": "746c72b9181d33883f43bc9db481c023d02abadfe67346df5b90d18f05c210f2",
    "checkpoints": "0abf125b2110d0dded82c16f9b4b5ddf18a41fea1b7bb90af8f605bfa2526238",
    "bootstrap": "78d60e67d465ffacc716bbd6a27c0285aab8799b78030f495e62c5bfde2cac7c",
}


def calculate(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(reference)),
        "pearson_r": float(stats.pearsonr(reference, prediction).statistic),
        "spearman_rho": float(stats.spearmanr(reference, prediction).statistic),
        "mae": float(mean_absolute_error(reference, prediction)),
        "r2": float(r2_score(reference, prediction)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render final section-disjoint Figure 2d component.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = require_file(args.predictions, EXPECTED["predictions"])
    metrics_path = require_file(args.metrics, EXPECTED["metrics"])
    split_path = require_file(args.split_manifest, EXPECTED["split"])
    checkpoint_path = require_file(args.checkpoint_manifest, EXPECTED["checkpoints"])
    bootstrap_path = require_file(args.bootstrap_metrics, EXPECTED["bootstrap"])

    output_dir = args.output_dir.expanduser().resolve()
    output_pdf = output_dir / "panel_D_section_disjoint_scatter.pdf"
    output_png = output_dir / "panel_D_section_disjoint_scatter.png"
    provenance_path = output_dir / "panel_D_section_disjoint_scatter_provenance.json"
    overwritten = protect_outputs([output_pdf, output_png, provenance_path], args.overwrite)

    frame = pd.read_parquet(predictions_path)
    saved = json.loads(metrics_path.read_text())
    split = json.loads(split_path.read_text())
    checkpoints = pd.read_csv(checkpoint_path, sep="\t")
    if len(frame) != 25_436 or frame["global_index"].nunique() != 25_436:
        raise RuntimeError("Panel D requires exactly 25,436 unique out-of-fold cells")
    if frame["batch"].nunique() != 8 or frame["training_villus_id"].nunique() != 72:
        raise RuntimeError("Panel D requires all eight sections and 72 annotated villi")
    if not (frame["evaluation_status"] == "SECTION_DISJOINT_OUT_OF_FOLD").all():
        raise RuntimeError("Prediction rows are not marked section-disjoint out-of-fold")
    if split.get("status") != "LOCKED_BEFORE_TRAINING_OR_TEST_METRICS":
        raise RuntimeError("Split manifest is not locked")
    if len(checkpoints) != 16 or checkpoints["checkpoint_sha256"].nunique() != 16:
        raise RuntimeError("Expected 16 distinct fold-by-coordinate checkpoints")
    for row in checkpoints.itertuples(index=False):
        from checkpoint_lineage import validate_graph_checkpoint_reference
        validate_graph_checkpoint_reference(row.checkpoint, row.checkpoint_sha256)

    specifications = (
        ("crypt_villus", "crypt_villus_label", "predicted_crypt_villus", "Crypt-villus coordinate", "viridis"),
        (
            "epithelial_distance",
            "epithelial_distance_label",
            "predicted_epithelial_distance",
            "Epithelial-distance coordinate",
            "coolwarm",
        ),
    )
    recomputed: dict[str, dict[str, float | int]] = {}
    for coordinate, reference_column, prediction_column, _, _ in specifications:
        values = calculate(frame[reference_column].to_numpy(float), frame[prediction_column].to_numpy(float))
        recomputed[coordinate] = values
        for metric, value in values.items():
            if not np.isclose(value, saved["pooled"][coordinate][metric], rtol=0, atol=1e-12):
                raise RuntimeError(f"Metric mismatch for {coordinate}/{metric}")

    plt.rcParams.update({"font.size": 8})
    figure, axes = plt.subplots(1, 2, figsize=(7.3, 3.25))
    figure.subplots_adjust(left=0.075, right=0.965, bottom=0.16, top=0.80, wspace=0.39)
    for axis, (coordinate, reference_column, prediction_column, title, cmap) in zip(axes, specifications, strict=True):
        artist = axis.hexbin(
            frame[reference_column], frame[prediction_column], gridsize=55,
            extent=(0, 1, 0, 1), mincnt=1, cmap=cmap, norm=LogNorm(),
        )
        axis.plot([0, 1], [0, 1], "--", color="#222222", linewidth=0.9)
        value = recomputed[coordinate]
        axis.text(
            0.03, 0.97,
            f"Pearson r = {value['pearson_r']:.3f}\n"
            f"MAE = {value['mae']:.3f}\n"
            f"R² = {value['r2']:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            linespacing=1.25,
            zorder=10,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#444444", "linewidth": 0.6, "alpha": 0.96},
        )
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Manual reference", ylabel="Graph-model prediction")
        axis.set_aspect("equal")
        axis.set_title(title, fontweight="bold")
        colorbar = figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.02)
        colorbar.set_label("Cell count")
    figure.suptitle("Section-disjoint out-of-fold evaluation", y=0.985, fontsize=9.2, fontweight="bold")
    figure.savefig(output_pdf, bbox_inches="tight", dpi=args.dpi)
    figure.savefig(output_png, bbox_inches="tight", dpi=args.dpi)
    plt.close(figure)

    write_provenance(
        provenance_path,
        script=__file__,
        inputs=[
            file_record(predictions_path, "section_disjoint_predictions"),
            file_record(metrics_path, "verified_metrics"),
            file_record(split_path, "locked_split_manifest"),
            file_record(checkpoint_path, "fold_checkpoint_manifest"),
            file_record(bootstrap_path, "section_grouped_bootstrap"),
        ],
        outputs=[output_pdf, output_png],
        seeds={"fold_model_seed_min": 420000, "fold_model_seed_max": 420015},
        warnings=[split["feature_lineage"]["limitation"]],
        extra={
            "panel": "Figure 2d",
            "evaluation_status": "SECTION_DISJOINT_OUT_OF_FOLD",
            "reference_lineage": "Manual LabelMe-derived coordinate annotations",
            "visible_metrics": ["pearson_r", "mae", "r2"],
            "metrics": recomputed,
        },
        overwritten=overwritten,
    )
    print(json.dumps(recomputed, indent=2))


if __name__ == "__main__":
    main()
