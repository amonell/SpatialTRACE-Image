#!/usr/bin/env python3
"""Render the frozen Peyer's patch image-classifier strategy comparison."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_3_Extended.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402

ORDER = (
    "frozen_manual_sparse",
    "full_manual_sparse",
    "frozen_gat_soft_sparse",
    "full_gat_soft_sparse",
)
LABELS = {
    "frozen_manual_sparse": "Manual · frozen ViT",
    "full_manual_sparse": "Manual · full fine-tuning",
    "frozen_gat_soft_sparse": "GAT soft · frozen ViT",
    "full_gat_soft_sparse": "GAT soft · full fine-tuning",
}
COLORS = {
    "frozen_manual_sparse": "#3569a8",
    "full_manual_sparse": "#3569a8",
    "frozen_gat_soft_sparse": "#d36f2f",
    "full_gat_soft_sparse": "#d36f2f",
}
LINESTYLES = {
    "frozen_manual_sparse": "--",
    "full_manual_sparse": "-",
    "frozen_gat_soft_sparse": "--",
    "full_gat_soft_sparse": "-",
}
from frozen_preprocessing.peyer_representation_v1.release import load_release, RELEASE_POINTER


def load_frozen_roc(manifest_path: Path, summary: pd.DataFrame):
    """Read and validate prepared display tables; no metrics are calculated."""
    _, release = load_release()
    manifest_path = require_file(manifest_path, release["roc_manifest"]["sha256"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "FROZEN_ROC_DISPLAY_TABLES_V1" or manifest["variant_order"] != list(ORDER):
        raise RuntimeError("Unexpected frozen ROC display release")
    if manifest["heldout_cells"] != 819 or manifest["positive_cells"] != 91:
        raise RuntimeError("Frozen ROC cohort changed")
    curve_path = require_file(manifest["curve_table"]["path"], manifest["curve_table"]["sha256"])
    metric_path = require_file(manifest["display_metrics"]["path"], manifest["display_metrics"]["sha256"])
    frame = pd.read_csv(curve_path, sep="\t", float_precision="round_trip")
    metrics = pd.read_csv(metric_path, sep="\t", float_precision="round_trip")
    if set(frame.variant) != set(ORDER):
        raise RuntimeError("Incomplete ROC variants")
    curves = {}
    for record in manifest["verified_metrics"]:
        variant = record["variant"]
        saved = summary.loc[summary.variant.eq(variant)].iloc[0]
        displayed = metrics.loc[metrics.variant.eq(variant)].iloc[0]
        for name, source_name in (("auroc", "test_auroc"), ("average_precision", "test_average_precision")):
            if abs(float(record[name]) - float(saved[source_name])) > 1e-12 or abs(float(displayed[name]) - float(record[name])) > 1e-12:
                raise RuntimeError("Prepared display metrics differ from the frozen scientific summary")
        points = frame.loc[frame.variant.eq(variant)]
        for name in ("fpr", "tpr"):
            values = points[name]
            if not values.between(0, 1).all() or values.diff().dropna().lt(-1e-14).any() or values.iloc[0] != 0 or values.iloc[-1] != 1:
                raise RuntimeError("Invalid frozen ROC coordinates")
        curves[variant] = (points.fpr, points.tpr)
    return manifest_path, curve_path, metric_path, manifest["verified_metrics"], curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--roc-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    active_path, active = load_release()
    release_path = require_file(args.release_manifest, active["comparison"]["sha256"])
    release = json.loads(release_path.read_text())
    if release['selected_variant'] != 'full_manual_sparse' or release['selection_changed'] or release['threshold'] != .5:
        raise RuntimeError('The prespecified classifier strategy or hard-call threshold changed')
    require_file(release['lock']['path'], release['lock']['sha256'])
    models = release['models']
    summary_path = require_file(args.summary, release['strategy_metrics']['sha256'])
    checkpoint = require_file(args.checkpoint, models['full_manual_sparse']['checkpoint']['sha256'])
    model_paths = {name: require_file(models[name]['checkpoint']['path'], models[name]['checkpoint']['sha256']) for name in ORDER}
    table = pd.read_csv(summary_path, sep='\t').set_index("variant").reindex(ORDER).reset_index()
    table = table.rename(columns={'auroc': 'test_auroc', 'average_precision': 'test_average_precision'})
    table['checkpoint'] = table.variant.map({name: str(path) for name, path in model_paths.items()})
    if table["variant"].isna().any() or set(table["variant"]) != set(ORDER):
        raise RuntimeError("Peyer's classifier strategy table is incomplete")
    selected_path = Path(table.loc[table["variant"].eq("full_manual_sparse"), "checkpoint"].iloc[0].replace("/Figure5/", "/Peyers_Pipeline/"))
    if selected_path.resolve() != checkpoint.resolve():
        raise RuntimeError("Selected strategy checkpoint does not match the supplied frozen checkpoint")
    roc_manifest, frozen_curves, frozen_metrics, verified_rows, curves = load_frozen_roc(args.roc_manifest, table)
    verified = pd.DataFrame(verified_rows)
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_H_peyer_classifier_strategy"
    source_table = out / "panel_H_peyer_classifier_strategy.tsv"
    curve_table = out / "panel_H_roc_curve.tsv"
    provenance = out / "panel_H_provenance.json"
    overwritten = protect_outputs(
        [stem.with_suffix(".pdf"), stem.with_suffix(".png"), source_table, curve_table, provenance],
        args.overwrite,
    )

    figure, axis = plt.subplots(figsize=(3.65, 3.05))
    axis.plot([0, 1], [0, 1], color="#999999", lw=0.8, linestyle="--", zorder=1)
    for variant in ORDER:
        row = verified.loc[verified["variant"].eq(variant)].iloc[0]
        fpr, tpr = curves[variant]
        axis.plot(
            fpr,
            tpr,
            color=COLORS[variant],
            linestyle=LINESTYLES[variant],
            lw=2.0 if "full" in variant else 1.5,
            label=f"{LABELS[variant]}  AUROC = {row['auroc']:.2f}",
            zorder=3 if "full" in variant else 2,
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Peyer's patch classifier strategy", fontsize=9.5)
    axis.text(0.97, 0.04, "n = 819; 91 positive", ha="right", va="bottom", fontsize=7, color="#444444")
    axis.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), frameon=False, fontsize=7, handlelength=2.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#e5e5e5", lw=0.45)
    figure.subplots_adjust(left=0.18, right=0.985, bottom=0.18, top=0.90)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    shutil.copyfile(frozen_metrics, source_table)
    shutil.copyfile(frozen_curves, curve_table)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[
            file_record(summary_path, "frozen_classifier_strategy_metrics"),
            file_record(RELEASE_POINTER, "active_classifier_release_pointer"),
            file_record(active_path, "representation_classifier_release"),
            file_record(checkpoint, "selected_peyer_image_classifier_checkpoint"),
            file_record(release_path, "corrected_label_sparse_classifier_release"),
            *[file_record(model_paths[name], f"{name}_checkpoint") for name in ORDER],
            file_record(roc_manifest, "frozen_ROC_preparation_manifest"),
            file_record(frozen_curves, "frozen_ROC_coordinates"),
            file_record(frozen_metrics, "frozen_ROC_display_metrics"),
        ],
        outputs=[*rendered, source_table, curve_table],
        extra={
            "panel": "Extended Figure 3h",
            "lineage": "representation-pretrained axis-architecture Peyer classifier",
            "selected_variant": "full_manual_sparse",
            "selection_warning": "Full-model/manual-target SFT was prespecified before fitting; test results do not select the production strategy.",
            "display": "complete 2x2 ROC comparison: manual versus GAT-soft training targets crossed with frozen-encoder versus full-model SFT",
            "heldout_cells": 819,
            "positive_cells": 91,
            "frozen_verified_metrics": verified_rows,
            "ROC_coordinates_recomputed_during_rendering": False,
            "metrics_recomputed_during_rendering": False,
            "panel_rendering_training_or_inference_run": False,
            "upstream_new_variant_training_run": True,
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
