#!/usr/bin/env python3
"""Plot the frozen three-seed pretraining-objective comparison."""
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
from Figure_3_Extended.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402

EXPECTED_METRICS = "6fc901ad0041a81dea630c9b9a00ca413da3bb1cb6406cb98d35ae787e890bae"
EXPECTED_RUN = "8ec3a212915d40fafd35f9da8cbd27dbdac085a082b3a85b41294fb3a31188da"
ORDER = ("scratch", "reconstruction", "representation")
LABELS = {
    "scratch": "No pretraining",
    "reconstruction": "Masked-pixel\nreconstruction",
    "representation": "Paired representation\nprediction",
}
COLORS = {"scratch": "#888888", "reconstruction": "#d8862b", "representation": "#2778ae"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metrics_path = require_file(args.metrics, EXPECTED_METRICS)
    run_path = require_file(args.run_summary, EXPECTED_RUN)
    metrics = pd.read_csv(metrics_path)
    run = json.loads(run_path.read_text())
    expected = {(condition, seed, axis) for condition in ORDER for seed in (42, 43, 44) for axis in ("crypt_villus", "epithelial_distance")}
    observed = set(zip(metrics["condition"], metrics["seed"].astype(int), metrics["axis"], strict=True))
    if observed != expected or run["evaluation"]["test_rows"] != 5107:
        raise RuntimeError("Pretraining-ablation source is not the complete matched three-seed analysis")
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_D_pretraining_objective_ablation"
    table = out / "panel_D_pretraining_ablation.tsv"
    mean_table = out / "panel_D_pretraining_ablation_means.tsv"
    provenance = out / "panel_D_provenance.json"
    overwritten = protect_outputs(
        [stem.with_suffix(".pdf"), stem.with_suffix(".png"), table, mean_table, provenance],
        args.overwrite,
    )
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.9), sharey=True)
    mean_records = []
    for axis, coordinate, title in zip(axes, ("crypt_villus", "epithelial_distance"), ("Crypt-villus coordinate", "Epithelial-distance coordinate"), strict=True):
        selected = metrics.loc[metrics["axis"].eq(coordinate)]
        for seed in (42, 43, 44):
            values = selected.loc[selected["seed"].astype(int).eq(seed)].set_index("condition").reindex(ORDER)["pearson_r"]
            axis.plot(np.arange(3), values, color="#c8c8c8", lw=0.8, zorder=1)
        for index, condition in enumerate(ORDER):
            values = selected.loc[selected["condition"].eq(condition), "pearson_r"].to_numpy(float)
            mean_value = float(values.mean())
            mean_records.append(
                {
                    "axis": coordinate,
                    "condition": condition,
                    "display_name": LABELS[condition].replace("\n", " "),
                    "mean_pearson_r": mean_value,
                    "n_seeds": int(len(values)),
                }
            )
            axis.scatter(index + np.linspace(-0.045, 0.045, len(values)), values, color=COLORS[condition], s=24, zorder=3)
            axis.hlines(mean_value, index - 0.16, index + 0.16, color=COLORS[condition], lw=2.2, zorder=4)
            axis.text(
                index,
                float(values.max()) + 0.006,
                f"mean = {mean_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color=COLORS[condition],
                zorder=5,
            )
        axis.set_xticks(np.arange(3), [LABELS[value] for value in ORDER], rotation=18, ha="right")
        axis.set_title(title, fontsize=8.5)
        axis.set_xlim(-0.35, 2.35)
        axis.set_ylim(0.65, 0.905)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pearson r")
    figure.suptitle("Pretraining initialization comparison · 5% labeled cells", fontsize=9.5, y=0.99)
    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.30, top=0.80, wspace=0.22)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    metrics.to_csv(table, sep="\t", index=False)
    means = pd.DataFrame(mean_records)
    means.to_csv(mean_table, sep="\t", index=False)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(metrics_path, "three_seed_pretraining_comparison"), file_record(run_path, "pretraining_ablation_run_summary")],
        outputs=[*rendered, table, mean_table],
        extra={
            "panel": "Extended Figure 3d",
            "supervised_label_fraction": 0.05,
            "interpretation": "Comparison of saved initialization recipes, not an isolated loss-only experiment; three supervised seeds, not three independent pretraining seeds",
            "seeds": [42, 43, 44],
            "evaluation_cells": 5107,
            "evaluation_scope": "prepared development test; locked SI2 confirmatory section not used",
            "model_training_or_inference_run": False,
            "mean_pearson_labels": means.to_dict(orient="records"),
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
