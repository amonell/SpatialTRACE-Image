from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crypt_villus_vit.gates import GATE_COLORS
from crypt_villus_vit.gates import GATE_ORDER


def plot_prediction_scatter(predictions_csv: Path, output_path: Path) -> Path:
    table = pd.read_csv(predictions_csv)
    has_axis_target = "target_axis" in table.columns
    has_epithelial_target = "epithelial_distance_clipped_1p0" in table.columns
    if has_axis_target or has_epithelial_target:
        figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=160)
        plot_axis = axes[0]
    else:
        figure, plot_axis = plt.subplots(figsize=(5.5, 4.8), dpi=160)
        axes = None
    for gate in GATE_ORDER:
        subset = table[table["predicted_gate_name"].astype(str) == gate]
        if subset.empty:
            continue
        plot_axis.scatter(
            subset["predicted_epithelial_distance_clipped_1p0"],
            subset["predicted_axis_coordinate"],
            s=10,
            alpha=0.65,
            label=gate,
            color=GATE_COLORS[gate],
            linewidths=0,
        )
    plot_axis.set_xlim(0, 1)
    plot_axis.set_ylim(0, 1)
    plot_axis.set_xlabel("Predicted epithelial distance")
    plot_axis.set_ylabel("Predicted crypt-villus axis")
    plot_axis.legend(frameon=False, fontsize=8)
    plot_axis.grid(True, alpha=0.2)
    if axes is not None:
        truth_axis = axes[1]
        if has_axis_target:
            truth_axis.scatter(
                table["target_axis"],
                table["predicted_axis_coordinate"],
                s=22,
                color="#2C7DA0",
                alpha=0.85,
                label="axis",
            )
        if has_epithelial_target:
            truth_axis.scatter(
                table["epithelial_distance_clipped_1p0"],
                table["predicted_epithelial_distance_clipped_1p0"],
                s=22,
                color="#B08968",
                alpha=0.85,
                label="epithelial distance",
            )
        truth_axis.plot([0, 1], [0, 1], color="#111827", linewidth=1.0, alpha=0.7)
        truth_axis.set_xlim(0, 1)
        truth_axis.set_ylim(0, 1)
        truth_axis.set_xlabel("Ground truth")
        truth_axis.set_ylabel("Predicted")
        truth_axis.legend(frameon=False, fontsize=8)
        truth_axis.grid(True, alpha=0.2)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
