#!/usr/bin/env python3
"""Plot a frozen interim coordinate snapshot without fitting or inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from render_review_panels import COLORS, GATES, config, gate_background, sha


CONDITION_COLORS = {"DMSO": "#37788F", "RARi": "#B05B49"}


class Snapshot:
    def __init__(self, source, output):
        self.source, self.output = source.resolve(), output.resolve()
        if self.output.exists():
            raise FileExistsError(f"Choose a new preview output directory: {self.output}")
        self.manifest = json.loads((self.source / "manifest.json").read_text())
        if self.manifest["status"] not in {"frozen_interim_coordinate_preview", "frozen_nine_mouse_figure_release"}:
            raise RuntimeError("Expected a frozen interim snapshot")
        self.inputs = self.manifest["outputs"]
        for entry in self.inputs:
            if sha(entry["path"]) != entry["sha256"]:
                raise RuntimeError(f"Snapshot input changed: {entry['path']}")
        self.files = {Path(r["path"]).name: Path(r["path"]) for r in self.inputs}
        self.sections = pd.read_csv(self.files["sections.csv"])
        self.sections["condition_order"] = self.sections.condition.map({"RARi": 0, "DMSO": 1})
        self.sections["cohort_order"] = self.sections.cohort.map({"new": 0, "original": 1})
        self.sections = self.sections.sort_values(["condition_order", "cohort_order", "source_id"])
        self.gates = pd.read_csv(self.files["gates.csv"])
        self.outputs = []
        (self.output / "pdfs").mkdir(parents=True)
        (self.output / "previews").mkdir()

    def label(self, row):
        return "Original" if row.cohort == "original" else row.source_id[-2:]

    def footer(self):
        if self.manifest["status"] == "frozen_nine_mouse_figure_release":
            return "Nine mice: four DMSO and five RARi. Original mice shown separately; tissue 08 excluded at the author's request."
        counts = self.manifest["n_new_mice_by_condition"]
        pending = len(self.manifest["pending_source_ids"])
        return (f"Interim snapshot: {counts.get('RARi', 0)} new RARi and {counts.get('DMSO', 0)} new DMSO mice; "
                f"{pending} new sections pending. Original mice shown separately. Fixed gates; no treatment tests.")

    def save(self, fig, stem):
        if self.manifest["status"] == "frozen_nine_mouse_figure_release":
            stem = stem.replace("Preview_", "Additional_")
        for folder, extension, dpi in (("pdfs", "pdf", 600), ("previews", "png", 160)):
            path = self.output / folder / f"{stem}.{extension}"
            if path.exists():
                raise FileExistsError(path)
            fig.savefig(path, dpi=dpi)
            self.outputs.append({"path": str(path), "sha256": sha(path)})
        plt.close(fig)
        print(f"Rendered {self.output / 'pdfs' / (stem + '.pdf')}", flush=True)


def regional_comparison(data):
    fig, axes = plt.subplots(3, 5, figsize=(13, 9.4))
    fig.subplots_adjust(left=.095, right=.985, top=.895, bottom=.125, hspace=.42, wspace=.47)
    fig.suptitle("Regional P14 distributions across mice", x=.5, y=.975, fontsize=14)
    metrics = [("p14_composition_percent", "P14 in region\n(% of all P14 cells)"),
               ("p14_per_10000_non_p14", "P14 per 10,000 non-P14 cells\nin the same region"),
               ("p14_regional_enrichment", "Regional enrichment\n(P14 fraction / non-P14 fraction)")]
    for row_index, (metric, ylabel) in enumerate(metrics):
        for j, gate in enumerate(GATES):
            ax = axes[row_index, j]
            plotted = []
            for x, condition in enumerate(("RARi", "DMSO")):
                mice = data.sections.loc[data.sections.condition == condition]
                offsets = np.linspace(-.22, .22, len(mice)) if len(mice) > 1 else [0]
                for offset, (_, mouse) in zip(offsets, mice.iterrows()):
                    value = data.gates.loc[(data.gates.source_id == mouse.source_id) & (data.gates.gate == gate), metric].item()
                    old = mouse.cohort == "original"
                    color = CONDITION_COLORS[condition]
                    ax.scatter(x + offset, value, marker="s" if old else "o", s=30,
                               facecolors="white" if old else color, edgecolors=color, linewidths=1, zorder=3)
                    ax.annotate("O" if old else mouse.source_id[-2:], (x + offset, value),
                                xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.4, color=color)
                    plotted.append(value)
            if metric == "p14_regional_enrichment":
                ax.axhline(1, color=".6", lw=.7, ls="--", zorder=0)
            ax.set(xticks=[0, 1], xticklabels=["RARi", "DMSO"], xlim=(-.5, 1.5),
                   ylim=(0, max(max(plotted) * 1.23, 1.25 if row_index == 2 else .1)))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=".92", lw=.5, zorder=0)
            if row_index == 0:
                ax.set_title(gate, fontsize=10, fontweight="bold", pad=10)
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=10)
                ax.text(-.45, 1.07, chr(97 + row_index), transform=ax.transAxes, fontsize=12, fontweight="bold")
    handles = [Line2D([], [], marker="o", color=".4", linestyle="none", label="New mouse (number = section)"),
               Line2D([], [], marker="s", markerfacecolor="white", color=".4", linestyle="none", label="Original mouse (O)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .947), frameon=False, ncol=2)
    fig.text(.5, .063, "Enrichment of 1: the region contains the same share of P14 and non-P14 cells.", ha="center", fontsize=8)
    fig.text(.5, .025, data.footer(), ha="center", fontsize=7.5, color=".35")
    data.save(fig, "Preview_01_regional_consistency")


def imaps(data):
    width = max(int((data.sections.condition == condition).sum()) for condition in ("RARi", "DMSO"))
    fig, axes = plt.subplots(2, width, figsize=(3.05 * width, 7.1), squeeze=False)
    fig.subplots_adjust(left=.052, right=.985, top=.86, bottom=.13, wspace=.23, hspace=.48)
    fig.suptitle("P14 anatomical-coordinate maps", y=.98, fontsize=14)
    fig.text(.5, .93, "Identical coordinate scales and gate boundaries for every mouse", ha="center", fontsize=9)
    for row_index, condition in enumerate(("RARi", "DMSO")):
        mice = data.sections.loc[data.sections.condition == condition]
        for col, (_, mouse) in enumerate(mice.iterrows()):
            ax = axes[row_index, col]
            frame = pd.read_parquet(data.files[f"{mouse.source_id}.parquet"])
            frame = frame.loc[frame.is_p14 == 1]
            gate_background(ax)
            ax.scatter(frame.predicted_epithelial_distance_clipped_1p0, frame.predicted_axis_coordinate,
                       c=CONDITION_COLORS[condition], s=1.4, alpha=.4, linewidths=0)
            table = data.gates.loc[data.gates.source_id == mouse.source_id].set_index("gate")
            for gate, x, y, ha in (("Top IE", .025, .80, "left"), ("Top LP", .975, .80, "right"),
                                   ("Crypt IE", .025, .08, "left"), ("Crypt LP", .975, .325, "right"),
                                   ("Muscularis", .975, .08, "right")):
                title = "Musc" if gate == "Muscularis" else gate
                ax.text(x, y, f"{title}\n{table.loc[gate, 'p14_composition_percent']:.1f}%",
                        ha=ha, va="center", fontsize=7,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 1})
            label = f"{condition} · {data.label(mouse)}"
            ax.set_title(f"{label}\n{len(frame):,} P14 cells", color=CONDITION_COLORS[condition], fontsize=9, pad=9)
            ax.set_aspect("equal")
            if col == 0:
                ax.set_ylabel("Crypt–villus coordinate")
            if row_index == 1:
                ax.set_xlabel("Epithelial-distance coordinate")
        for col in range(len(mice), width):
            axes[row_index, col].axis("off")
    fig.text(.5, .035, data.footer(), ha="center", fontsize=8, color=".35")
    data.save(fig, "Preview_02_individual_P14_IMAPs")


def compositions(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=.22, wspace=.25)
    fig.suptitle("Regional composition of P14 and background cells", y=.97, fontsize=14)
    n = len(data.sections)
    for ax, metric, title in zip(axes, ("p14_composition_percent", "non_p14_composition_percent"),
                                  ("P14 cells", "Non-P14 cells")):
        bottom = np.zeros(n)
        for gate, color in zip(GATES, COLORS):
            values = data.gates.loc[data.gates.gate == gate].set_index("source_id").loc[data.sections.source_id, metric].to_numpy()
            ax.bar(np.arange(n), values, bottom=bottom, color=color, width=.72, label=gate)
            bottom += values
        labels = [f"{mouse.condition}\n{data.label(mouse)}" for _, mouse in data.sections.iterrows()]
        ax.set(xticks=np.arange(n), xticklabels=labels, ylim=(0, 100), ylabel="Cells in each region (%)", title=title)
        ax.spines[["top", "right"]].set_visible(False)
        for label, (_, mouse) in zip(ax.get_xticklabels(), data.sections.iterrows()):
            label.set_color(CONDITION_COLORS[mouse.condition])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .9), ncol=5, frameon=False)
    fig.text(.5, .095, "Each bar represents one mouse; all cells are included.", ha="center", fontsize=9)
    fig.text(.5, .035, data.footer(), ha="center", fontsize=7.5, color=".35")
    data.save(fig, "Preview_03_regional_composition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config()
    data = Snapshot(args.input, args.output)
    regional_comparison(data)
    imaps(data)
    compositions(data)
    manifest = {"status": "rendered_pending_visual_review", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": [sys.executable, *sys.argv], "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib", "pyarrow")},
                "code": [{"path": str(path.resolve()), "sha256": sha(path)} for path in
                         (Path(__file__), Path(__file__).with_name("render_review_panels.py"))],
                "git_state": "Project directory is not a Git repository", "random_seed": None,
                "inputs": [{"path": str(data.source / "manifest.json"), "sha256": sha(data.source / "manifest.json")}, *data.inputs],
                "outputs": data.outputs, "warnings": data.manifest["warnings"]}
    with (data.output / "rendering_manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
