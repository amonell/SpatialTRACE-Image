#!/usr/bin/env python3
"""Plot frozen raw-crop shuffling sensitivity for the released full model."""
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

EXPECTED_DELTAS = "23608adef77b6a27417187df298ad466c960fedf5710d8dc348e91e8ef061247"
EXPECTED_SUMMARY = "55bf6d53307ba451cc16b27b1f0ef4a1b0e4bbaf00c77cbb4d1c4e0adba796ac"
EXPECTED_CHECKPOINT = "d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137"
ORDER = ("local", "context", "fine")
LABELS = {"local": "Local branch", "context": "Context branch", "fine": "Fine branch"}
COLORS = {"crypt_villus": "#3569a8", "epithelial_distance": "#d36f2f"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    deltas_path = require_file(args.deltas, EXPECTED_DELTAS)
    summary_path = require_file(args.summary, EXPECTED_SUMMARY)
    checkpoint = require_file(args.checkpoint, EXPECTED_CHECKPOINT)
    deltas = pd.read_csv(deltas_path)
    summary = json.loads(summary_path.read_text())
    expected = {(scale, coordinate, repeat) for scale in ORDER for coordinate in ("crypt_villus", "epithelial_distance") for repeat in range(5)}
    observed = set(zip(deltas["scale"], deltas["axis"], deltas["repeat"].astype(int), strict=True))
    if observed != expected or summary["method"] != "batch_shuffle" or int(summary["row_count"]) != 5107:
        raise RuntimeError("Branch-shuffling inputs do not match the frozen five-repeat raw-crop analysis")
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_E_branch_input_shuffling"
    table = out / "panel_E_branch_input_shuffling.tsv"
    provenance = out / "panel_E_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), table, provenance], args.overwrite)
    figure, axes = plt.subplots(2, 1, figsize=(3.5, 4.7), sharey=False)
    for axis, coordinate, title in zip(axes, ("crypt_villus", "epithelial_distance"), ("Crypt-villus coordinate", "Epithelial-distance coordinate"), strict=True):
        coordinate_rows = deltas.loc[deltas["axis"].eq(coordinate)]
        for index, scale in enumerate(ORDER):
            values = coordinate_rows.loc[coordinate_rows["scale"].eq(scale), "pearson_r_drop"].to_numpy(float)
            axis.scatter(index + np.linspace(-0.05, 0.05, len(values)), values, color=COLORS[coordinate], s=20, alpha=0.85, zorder=3)
            axis.hlines(values.mean(), index - 0.18, index + 0.18, color="#111111", lw=2.0, zorder=4)
        axis.axhline(0, color="#777777", lw=0.7)
        axis.set_xticks(np.arange(3), [LABELS[value] for value in ORDER], rotation=16, ha="right")
        axis.set_title(title, fontsize=8.5)
        axis.set_ylabel("Pearson r reduction")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Branch-input shuffling", fontsize=9.5, y=0.99)
    figure.subplots_adjust(left=0.19, right=0.99, bottom=0.11, top=0.88, hspace=0.68)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    deltas.to_csv(table, sep="\t", index=False)
    means = (
        deltas.groupby(["scale", "axis"], sort=False)["pearson_r_drop"].mean().rename("mean_pearson_r_reduction").reset_index().to_dict("records")
    )
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(deltas_path, "frozen_branch_shuffle_replicates"), file_record(summary_path, "frozen_shuffle_summary"), file_record(checkpoint, "released_full_model_checkpoint")],
        outputs=[*rendered, table],
        extra={
            "panel": "Extended Figure 3e",
            "component_order": ["crypt_villus_top", "epithelial_distance_bottom"],
            "former_panel": "Extended Figure 3f",
            "perturbation": "raw prepared branch crops permuted across cells within each inference batch",
            "method": "batch_shuffle",
            "repeat_count": 5,
            "evaluation_cells": 5107,
            "mean_reductions": means,
            "interpretation": "fitted-model sensitivity, not retrained ablation and not causal importance",
            "evaluation_scope": "prepared development test; locked SI2 confirmatory section not used",
            "model_training_or_inference_run": False,
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
