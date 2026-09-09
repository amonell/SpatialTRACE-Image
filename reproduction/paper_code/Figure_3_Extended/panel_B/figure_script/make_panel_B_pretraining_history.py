#!/usr/bin/env python3
"""Plot the frozen paired-representation pretraining history."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_3_Extended.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402

EXPECTED_HISTORY = "d3227f0c39ee235cb04335b8da0f55dad87a8f46398fae243a56e014208fbb59"
EXPECTED_SUMMARY = "daecdd0e75be017a65bd231a062d2a7ca77960cf237a169ab4dca552ae023f1f"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    history_path = require_file(args.history, EXPECTED_HISTORY)
    summary_path = require_file(args.summary, EXPECTED_SUMMARY)
    history = pd.read_csv(history_path)
    summary = json.loads(summary_path.read_text())
    if len(history) != 35 or int(summary["epochs_completed"]) != 35:
        raise RuntimeError("Expected exactly 35 completed representation-pretraining epochs")
    selected = int(summary["best_epoch"])
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_B_paired_representation_pretraining_history"
    table = out / "panel_B_pretraining_history.tsv"
    provenance = out / "panel_B_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), table, provenance], args.overwrite)
    specs = (
        ("train_loss", "validation_loss", "Composite representation loss"),
        ("train_alignment_loss", "validation_alignment_loss", "Alignment loss"),
        ("train_feature_std", "validation_feature_std", "Feature standard deviation"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(8.0, 2.55), sharex=True)
    for axis, (train_key, validation_key, title) in zip(axes, specs, strict=True):
        axis.plot(history["epoch"], history[train_key], color="#333333", lw=1.35, label="Training")
        axis.plot(history["epoch"], history[validation_key], color="#3575a8", lw=1.45, label="Validation")
        axis.axvline(selected, color="#9b2f2f", lw=0.9, ls="--")
        selected_value = history.loc[history["epoch"].astype(int).eq(selected), validation_key].iloc[0]
        axis.scatter([selected], [selected_value], color="#9b2f2f", marker="*", s=42, zorder=4)
        axis.set_title(title, fontsize=8.5)
        axis.set_xlabel("Completed epoch")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    figure.suptitle("Paired representation pretraining", fontsize=9.5, y=0.99)
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.20, top=0.82, wspace=0.35)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    history.to_csv(table, sep="\t", index=False)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(history_path, "frozen_pretraining_history"), file_record(summary_path, "frozen_pretraining_summary")],
        outputs=[*rendered, table],
        extra={
            "panel": "Extended Figure 3b",
            "completed_epochs": 35,
            "selected_epoch": selected,
            "selection_metric": "minimum validation composite representation loss",
            "objective": summary["objective"],
            "model_inference_run": False,
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()

