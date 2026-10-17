#!/usr/bin/env python3
"""Render the frozen two-stage production-v2 IF fine-tuning history."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_4_Extended.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402
from Figure_4.release import load_release, release_file
from Figure_4.frozen_annotations import joined_training_history
from Figure_4_Extended.history_labels import draw_stage_labels

EXPECTED = ('stage1_history', 'stage1_summary', 'stage2_history', 'stage2_summary')


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in EXPECTED:
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    release = load_release(args.release_manifest)
    paths = {name: release_file(getattr(args, name), release) for name in EXPECTED}
    checkpoint = release_file(args.checkpoint, release)
    summaries = {name: json.loads(paths[name].read_text()) for name in ("stage1_summary", "stage2_summary")}
    stage1, stage2 = pd.read_csv(paths["stage1_history"]), pd.read_csv(paths["stage2_history"])
    history, history_info = joined_training_history([stage1, stage2], [summaries["stage1_summary"], summaries["stage2_summary"]])
    selected1, selected2 = history_info["selected_stage_epochs"]
    combined1, combined2 = history_info["selected_combined_epochs"]
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_D_IF_fine_tuning_history"
    table = out / "panel_D_training_history.tsv"
    provenance = out / "panel_D_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), table, provenance], args.overwrite)
    specs = (("train_loss", "validation_loss", "Smooth L1 objective"), ("train_axis_mae", "validation_axis_mae", "Crypt-villus MAE"), ("train_epithelial_mae", "validation_epithelial_mae", "Epithelial-distance MAE"))
    figure, axes = plt.subplots(1, 3, figsize=(7.65, 2.55), sharex=True)
    for axis, (train_key, validation_key, title) in zip(axes, specs, strict=True):
        axis.plot(history.completed_epoch, history[train_key], color="#4c4c4c", marker="o", ms=2.7, lw=1.2, label="Training")
        axis.plot(history.completed_epoch, history[validation_key], color="#2878a8", marker="o", ms=2.7, lw=1.3, label="Validation")
        axis.axvline(history_info["stage_boundary"], color="#777777", lw=0.8, ls=":")
        axis.scatter([selected1], [history.loc[history.completed_epoch.eq(selected1), validation_key].iloc[0]], facecolor="white", edgecolor="#777777", marker="*", s=38, linewidth=0.8, zorder=4, label="Stage-1 selection" if axis is axes[0] else None)
        axis.scatter([combined2], [history.loc[history.completed_epoch.eq(combined2), validation_key].iloc[0]], color="#c75b12", s=26, zorder=4, label="Selected checkpoint" if axis is axes[0] else None)
        axis.set_title(title, fontsize=8.5)
        axis.set_xlabel("Completed epoch")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5)
    stage_labels = draw_stage_labels(axes[1], history_info, fontsize=7.5, y=.98)
    figure.suptitle("IF supervised fine-tuning history", fontsize=9.5, y=0.99)
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.20, top=0.82, wspace=0.34)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    history.to_csv(table, sep="\t", index=False)
    write_provenance(
        provenance, script=Path(__file__), inputs=[file_record(args.release_manifest, "repaired_IF_release"), *[file_record(paths[name], f"frozen_{name}") for name in EXPECTED], file_record(REPO / "Figure_4/frozen_annotations.py", "frozen_history_annotation_helper"), file_record(REPO / "Figure_4_Extended/history_labels.py", "training_stage_label_helper"), file_record(checkpoint, "released_IF_checkpoint")], outputs=[*rendered, table],
        extra={"panel": "Extended Figure 4d", "completed_epochs": len(history), "stage1_completed_epochs": len(stage1), "stage1_selected_epoch": selected1, "stage2_initialized_from_stage1_selected_epoch": selected1, "stage2_completed_epochs": len(stage2), "selected_checkpoint_stage": 2, "selected_stage_epoch": selected2, "selected_combined_run_epoch": combined2, "summaries": summaries, "stage_labels": stage_labels, "model_training_run": False, "train_validation_crop_overlap": 0}, overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
