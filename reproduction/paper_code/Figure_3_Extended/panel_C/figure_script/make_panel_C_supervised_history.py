#!/usr/bin/env python3
"""Plot the supervised history of the released production-v2 checkpoint."""
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

EXPECTED_HISTORY = "72dde255d67cf441e39d54042b354f06ae2b410551d8ddf262518a60312e3516"
EXPECTED_SELECTION = "e89cba4983d95acbe8ab981c7502dd19e54abcf0acee8fd04b5191630d4fe4aa"
EXPECTED_CHECKPOINT = "d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    history_path = require_file(args.history, EXPECTED_HISTORY)
    selection_path = require_file(args.selection, EXPECTED_SELECTION)
    checkpoint = require_file(args.checkpoint, EXPECTED_CHECKPOINT)
    history = pd.read_csv(history_path)
    selection = json.loads(selection_path.read_text())
    selected = int(selection["selected_epoch"])
    if int(selection["selected_seed"]) != 44 or len(history) != 20 or selected != 9:
        raise RuntimeError("Released SFT history must be seed 44, 20 completed epochs, selected epoch 9")
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_C_released_supervised_training_history"
    table = out / "panel_C_supervised_training_history.tsv"
    provenance = out / "panel_C_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), table, provenance], args.overwrite)
    specs = (
        ("train_loss", "validation_loss", "Smooth L1 objective"),
        ("train_axis_mae", "validation_axis_mae", "Crypt-villus MAE"),
        ("train_epithelial_mae", "validation_epithelial_mae", "Epithelial-distance MAE"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(8.0, 2.55), sharex=True)
    for axis, (train_key, validation_key, title) in zip(axes, specs, strict=True):
        axis.plot(history["epoch"], history[train_key], color="#333333", lw=1.35, marker="o", ms=2, label="Training")
        axis.plot(history["epoch"], history[validation_key], color="#3575a8", lw=1.45, marker="o", ms=2, label="Validation")
        axis.axvline(selected, color="#9b2f2f", lw=0.9, ls="--")
        selected_value = history.loc[history["epoch"].astype(int).eq(selected), validation_key].iloc[0]
        axis.scatter([selected], [selected_value], color="#9b2f2f", marker="*", s=42, zorder=4)
        axis.set_title(title, fontsize=8.5)
        axis.set_xlabel("Completed epoch")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    figure.suptitle("Coordinate-model fine-tuning", fontsize=9.5, y=0.99)
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.20, top=0.82, wspace=0.35)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    history.to_csv(table, sep="\t", index=False)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(history_path, "released_SFT_history"), file_record(selection_path, "validation_only_selection"), file_record(checkpoint, "released_checkpoint")],
        outputs=[*rendered, table],
        extra={
            "panel": "Extended Figure 3c",
            "completed_epochs": 20,
            "selected_seed": 44,
            "selected_epoch": selected,
            "selection_rule": selection["selection_rule"],
            "test_split_used_for_selection": selection["test_split_used_for_selection"],
            "model_inference_run": False,
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
