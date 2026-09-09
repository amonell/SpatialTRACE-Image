#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, load_prepared_manifest, protect_outputs, require_file, save_pair, write_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path, manifest = load_prepared_manifest(args.prepared_manifest)
    history_path = require_file(args.history, manifest["prepared_outputs"]["panel_b_history"]["sha256"])
    history = pd.read_csv(history_path, sep="\t")
    best_epoch = int(manifest["model_architecture"]["selected_epoch"])
    completed = int(manifest["model_architecture"].get("completed_epochs", 18))
    if history["epoch"].astype(int).tolist() != list(range(1, completed + 1)) or not 1 <= best_epoch <= completed:
        raise RuntimeError("Frozen history or selected epoch changed")

    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_B_peyer_gat_training_history"
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    provenance = out / "panel_B_peyer_gat_training_history_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3), constrained_layout=True)
    specs = [
        ("loss", "Loss", None),
        ("average_precision", "Average precision", (0, 1)),
        ("auroc", "AUROC", (0, 1)),
        ("f1", "F1 at 0.5", (0, 1)),
    ]
    for ax, (metric, title, ylim) in zip(axes.reshape(-1), specs, strict=True):
        plot_rows = history
        ax.plot(plot_rows["epoch"], plot_rows[f"train_{metric}"], color="#909090", linewidth=1.8, label="train")
        ax.plot(plot_rows["epoch"], plot_rows[f"validation_{metric}"], color="#A75D50", linewidth=1.8, label="validation")
        ax.axvline(best_epoch, color="#333333", linestyle=":", linewidth=1.1)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Completed epoch")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save_pair(fig, stem)
    plt.close(fig)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(history_path, "frozen_training_history"), file_record(manifest_path, "prepared_input_manifest")],
        outputs=outputs,
        overwritten=overwritten,
        extra={"panel": "B", "selected_epoch": best_epoch, "selection_metric": "validation average precision", "history_epochs": int(len(history)), "hard_call_threshold": manifest["model_architecture"]["hard_call_threshold"]},
    )
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
