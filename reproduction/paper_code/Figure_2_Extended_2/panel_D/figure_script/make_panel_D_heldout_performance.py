#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, load_prepared_manifest, protect_outputs, require_file, save_pair, write_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path, manifest = load_prepared_manifest(args.prepared_manifest)
    metrics_path = require_file(args.metrics, manifest["prepared_outputs"]["panel_d_section_metrics"]["sha256"])
    metrics = pd.read_csv(metrics_path, sep="\t")
    expected_sections = ["day6_SI_r2", "day8_SI_r2", "day30_SI_r2", "day90_SI_r2", "test pooled"]
    if metrics["section"].astype(str).tolist() != expected_sections:
        raise RuntimeError("Panel D section order changed")
    threshold = float(manifest["model_architecture"]["hard_call_threshold"])
    if not np.allclose(metrics["threshold"], threshold):
        raise RuntimeError("Panel D metrics do not use the frozen threshold")

    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_D_peyer_gat_heldout_performance"
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    table = out / "panel_D_peyer_gat_heldout_performance.tsv"
    provenance = out / "panel_D_peyer_gat_heldout_performance_provenance.json"
    overwritten = protect_outputs(outputs + [table, provenance], args.overwrite)
    metrics.to_csv(table, sep="\t", index=False)
    metrics = metrics.loc[metrics.split.eq("test")].copy()
    if metrics.section.tolist() != ["day30_SI_r2", "day90_SI_r2", "test pooled"] or int(metrics.iloc[-1].n) != 500050:
        raise RuntimeError("Test-only summary membership changed")
    colors = ["#4F7F8A", "#4F7F8A", "#555555"]
    labels = ["Day 30 SI\nreplicate 2", "Day 90 SI\nreplicate 2", "Test sections\npooled"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(4.4, 3.25), constrained_layout=True)
    bars = ax.bar(x, metrics["balanced_accuracy"], color=colors, edgecolor="#333333", linewidth=0.45, width=0.66)
    for bar, balanced, accuracy in zip(bars, metrics["balanced_accuracy"], metrics["accuracy"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, float(balanced) + 0.018, f"{balanced:.2f}\nacc {accuracy:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("GAT test-section classification performance", fontsize=9, fontweight="bold")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.22)
    save_pair(fig, stem)
    plt.close(fig)
    outputs.append(table)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(metrics_path, "recomputed_section_metrics"), file_record(manifest_path, "prepared_input_manifest")],
        outputs=outputs,
        overwritten=overwritten,
        extra={"panel": "D", "metric": "balanced_accuracy", "annotation_secondary_metric": "total_accuracy", "hard_call_threshold": threshold, "validation_sections_displayed": 0, "test_sections": 2, "test_cells": 500050, "validation_retained_separately_in_source_table": True, "limitation": "Upstream scVI representation was jointly learned across sections"},
    )
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
