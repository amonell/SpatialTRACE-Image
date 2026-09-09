#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402

COLORS = {"crypt_villus": "#1f77b4", "epithelial_distance": "#d62728"}
LABELS = {"crypt_villus": "Crypt-villus coordinate", "epithelial_distance": "Epithelial-distance coordinate"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-records", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest_path = require_file(args.benchmark_manifest, "ff9c26154f9422e7063a008e32940c074aafdf36631e7ac7aed8599d0db91de8")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("runs") != 160:
        raise RuntimeError("Runtime benchmark is incomplete")
    if manifest["split"]["sha256"] != "e1779c3caab90164eea05ba38d2dc93c3432ad845711fd7428f281bd52eaa0f8":
        raise RuntimeError("Runtime benchmark split does not match main Figure 2e")
    source = require_file(args.runtime_records, manifest["records"]["sha256"])
    frame = pd.read_csv(source, sep="\t")
    if len(frame) != 160 or frame.duplicated(["axis", "n_train_villi", "fold_id"]).any() or not frame.groupby(["axis", "n_train_villi"]).size().eq(8).all():
        raise RuntimeError("Expected eight section folds at each coordinate and villus count")
    summary = frame.groupby(["axis", "n_train_villi"])["fit_seconds"].agg(median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75), runs="count").reset_index()
    out = args.output_dir.expanduser().resolve()
    pdf, png, table, provenance = out / "panel_B_training_runtime.pdf", out / "panel_B_training_runtime.png", out / "panel_B_training_runtime_summary.tsv", out / "panel_B_training_runtime_provenance.json"
    overwritten = protect_outputs([pdf, png, table, provenance], args.overwrite)
    summary.to_csv(table, sep="\t", index=False)
    figure, axis = plt.subplots(figsize=(4.5, 3.5))
    for coordinate in COLORS:
        subset = summary[summary.axis.eq(coordinate)].sort_values("n_train_villi")
        axis.plot(subset.n_train_villi, subset["median"], marker="o", ms=3.5, lw=1.8, color=COLORS[coordinate], label=LABELS[coordinate])
        axis.fill_between(subset.n_train_villi, subset.q1, subset.q3, color=COLORS[coordinate], alpha=.17, linewidth=0)
    axis.set(xlabel="Training villi", ylabel="Fit time including validation (s)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    rendered = save_pair(figure, pdf.with_suffix(""), args.dpi)
    plt.close(figure)
    write_provenance(provenance, script=Path(__file__), inputs=[file_record(source, "per_run_runtime_records"), file_record(manifest_path, "matched_runtime_benchmark")], outputs=[*rendered, table], extra={"panel": "Extended Figure 2b", "display": "median and interquartile range across eight section folds", "scientific_checkpoints_replaced": False, "endpoint": manifest["fitting_endpoint"], "hardware": frame.gpu.unique().tolist()}, overwritten=overwritten)


if __name__ == "__main__":
    main()
