#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402
from scaling import load_scaling

COLORS = {"crypt_villus": "#1f77b4", "epithelial_distance": "#d62728"}
LABELS = {"crypt_villus": "Crypt-villus coordinate", "epithelial_distance": "Epithelial-distance coordinate"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source, split_path, frame = load_scaling(args.metrics, args.split_manifest)
    summary = frame.groupby(["axis", "n_train_villi"])["mae"].agg(mean="mean", sem="sem", runs="count").reset_index()
    out = args.output_dir.expanduser().resolve()
    stem, table, provenance = out / "panel_C_heldout_villus_mae", out / "panel_C_heldout_villus_mae_summary.tsv", out / "panel_C_heldout_villus_mae_provenance.json"
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png"), table, provenance]
    overwritten = protect_outputs(outputs, args.overwrite)
    summary.to_csv(table, sep="\t", index=False)
    figure, axis = plt.subplots(figsize=(4.5, 3.5))
    for coordinate in COLORS:
        subset = summary[summary.axis.eq(coordinate)].sort_values("n_train_villi")
        for count, points in frame.loc[frame.axis.eq(coordinate)].groupby('n_train_villi'):
            points = points.sort_values('fold_id')
            axis.scatter(count + np.linspace(-.55, .55, len(points)), points.mae, s=8, alpha=.30, color=COLORS[coordinate], linewidths=0)
        axis.plot(subset.n_train_villi, subset["mean"], marker="o", ms=3.5, lw=1.8, color=COLORS[coordinate], label=LABELS[coordinate])
        axis.fill_between(subset.n_train_villi, subset["mean"] - subset["sem"], subset["mean"] + subset["sem"], color=COLORS[coordinate], alpha=.17, linewidth=0)
    axis.set(xlabel="Training villi", ylabel="Held-out-section MAE")
    axis.legend(frameon=False, fontsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    write_provenance(provenance, script=Path(__file__), inputs=[file_record(source, "eight_fold_section_disjoint_metrics"), file_record(split_path, "locked_split")], outputs=[*rendered, table], extra={"panel": "Extended Figure 2c", "display": "individual held-out sections and mean plus or minus SEM across eight folds", "evaluation_unit": "held-out tissue section", "same_fits_as_main_figure2e": True, "model_refitted": False, "warnings": ["Upstream scVI features were learned jointly across sections."]}, overwritten=overwritten)


if __name__ == "__main__":
    main()
