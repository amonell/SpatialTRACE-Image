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


SECTIONS = ["day6_SI", "day8_SI_Ctrl", "day30_SI", "day90_SI"]
TITLES = ["Day 6 SI rep1", "Day 8 SI rep1", "Day 30 SI rep1", "Day 90 SI rep1"]


def style_spatial(ax, title: str) -> None:
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--training-seeds", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path, manifest = load_prepared_manifest(args.prepared_manifest)
    background_path = require_file(args.background, manifest["prepared_outputs"]["panel_a_background"]["sha256"])
    seeds_path = require_file(args.training_seeds, manifest["prepared_outputs"]["panel_a_training_seeds"]["sha256"])
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_A_peyer_training_seeds"
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    provenance = out / "panel_A_peyer_training_seeds_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)

    background = pd.read_parquet(background_path)
    seeds = pd.read_parquet(seeds_path)
    if sorted(background["batch"].unique()) != sorted(SECTIONS):
        raise RuntimeError("Panel A background sections do not match the frozen training sections")
    if not set(seeds["batch"].unique()).issubset(SECTIONS):
        raise RuntimeError("Panel A seeds contain a non-training section")

    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.65), constrained_layout=True)
    legend_handles = None
    for ax, section, title in zip(axes, SECTIONS, TITLES, strict=True):
        bg = background.loc[background["batch"].astype(str).eq(section)]
        ss = seeds.loc[seeds["batch"].astype(str).eq(section)]
        ax.scatter(bg["x_spatial"], bg["y_spatial"], s=0.12, color="#D7D7D7", alpha=0.58, linewidths=0, rasterized=True)
        negative = ss["peyer_manual_label"].astype(int).eq(0)
        hneg = ax.scatter(ss.loc[negative, "x_spatial"], ss.loc[negative, "y_spatial"], s=1.4, color="#C9C9C9", alpha=0.9, linewidths=0, rasterized=True, label="negative seed")
        hpos = ax.scatter(ss.loc[~negative, "x_spatial"], ss.loc[~negative, "y_spatial"], s=4.0, color="#7652A7", alpha=0.98, linewidths=0, rasterized=True, label="Peyer seed")
        if legend_handles is None:
            legend_handles = [hneg, hpos]
        style_spatial(ax, title)
    fig.legend(legend_handles, ["negative seed", "Peyer seed"], frameon=False, loc="lower center", ncol=2, markerscale=3.0, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    save_pair(fig, stem)
    plt.close(fig)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(background_path, "prepared_section_backgrounds"), file_record(seeds_path, "exact_GAT_training_seeds"), file_record(manifest_path, "prepared_input_manifest")],
        outputs=outputs,
        overwritten=overwritten,
        extra={"panel": "A", "sections": SECTIONS, "training_seed_count": int(len(seeds)), "model_checkpoint_sha256": manifest.get("checkpoint", manifest["source_inputs"][0])["sha256"]},
    )
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
