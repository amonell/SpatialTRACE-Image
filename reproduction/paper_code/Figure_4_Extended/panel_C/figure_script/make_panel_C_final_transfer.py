#!/usr/bin/env python3
"""Render the frozen final-lineage zero-shot versus IF-fine-tuned comparison."""
from __future__ import annotations

import argparse
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_4_Extended.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402
from Figure_4.release import load_release, release_file

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--xenium-checkpoint", type=Path, required=True)
    parser.add_argument("--if-checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    release = load_release(args.release_manifest)
    metrics_path = release_file(args.metrics, release)
    bootstrap_path = release_file(args.bootstrap, release)
    xenium = require_file(args.xenium_checkpoint, "d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137")
    if_checkpoint = release_file(args.if_checkpoint, release)
    split = require_file(args.split_manifest, "7d9c9120c5ef55a2c30a2fe2b3c0e0fe3b16a111ca61ff5339029ae6e4a2ac48")
    data = pd.read_csv(metrics_path, sep="\t")
    if len(data) != 4 or set(data.n.astype(int)) != {4843}:
        raise RuntimeError("Expected two models × two coordinates on n=4,843")
    model_order = ["zero_shot", "fine_tuned"]
    coordinate_order = ["Crypt-villus", "Epithelial distance"]
    model_labels = ["Zero-shot", "IF fine-tuned"]
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_C_final_lineage_transfer"
    table = out / "panel_C_transfer_metrics.tsv"
    svg = stem.with_suffix(".svg")
    provenance = out / "panel_C_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), svg, table, provenance], args.overwrite)
    matrix = data.pivot(index="model_key", columns="coordinate", values="spearman_rho").reindex(index=model_order, columns=coordinate_order).to_numpy(float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise RuntimeError("Missing/invalid frozen Spearman values")
    if not ((matrix >= -1) & (matrix <= 1)).all():
        raise RuntimeError("Spearman values fall outside the prespecified display scale")
    # Each tile is an ordinary opaque RGB vector rectangle. No indexed-color
    # image, gradient/colorbar bitmap, soft mask, or transparent overlay is used.
    figure = plt.figure(figsize=(3.25, 1.95), facecolor="white")
    axis = figure.add_axes([0.31, 0.29, 0.67, 0.49], facecolor="white")
    # Opaque zero-width background edges avoid even unused alpha-zero stroke
    # settings in PDF readers/importers that mishandle graphics-state reuse.
    figure.patch.set_edgecolor("white")
    figure.patch.set_linewidth(0)
    axis.patch.set_edgecolor("white")
    axis.patch.set_linewidth(0)
    palette = []
    colormap, normalization = plt.get_cmap("Blues"), Normalize(-1, 1)
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            color = to_hex(colormap(normalization(value)), keep_alpha=False)
            palette.append({"model_key": model_order[row], "coordinate": coordinate_order[column],
                            "spearman_rho": float(value), "rgb_hex": color, "alpha": 1.0})
            axis.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1,
                                     facecolor=color, edgecolor="white", linewidth=0.8, alpha=1.0))
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=10, fontweight="bold", color="white" if value > 0.62 else "#202020")
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(1.5, -0.5)
    axis.set_xticks(range(2), ["Crypt-villus", "Epithelial\ndistance"])
    axis.set_yticks(range(2), model_labels)
    axis.tick_params(length=0, labelsize=8.5, pad=5)
    axis.spines[:].set_visible(False)
    figure.text(0.03, 0.92, "IF transfer · Spearman ρ", fontsize=9, fontweight="bold", ha="left", va="top")
    rendered = save_pair(figure, stem, args.dpi)
    # Illustrator can also open this direct-RGB vector source. Text remains
    # editable Arial rather than being converted into hundreds of glyph paths.
    with plt.rc_context({"svg.fonttype": "none"}):
        figure.savefig(svg, bbox_inches="tight", pad_inches=0.02, transparent=False)
    rendered.append(svg)
    plt.close(figure)
    data.to_csv(table, sep="\t", index=False)
    write_provenance(
        provenance, script=Path(__file__),
        inputs=[file_record(args.release_manifest, "repaired_IF_release"), file_record(metrics_path, "frozen_transfer_metrics"), file_record(bootstrap_path, "frozen_grouped_bootstrap"), file_record(xenium, "released_Xenium_checkpoint"), file_record(if_checkpoint, "released_IF_checkpoint"), file_record(split, "locked_strict_IF_split")],
        outputs=[*rendered, table],
        extra={"panel": "Extended Figure 4c", "displayed_metric": "Spearman rho", "R2_displayed": False,
               "n": 4843, "model_inference_run": False, "model_selection_run": False,
               "uncertainty_available_in_input_table": str(bootstrap_path), "random_seed": None,
               "plot_type": "2x2 opaque RGB vector tile table", "display_model_labels": model_labels,
               "model_label_definitions": {"Zero-shot": "Released Xenium representation-pretrained model applied to IF without adaptation",
                                           "IF fine-tuned": "Released IF-adapted checkpoint initialized from the same Xenium model"},
               "evaluation_scope": "Same 4843 train- and validation-crop-disjoint IF cells for both models",
               "colormap": "Blues", "color_limits": [-1, 1], "tile_palette": palette,
               "pdf_raster_images": False, "indexed_color_images": False, "transparency": False,
               "svg_text": "editable Arial", "png_dpi": args.dpi,
               "warnings": ["Use an RGB Illustrator document to retain the specified RGB colors; conversion into a CMYK document can change appearance."]},
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
