#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from common import file_record, protect_outputs, require_graph_inputs, write_provenance  # noqa: E402

BATCHES = ("day90_SI_r2",)
TITLES = ("Day 90 SI",)
ZOOM_VILLI = ("day90_SI_r2__villus_006", "day90_SI_r2__villus_004")


def equal_limits(ax: plt.Axes, data: pd.DataFrame) -> None:
    span = max(data.x.max() - data.x.min(), data.y.max() - data.y.min(), 1.0)
    pad = span * 0.03
    ax.set_xlim(data.x.min() - pad, data.x.max() + pad)
    ax.set_ylim(data.y.max() + pad, data.y.min() - pad)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[1], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return paths


def draw_all(sections: pd.DataFrame, labels: pd.DataFrame, value: str, cmap: str, vmax: float, colorbar: str, stem: Path, dpi: int, limits: dict) -> list[Path]:
    norm = Normalize(0, vmax)
    fig = plt.figure(figsize=(4.8, 4.6))
    axes = [[fig.add_axes([0.025, 0.06, 0.735, 0.82])]]
    cax = fig.add_axes([0.82, 0.12, 0.035, 0.70])
    for ax, batch, title in zip(axes[0], BATCHES, TITLES, strict=True):
        background = sections[sections.batch == batch]
        annotated = labels[labels.batch.astype(str) == batch]
        ax.scatter(background.x, background.y, c="#d0d0d0", s=0.12, linewidths=0, rasterized=True)
        ax.scatter(annotated.x, annotated.y, c=annotated[value], cmap=cmap, norm=norm, s=2.0, linewidths=0, rasterized=True)
        equal_limits(ax, background)
        if batch == "day90_SI_r2":
            ax.add_patch(Rectangle((limits["x_min"], limits["y_min"]), limits["x_max"] - limits["x_min"], limits["y_max"] - limits["y_min"], fill=False, edgecolor="black", linewidth=1.3))
        ax.set_title(title, fontweight="bold")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label(colorbar)
    return save(fig, stem, dpi)


def draw_zoom(zoom: pd.DataFrame, labels: pd.DataFrame, value: str, cmap: str, vmax: float, colorbar: str, stem: Path, dpi: int, limits: dict) -> list[Path]:
    annotated = labels[(labels.batch.astype(str) == "day90_SI_r2") & labels.training_villus_id.astype(str).isin(ZOOM_VILLI)]
    fig = plt.figure(figsize=(5.2, 4.0))
    ax = fig.add_axes([0.06, 0.10, 0.68, 0.78])
    cax = fig.add_axes([0.82, 0.12, 0.035, 0.70])
    ax.scatter(zoom.x, zoom.y, c="#b8b8b8", s=1.0, linewidths=0, rasterized=True)
    norm = Normalize(0, vmax)
    ax.scatter(annotated.x, annotated.y, c=annotated[value], cmap=cmap, norm=norm, s=8, linewidths=0, rasterized=True)
    ax.set_xlim(limits["x_min"], limits["x_max"])
    ax.set_ylim(limits["y_max"], limits["y_min"])
    ax.set_aspect("equal")
    ax.axis("off")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label(colorbar)
    return save(fig, stem, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 2b sparse-label components from compact inputs.")
    parser.add_argument("--section-cells", type=Path, required=True)
    parser.add_argument("--sparse-labels", type=Path, required=True)
    parser.add_argument("--zoom-cells", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    names = ("panel_B_crypt_villus_day90_whole_section", "panel_B_crypt_villus_day90_zoom", "panel_B_epithelial_distance_day90_whole_section", "panel_B_epithelial_distance_day90_zoom")
    outputs = [out / f"{name}{suffix}" for name in names for suffix in (".pdf", ".png")]
    provenance = out / "panel_B_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)
    manifest = require_graph_inputs(args.input_manifest, [args.section_cells, args.sparse_labels, args.zoom_cells])
    sections = pd.read_parquet(args.section_cells)
    labels = pd.read_parquet(args.sparse_labels)
    zoom = pd.read_parquet(args.zoom_cells)
    limits = manifest["zoom_limits_um"]
    rendered: list[Path] = []
    rendered += draw_all(sections, labels, "crypt_villus_label", "viridis", 1.0, "Crypt-villus label", out / names[0], args.dpi, limits)
    rendered += draw_zoom(zoom, labels, "crypt_villus_label", "viridis", 1.0, "Crypt-villus label", out / names[1], args.dpi, limits)
    rendered += draw_all(sections, labels, "epithelial_distance_label", "coolwarm", 0.7, "Epithelial-distance label", out / names[2], args.dpi, limits)
    rendered += draw_zoom(zoom, labels, "epithelial_distance_label", "coolwarm", 0.7, "Epithelial-distance label", out / names[3], args.dpi, limits)
    inputs = [file_record(args.section_cells, "compact_section_cells"), file_record(args.sparse_labels, "sparse_labels"), file_record(args.zoom_cells, "zoom_cells"), file_record(args.input_manifest, "compact_input_manifest")]
    write_provenance(provenance, script=HERE, inputs=inputs, outputs=rendered, extra={"panel": "Figure 2b", "rendered_from_numeric_inputs": True}, overwritten=overwritten)
    print(provenance)


if __name__ == "__main__":
    main()
