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


def equal_limits(ax: plt.Axes, data: pd.DataFrame) -> None:
    span = max(data.x.max() - data.x.min(), data.y.max() - data.y.min(), 1.0)
    pad = span * 0.03
    ax.set_xlim(data.x.min() - pad, data.x.max() + pad)
    ax.set_ylim(data.y.max() + pad, data.y.min() - pad)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def draw_sections(data: pd.DataFrame, column: str, cmap: str, vmax: float, stem: Path, dpi: int, crop: dict) -> list[Path]:
    norm = Normalize(0, vmax)
    fig = plt.figure(figsize=(4.8, 4.6))
    axes = [[fig.add_axes([0.025, 0.06, 0.735, 0.82])]]
    cax = fig.add_axes([0.82, 0.12, 0.035, 0.70])
    for ax, batch, title in zip(axes[0], BATCHES, TITLES, strict=True):
        part = data[data.batch == batch]
        ax.scatter(part.x, part.y, c=part[column], cmap=cmap, norm=norm, s=0.2, linewidths=0, rasterized=True)
        equal_limits(ax, part)
        ax.set_title(title, fontweight="bold")
        if batch == "day90_SI_r2":
            ax.add_patch(Rectangle((crop["x_min"], crop["y_min"]), crop["x_max"] - crop["x_min"], crop["y_max"] - crop["y_min"], fill=False, edgecolor="black", linewidth=1.3))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    return save(fig, stem, dpi)


def draw_zoom(data: pd.DataFrame, column: str, cmap: str, vmax: float, stem: Path, dpi: int, crop: dict) -> list[Path]:
    norm = Normalize(0, vmax)
    fig = plt.figure(figsize=(5.2, 4.0))
    ax = fig.add_axes([0.06, 0.10, 0.68, 0.78])
    cax = fig.add_axes([0.82, 0.12, 0.035, 0.70])
    ax.scatter(data.x, data.y, c=data[column], cmap=cmap, norm=norm, s=1.8, linewidths=0, rasterized=True)
    ax.set_xlim(crop["x_min"], crop["x_max"])
    ax.set_ylim(crop["y_max"], crop["y_min"])
    ax.set_aspect("equal")
    ax.axis("off")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    return save(fig, stem, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 2c dense graph-coordinate maps from frozen predictions.")
    parser.add_argument("--section-cells", type=Path, required=True)
    parser.add_argument("--zoom-cells", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--frozen-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    names = ("panel_C_crypt_villus_day90_whole_section", "panel_C_crypt_villus_day90_zoom", "panel_C_epithelial_distance_day90_whole_section", "panel_C_epithelial_distance_day90_zoom")
    outputs = [out / f"{name}{suffix}" for name in names for suffix in (".pdf", ".png")]
    provenance = out / "panel_C_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)
    manifest = require_graph_inputs(args.input_manifest, [args.section_cells, args.zoom_cells])
    sections = pd.read_parquet(args.section_cells)
    zoom = pd.read_parquet(args.zoom_cells)
    crop = manifest["zoom_limits_um"]
    registry = json.loads(args.frozen_registry.read_text())
    model = registry["models"]["graph_coordinate_all_label_visualization"]
    if model["status"] != "FROZEN_FOR_SPATIAL_VISUALIZATION_NOT_HELD_OUT_EVALUATION":
        raise RuntimeError("The registry does not identify the frozen graph visualization lineage.")
    rendered: list[Path] = []
    rendered += draw_sections(sections, "crypt_villus_prediction", "viridis", 1.0, out / names[0], args.dpi, crop)
    rendered += draw_zoom(zoom, "crypt_villus_prediction", "viridis", 1.0, out / names[1], args.dpi, crop)
    rendered += draw_sections(sections, "epithelial_distance_prediction", "coolwarm", 0.7, out / names[2], args.dpi, crop)
    rendered += draw_zoom(zoom, "epithelial_distance_prediction", "coolwarm", 0.7, out / names[3], args.dpi, crop)
    inputs = [file_record(args.section_cells, "compact_section_cells"), file_record(args.zoom_cells, "zoom_cells"), file_record(args.input_manifest, "compact_input_manifest"), file_record(args.frozen_registry, "frozen_model_registry")]
    write_provenance(provenance, script=HERE, inputs=inputs, outputs=rendered, extra={"panel": "Figure 2c", "rendered_from_numeric_inputs": True, "checkpoint_sha256": [item["sha256"] for item in model["checkpoints"]], "evaluation_scope": "all-label fitted visualization; not a held-out metric"}, overwritten=overwritten)
    print(provenance)


if __name__ == "__main__":
    main()
