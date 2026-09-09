#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from common import file_record, protect_outputs, require_graph_inputs, write_provenance  # noqa: E402

BATCHES = ("day6_SI_r2", "day8_SI_Ctrl", "day30_SI_r2", "day90_SI_r2")
TITLES = ("Day 6 SI", "Day 8 SI", "Day 30 SI", "Day 90 SI")
ORDER = ("Epithelial", "Immune", "Stromal", "Neuronal")
COLORS = {"Epithelial": "#4E79A7", "Immune": "#59A14F", "Stromal": "#B07AA1", "Neuronal": "#F28E2B"}


def spatial_limits(ax: plt.Axes, data: pd.DataFrame) -> None:
    span = max(data.x.max() - data.x.min(), data.y.max() - data.y.min(), 1.0)
    pad = span * 0.03
    ax.set_xlim(data.x.min() - pad, data.x.max() + pad)
    ax.set_ylim(data.y.max() + pad, data.y.min() - pad)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_classes(ax: plt.Axes, data: pd.DataFrame, size: float, alpha: float) -> None:
    counts = data.cell_class.value_counts()
    for label in sorted(ORDER, key=lambda value: int(counts.get(value, 0)), reverse=True):
        part = data[data.cell_class == label]
        ax.scatter(part.x, part.y, s=size, c=COLORS[label], alpha=alpha, linewidths=0, rasterized=True)


def save(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 2a data components from compact frozen inputs.")
    parser.add_argument("--section-cells", type=Path, required=True)
    parser.add_argument("--graph-nodes", type=Path, required=True)
    parser.add_argument("--graph-edges", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    stems = [out / "panel_A_section_class_maps", out / "panel_A_scvi_embedding", out / "panel_A_spatial_neighbor_graph"]
    outputs = [stem.with_suffix(suffix) for stem in stems for suffix in (".pdf", ".png")]
    provenance = out / "panel_A_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)

    require_graph_inputs(args.input_manifest, [args.section_cells, args.graph_nodes, args.graph_edges])
    cells = pd.read_parquet(args.section_cells)
    nodes = pd.read_parquet(args.graph_nodes)
    edges = pd.read_csv(args.graph_edges, sep="\t")

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.6), squeeze=False)
    fig.subplots_adjust(wspace=0.02, hspace=0.10, bottom=0.12, top=0.94)
    for ax, batch, title in zip(axes.flat, BATCHES, TITLES, strict=True):
        part = cells[cells.batch == batch]
        draw_classes(ax, part, 0.20, 0.78)
        spatial_limits(ax, part)
        ax.set_title(title, fontsize=12, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=COLORS[x], label=x, markersize=4) for x in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    rendered = save(fig, stems[0], args.dpi)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    embedding = cells[["embedding_x", "embedding_y", "cell_class"]].rename(
        columns={"embedding_x": "x", "embedding_y": "y"}
    )
    draw_classes(ax, embedding, 1.0, 0.55)
    ax.set_xlabel("MDE 1")
    ax.set_ylabel("MDE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("scVI latent-space projection", fontweight="bold")
    ax.legend(handles=handles, title="Class", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    rendered += save(fig, stems[1], args.dpi)

    segments = np.stack((nodes.loc[edges.source, ["x", "y"]].to_numpy(), nodes.loc[edges.target, ["x", "y"]].to_numpy()), axis=1)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.add_collection(LineCollection(segments, colors="#B8B8B8", linewidths=0.28, alpha=0.35, zorder=1))
    ax.scatter(nodes.x, nodes.y, s=30, color="#D8C7A3", linewidths=0, zorder=2)
    focus = nodes[nodes.is_focus].iloc[0]
    focus_neighbors = nodes[nodes.is_focus_neighbor]
    highlight = np.stack((focus_neighbors[["x", "y"]].to_numpy(), np.repeat([[focus.x, focus.y]], len(focus_neighbors), axis=0)), axis=1)
    ax.add_collection(LineCollection(highlight, colors="#B85C4A", linewidths=1.15, alpha=0.85, zorder=3))
    ax.scatter(focus_neighbors.x, focus_neighbors.y, s=34, facecolor="#D8C7A3", edgecolor="#2F6F73", linewidth=1.0, zorder=4)
    ax.scatter(focus.x, focus.y, s=42, facecolor="#D8C7A3", edgecolor="#B85C4A", linewidth=1.5, zorder=5)
    spatial_limits(ax, nodes)
    rendered += save(fig, stems[2], args.dpi)

    inputs = [file_record(args.section_cells, "compact_section_cells"), file_record(args.graph_nodes, "graph_nodes"), file_record(args.graph_edges, "graph_edges"), file_record(args.input_manifest, "compact_input_manifest")]
    write_provenance(provenance, script=HERE, inputs=inputs, outputs=rendered, extra={"panel": "Figure 2a", "rendered_from_numeric_inputs": True}, overwritten=overwritten)
    print(provenance)


if __name__ == "__main__":
    main()
