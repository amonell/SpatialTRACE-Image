#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from common import file_record, protect_outputs, write_provenance, require_frozen_input, load_source_regions  # noqa: E402


def save(fig: plt.Figure, stem: Path, dpi: int, pad: float = 0.02) -> list[Path]:
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    return [pdf, png]


def spatial(ax: plt.Axes, data: pd.DataFrame, color, size: float, norm=None, cmap=None) -> None:
    ax.scatter(data.x, data.y, c=color, s=size, norm=norm, cmap=cmap, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the data/image components used in Figure 1a.")
    parser.add_argument("--section-cells", type=Path, required=True)
    parser.add_argument("--zoom-cells", type=Path, required=True)
    parser.add_argument("--peyer-dapi", type=Path, required=True)
    parser.add_argument("--figure1-input-manifest", type=Path, required=True)
    parser.add_argument("--source-region-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    names = ("panel_A_day90_whole_tissue", "panel_A_day90_gray_villus_zoom", "panel_A_day90_peyer_dapi")
    outputs = [out / f"{name}{suffix}" for name in names for suffix in (".pdf", ".png")]
    provenance = out / "panel_A_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)
    regions = load_source_regions(args.source_region_lock)
    sections = pd.read_parquet(require_frozen_input(args.section_cells))
    day90 = sections[sections.batch == "day90_SI_r2"]
    zoom = pd.read_parquet(require_frozen_input(args.zoom_cells))
    dapi = np.load(require_frozen_input(args.peyer_dapi))
    rendered: list[Path] = []
    fig, ax = plt.subplots(figsize=(3.6, 3.6)); spatial(ax, day90, "#b8b8b8", 0.18); rendered += save(fig, out / names[0], args.dpi)
    fig, ax = plt.subplots(figsize=(4.0, 3.4)); spatial(ax, zoom, "#b8b8b8", 2.1)
    x0, x1, y0, y1 = regions["regions"]["villus_zoom"]["limits_um_xyxy"]
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    rendered += save(fig, out / names[1], args.dpi)
    low, high = np.percentile(dapi[dapi > 0], [0.5, 99.8]); fig, ax = plt.subplots(figsize=(4.0, 4.0)); ax.imshow(dapi, cmap="gray", vmin=low, vmax=high); ax.axis("off"); rendered += save(fig, out / names[2], args.dpi, 0)
    inputs = [file_record(args.section_cells, "compact_section_cells"), file_record(args.zoom_cells, "zoom_cells"), file_record(args.peyer_dapi, "raw_dapi_crop"), file_record(args.figure1_input_manifest, "figure1_input_manifest")]
    inputs.append(file_record(args.source_region_lock, "artwork_registered_source_regions"))
    write_provenance(provenance, script=HERE, inputs=inputs, outputs=rendered, extra={"panel": "Figure 1a", "rendered_from_numeric_or_raw_image_inputs": True, "villus_zoom_limits_um_xyxy": [x0,x1,y0,y1]}, overwritten=overwritten)
    print(provenance)


if __name__ == "__main__":
    main()
