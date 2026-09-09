#!/usr/bin/env python3
"""Render frozen IF whole-tissue maps with the original coordinate-locked zooms.

This is rendering only: neither predictions nor zoom selection are recomputed.
The historical zoom configuration supplies geometric annotations, not its
legacy class filter. Every cell in each frozen production table is displayed.
"""
from __future__ import annotations

import argparse
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import ConnectionPatch, Rectangle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_4.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402
from Figure_4.release import load_release, release_file
from Figure_4.spatial import common_tissue_limits  # noqa: E402

EXPECTED_DMSO = "141706c9c6f060ec866f2f1063fc236a19e61b994b29c977e407a07257dd618f"
EXPECTED_RARI = "8eb47f93a977727694616dfe75f8808a269f4d3fd1127f1445ac491ed1baf60d"
EXPECTED_ZOOM_CONFIG = "6447d3cc3ea0e4318be13db878fd0a68da0ad4981562721f33a7e885587dce3d"
EXPECTED_ZOOMS = {"DMSO": (45000, 51000, 35000, 39500), "RARi": (17000, 23000, 48000, 52500)}
EXPECTED_PIXEL_SIZE_UM = 0.16247698804244627
SPECS = (
    ("predicted_axis_coordinate", "crypt_villus", "Crypt-villus\ncoordinate", "viridis", 0.0, 1.0),
    ("predicted_epithelial_distance_clipped_1p0", "epithelial_distance", "Epithelial-distance\ncoordinate", "coolwarm", 0.0, 0.5),
)


def load_zooms(path: Path) -> tuple[float, dict[str, tuple[int, int, int, int]]]:
    settings = yaml.safe_load(path.read_text())["imap_if"]
    pixel_size = float(settings["pixel_size_um"])
    if not np.isclose(pixel_size, EXPECTED_PIXEL_SIZE_UM, rtol=0, atol=1e-15):
        raise RuntimeError("Physical pixel size differs from the frozen original zoom configuration")
    zooms = {
        item["id"].upper() if item["id"] == "dmso" else "RARi": tuple(
            int(item[key]) for key in ("x_min_fullres_px", "x_max_fullres_px", "y_min_fullres_px", "y_max_fullres_px")
        )
        for item in settings["spatial_zooms"]
    }
    if zooms != EXPECTED_ZOOMS:
        raise RuntimeError(f"Zoom regions differ from the original locked regions: {zooms}")
    return pixel_size, zooms


def style_map(axis, bounds, *, framed=False):
    xmin, xmax, ymin, ymax = bounds
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymax, ymin)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(framed)
        spine.set_color("#222222")
        spine.set_linewidth(1.1)


def add_external_scale(figure, axis, bounds, length_um, label, *, y):
    """Place the scale in white space outside the point cloud."""
    position = axis.get_position()
    fraction = length_um / (bounds[1] - bounds[0])
    x0 = position.x0 + 0.045 * position.width
    x1 = x0 + fraction * position.width
    figure.add_artist(plt.Line2D([x0, x1], [y, y], transform=figure.transFigure, color="#222222", lw=1.2))
    figure.text((x0 + x1) / 2, y - 0.011, label, fontsize=7.5, ha="center", va="top")


def render_columns(frames, zoom_frames, bounds, zoom_bounds, *, columns, stem, dpi):
    """A row of whole-tissue → magnification pairs, with no region selection."""
    count = len(columns)
    figure = plt.figure(figsize=(1.90 * count + 0.12, 4.40))
    left, right = 0.02, 0.99
    gap = 0.025 if count == 4 else 0.04
    column_width = (right - left - gap * (count - 1)) / count
    whole_axes = []
    for index, (condition, spec) in enumerate(columns):
        column, _, title, cmap, vmin, vmax = spec
        x = left + index * (column_width + gap)
        whole = figure.add_axes((x, 0.49, column_width, 0.435))
        zoom = figure.add_axes((x, 0.165, column_width, 0.300))
        whole_axes.append(whole)
        frame, zoom_frame = frames[condition], zoom_frames[condition]
        whole.scatter(frame.x_um, frame.y_um, c=frame[column], cmap=cmap, vmin=vmin, vmax=vmax,
                      s=0.11, linewidths=0, rasterized=True)
        artist = zoom.scatter(zoom_frame.x_um, zoom_frame.y_um, c=zoom_frame[column],
                              cmap=cmap, vmin=vmin, vmax=vmax, s=1.25, linewidths=0, rasterized=True)
        style_map(whole, bounds[condition])
        style_map(zoom, zoom_bounds[condition], framed=True)
        xmin, xmax, ymin, ymax = zoom_bounds[condition]
        whole.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                                  edgecolor="#222222", linewidth=1.1, zorder=5))
        # Connect the bottom of the selected tissue rectangle to the top of its
        # enlargement. Both ends use the actual data coordinate transforms.
        for xwhole, xzoom in ((xmin, xmin), (xmax, xmax)):
            figure.add_artist(ConnectionPatch(
                xyA=(xwhole, ymax), coordsA=whole.transData,
                xyB=(xzoom, ymin), coordsB=zoom.transData,
                color="#222222", linewidth=0.85, zorder=4, clip_on=False,
            ))
        color_axis = figure.add_axes((x, 0.122, column_width, 0.015))
        colorbar = figure.colorbar(artist, cax=color_axis, orientation="horizontal", ticks=[vmin, vmax])
        colorbar.ax.tick_params(labelsize=7.5, width=0.6, length=2, pad=1.5)
        if spec[1] == "epithelial_distance":
            colorbar.ax.set_xticklabels(["0", "≥0.5"])
        colorbar.outline.set_linewidth(0.6)
        colorbar.set_label(title, fontsize=8, labelpad=1.5)
        if count == 1:
            whole.set_title(condition, fontsize=9, fontweight="bold", pad=9)
    figure.canvas.draw()
    if count == 4:
        for condition, start in (("DMSO", 0), ("RARi", 2)):
            x0, x1 = whole_axes[start].get_position().x0, whole_axes[start + 1].get_position().x1
            figure.text((x0 + x1) / 2, 0.956, condition, ha="center", va="top", fontsize=9, fontweight="bold")
    # All four full maps share the same physical field; a scale per condition
    # also defines the directly connected, coordinate-locked zoom magnification.
    for index, (condition, _) in enumerate(columns):
        if index == 0 or (count == 4 and index == 2):
            add_external_scale(figure, whole_axes[index], bounds[condition], 1000, "1 mm", y=0.958)
    outputs = save_pair(figure, stem, dpi)
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dmso-predictions", type=Path, required=True)
    parser.add_argument("--rari-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--zoom-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    release = load_release(args.release_manifest)
    dmso_path = release_file(args.dmso_predictions, release)
    rari_path = release_file(args.rari_predictions, release)
    checkpoint = release_file(args.checkpoint, release)
    zoom_config = require_file(args.zoom_config, EXPECTED_ZOOM_CONFIG)
    pixel_size, zooms = load_zooms(zoom_config)
    frames = {"DMSO": pd.read_parquet(dmso_path), "RARi": pd.read_parquet(rari_path)}
    if {key: len(value) for key, value in frames.items()} != {"DMSO": 163992, "RARi": 136437}:
        raise RuntimeError("Whole-tissue prediction counts do not match the frozen run")
    zoom_frames, region_records = {}, []
    for condition, frame in frames.items():
        if frame.duplicated(["source_id", "object_id"]).any():
            raise RuntimeError(f"Duplicated frozen prediction identifiers in {condition}")
        for physical, pixels in (("x_um", "centroid_x_fullres_px"), ("y_um", "centroid_y_fullres_px")):
            if not np.allclose(frame[physical], frame[pixels] * pixel_size, rtol=0, atol=1e-8):
                raise RuntimeError(f"Physical-coordinate conversion mismatch in {condition}/{physical}")
        xmin, xmax, ymin, ymax = zooms[condition]
        mask = ((frame.centroid_x_fullres_px >= xmin) & (frame.centroid_x_fullres_px <= xmax)
                & (frame.centroid_y_fullres_px >= ymin) & (frame.centroid_y_fullres_px <= ymax))
        zoom_frames[condition] = frame.loc[mask]
        if zoom_frames[condition].empty:
            raise RuntimeError(f"No cells in the locked {condition} zoom")
        region_records.append({"condition": condition, "x_min_fullres_px": xmin, "x_max_fullres_px": xmax,
                               "y_min_fullres_px": ymin, "y_max_fullres_px": ymax, "pixel_size_um": pixel_size,
                               "width_um": (xmax - xmin) * pixel_size, "height_um": (ymax - ymin) * pixel_size,
                               "whole_tissue_cells": len(frame), "zoom_cells": int(mask.sum())})
    out = args.output_dir.expanduser().resolve()
    columns = [(condition, spec) for condition in ("DMSO", "RARi") for spec in SPECS]
    individual = [out / f"panel_D_{condition.lower()}_{spec[1]}_whole_and_zoom" for condition, spec in columns]
    provenance, regions = out / "panel_D_provenance.json", out / "panel_D_zoom_regions.tsv"
    outputs = [stem.with_suffix(suffix) for stem in individual for suffix in (".pdf", ".png")]
    overwritten = protect_outputs([*outputs, provenance, regions], args.overwrite)
    limits = common_tissue_limits(frames)
    zoom_limits = {condition: tuple(value * pixel_size for value in bound) for condition, bound in zooms.items()}
    rendered = []
    for column, stem in zip(columns, individual, strict=True):
        rendered += render_columns(frames, zoom_frames, limits, zoom_limits, columns=[column], stem=stem, dpi=args.dpi)
    pd.DataFrame(region_records).to_csv(regions, sep="\t", index=False)
    write_provenance(
        provenance, script=Path(__file__),
        inputs=[file_record(dmso_path, "frozen_DMSO_predictions"), file_record(rari_path, "frozen_RARi_predictions"),
                file_record(args.release_manifest, "repaired_IF_release"),
                file_record(checkpoint, "released_IF_checkpoint"), file_record(zoom_config, "original_zoom_geometry_config")],
        outputs=[*rendered, regions],
        extra={"panel": "Figure 4d", "display_downsampling": False,
               "counts": {key: len(value) for key, value in frames.items()},
               "class_filter": None, "legacy_config_class_filter_applied": False,
               "color_limits": {"crypt_villus": [0, 1], "epithelial_distance": [0, 0.5]},
               "colormaps": {"crypt_villus": "viridis", "epithelial_distance": "coolwarm"},
               "matched_physical_field_of_view": True, "whole_tissue_bounds_um": limits,
               "zoom_regions": region_records, "zoom_inclusion_rule": "centroid within inclusive locked full-resolution pixel bounds",
               "zoom_selection": "pre-existing original figure geometry; no prediction-error selection",
               "model_inference_run": False, "seed": None, "warnings": ["Epithelial map colors saturate above 0.5; predictions are not clipped to 0.5."]},
        overwritten=overwritten,
    )
    print(f"Rendered all frozen cells and original zooms: {out}")


if __name__ == "__main__":
    main()
