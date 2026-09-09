#!/usr/bin/env python3
"""Render the frozen held-out Xenium Peyer's patch prediction window."""
from __future__ import annotations

import argparse
import json
import math
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_3_Extended.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402

EXPECTED_IMAGE = "1be53f9fe3ec395b03e787f324d90ad2fce6d9dbd4d4e1818e8ddfb541508970"


def read_tiled_tiff_crop(
    image_path: Path,
    bounds_um: tuple[float, float, float, float],
    pixel_size_um: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    import tifffile

    xmin_um, xmax_um, ymin_um, ymax_um = bounds_um
    with tifffile.TiffFile(image_path) as tif:
        page = tif.pages[0]
        image_height, image_width = int(page.shape[-2]), int(page.shape[-1])
        tile_width = int(page.tilewidth or image_width)
        tile_height = int(page.tilelength or image_height)
        x0 = max(0, int(math.floor(xmin_um / pixel_size_um)))
        x1 = min(image_width, int(math.ceil(xmax_um / pixel_size_um)))
        y0 = max(0, int(math.floor(ymin_um / pixel_size_um)))
        y1 = min(image_height, int(math.ceil(ymax_um / pixel_size_um)))
        crop = np.zeros((y1 - y0, x1 - x0), dtype=page.dtype)
        tiles_across = int(math.ceil(image_width / tile_width))
        for tile_y in range(y0 // tile_height, (y1 - 1) // tile_height + 1):
            for tile_x in range(x0 // tile_width, (x1 - 1) // tile_width + 1):
                tile_index = tile_y * tiles_across + tile_x
                tif.filehandle.seek(page.dataoffsets[tile_index])
                encoded = tif.filehandle.read(page.databytecounts[tile_index])
                decoded, indices, _ = page.decode(encoded, tile_index)
                if decoded is None:
                    continue
                tile = np.squeeze(np.asarray(decoded))
                if tile.ndim != 2:
                    tile = tile.reshape(tile.shape[-2], tile.shape[-1])
                origin_y, origin_x = int(indices[2]), int(indices[3])
                end_y = min(origin_y + tile.shape[0], image_height)
                end_x = min(origin_x + tile.shape[1], image_width)
                oy0, oy1 = max(y0, origin_y), min(y1, end_y)
                ox0, ox1 = max(x0, origin_x), min(x1, end_x)
                if oy1 > oy0 and ox1 > ox0:
                    crop[oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0] = tile[
                        oy0 - origin_y : oy1 - origin_y, ox0 - origin_x : ox1 - origin_x
                    ]
    return crop, (x0 * pixel_size_um, x1 * pixel_size_um, y0 * pixel_size_um, y1 * pixel_size_um)


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image) & (image > 0)]
    low, high = np.percentile(values, [0.5, 99.8])
    return np.clip((image.astype(np.float32) - low) / max(high - low, 1e-12), 0, 1)


def style(axis: plt.Axes, bounds: tuple[float, float, float, float], title: str) -> None:
    x0, x1, y0, y1 = bounds
    axis.set_xlim(x0, x1)
    axis.set_ylim(y1, y0)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=8.5)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def scale_bar(axis: plt.Axes, bounds: tuple[float, float, float, float], color: str, outline: str) -> None:
    x0, x1, y0, y1 = bounds
    width, height = x1 - x0, y1 - y0
    start_x, y = x0 + 0.06 * width, y1 - 0.08 * height
    line = axis.plot([start_x, start_x + 100], [y, y], color=color, lw=2.0, solid_capstyle="butt")[0]
    line.set_path_effects([pe.Stroke(linewidth=3.5, foreground=outline), pe.Normal()])
    label = axis.text(start_x + 50, y - 0.035 * height, "100 µm", color=color, fontsize=7, ha="center", va="bottom")
    label.set_path_effects([pe.Stroke(linewidth=2.3, foreground=outline), pe.Normal()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold-manifest", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from frozen_preprocessing.peyer_representation_v1.release import load_release, RELEASE_POINTER
    active_path, active = load_release()
    threshold_manifest = require_file(args.threshold_manifest, active["window_release"]["sha256"])
    repaired = json.loads(threshold_manifest.read_text())
    if repaired['status'] != 'FROZEN_REPRESENTATION_CLASSIFIER_WINDOWS':
        raise RuntimeError('Panel G requires the representation-pretrained classifier window release')
    require_file(repaired['label_lock']['path'], repaired['label_lock']['sha256'])
    predictions_path = require_file(args.predictions, repaired["outputs"]["display_window_fixed_threshold.csv"]["sha256"])
    summary_path = require_file(args.summary, repaired["outputs"]["fixed_threshold_window_summary.json"]["sha256"])
    image_path = require_file(args.image, EXPECTED_IMAGE)
    checkpoint = require_file(args.checkpoint, repaired['checkpoint']['sha256'])
    predictions = pd.read_csv(predictions_path)
    summary = json.loads(summary_path.read_text())
    if len(predictions) != 3373 or set(predictions["batch"].astype(str)) != {"day90_SI_r2"}:
        raise RuntimeError("Panel G requires the frozen 3,373-cell day90_SI_r2 display window")
    checkpoint_from_summary = Path(summary["dense_prediction_summary"]["checkpoint_path"].replace("/Figure5/", "/Peyers_Pipeline/"))
    if checkpoint_from_summary.resolve() != checkpoint.resolve():
        raise RuntimeError("Prediction summary checkpoint does not match supplied Peyer's image classifier")
    requested = summary["requested_bounds_um"]
    requested_bounds = (float(requested["x_min"]), float(requested["x_max"]), float(requested["y_min"]), float(requested["y_max"]))
    image, bounds = read_tiled_tiff_crop(image_path, requested_bounds, float(summary["image"]["pixel_size_um"]))
    threshold = float(summary["classification_threshold"])
    if threshold != 0.5:
        raise RuntimeError("The released image-classifier display uses the fixed 0.5 threshold")
    positive = predictions["peyer_manual_label"].astype(int).eq(1)
    called = predictions["peyer_probability"].astype(float).ge(threshold)
    if not np.array_equal(called.astype(int), predictions["hard_class_fixed_0p5"].astype(int)):
        raise RuntimeError("Saved hard calls differ from the fixed threshold")

    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_G_heldout_peyer_prediction"
    cells = out / "panel_G_display_cells.tsv"
    provenance = out / "panel_G_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), cells, provenance], args.overwrite)
    figure, axes = plt.subplots(2, 2, figsize=(6.0, 5.6))
    axes[0, 0].imshow(contrast_stretch(image), cmap="gray", extent=(bounds[0], bounds[1], bounds[3], bounds[2]), interpolation="nearest")
    style(axes[0, 0], bounds, "Xenium DAPI morphology")
    scale_bar(axes[0, 0], bounds, "white", "black")
    axes[0, 1].scatter(predictions.loc[~positive, "x_spatial"], predictions.loc[~positive, "y_spatial"], s=6, color="#c8c8c8", linewidths=0, rasterized=True)
    axes[0, 1].scatter(predictions.loc[positive, "x_spatial"], predictions.loc[positive, "y_spatial"], s=12, color="#c94a44", linewidths=0, rasterized=True)
    style(axes[0, 1], bounds, "Manual Peyer's patch label")
    scatter = axes[1, 0].scatter(predictions["x_spatial"], predictions["y_spatial"], c=predictions["peyer_probability"], s=10, cmap="magma", vmin=0, vmax=1, linewidths=0, rasterized=True)
    style(axes[1, 0], bounds, "Image-classifier probability")
    colorbar = figure.colorbar(scatter, ax=axes[1, 0], fraction=0.046, pad=0.02)
    colorbar.set_label("Peyer's patch probability", fontsize=7)
    axes[1, 1].scatter(predictions.loc[~called, "x_spatial"], predictions.loc[~called, "y_spatial"], s=7, color="#c8c8c8", linewidths=0, rasterized=True)
    axes[1, 1].scatter(predictions.loc[called, "x_spatial"], predictions.loc[called, "y_spatial"], s=10, color="#8f3b76", linewidths=0, rasterized=True)
    style(axes[1, 1], bounds, f"Predicted class · threshold {threshold:.2f}")
    for axis in axes.ravel()[1:]:
        scale_bar(axis, bounds, "#222222", "white")
    figure.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.96, hspace=0.12, wspace=0.10)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    predictions.to_csv(cells, sep="\t", index=False)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[
            file_record(predictions_path, "frozen_dense_window_predictions"),
            file_record(threshold_manifest, "fixed_threshold_release"),
            file_record(summary_path, "frozen_display_window_summary"),
            file_record(image_path, "Xenium_morphology_image"),
            file_record(checkpoint, "selected_peyer_image_classifier_checkpoint"),
            file_record(RELEASE_POINTER, "active_classifier_release_pointer"),
            file_record(active_path, "representation_classifier_release"),
        ],
        outputs=[*rendered, cells],
        extra={
            "panel": "Extended Figure 3g",
            "component_order": ["DAPI_top_left", "manual_label_top_right", "probability_bottom_left", "hard_class_bottom_right"],
            "former_panel": "Extended Figure 3h",
            "lineage": "representation-pretrained axis-architecture Peyer classifier",
            "section": "day90_SI_r2",
            "display_cells": len(predictions),
            "manual_positive_cells": int(positive.sum()),
            "predicted_positive_cells": int(called.sum()),
            "classification_threshold": threshold,
            "model_training_or_inference_run": False,
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
