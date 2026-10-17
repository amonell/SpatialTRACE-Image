#!/usr/bin/env python3
"""Render the prespecified three-villus sparse IF training-label example."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_4.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402
from Figure_4.release import load_release, release_file

EXPECTED_MANIFEST = "72087f80815a2266d0d7439544ebe3daa6e9bed8225d0083e3e355e1718c7a72"
EXPECTED_IMAGE = "0451386a72b456d622e3f2f595d13d6e7cd5bdea830743b034894be8afb5f6ec"
EXPECTED_CALIBRATION_CONFIG = "2aed259a872b4319183c9a52e67727f46d0f5381e9083906e6c5624b231629e6"
SELECTED_VILLI = ("SI1_villus_009", "SI1_villus_010", "SI1_villus_031")
PIXEL_SIZE_UM = 0.16247698804244627
SCALE_BAR_UM = 200.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--example-manifest", type=Path, required=True)
    parser.add_argument("--calibration-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    release = load_release(args.release_manifest)
    manifest_path = release_file(args.manifest, release)
    example_path = require_file(args.example_manifest, "7bb07db00ed5be33d127420f80f9c914f1925ff5dbc24246ef0bf3d29353f8a2")
    example = json.loads(example_path.read_text())
    if example["training_manifest"]["sha256"] != file_record(manifest_path, "training")["sha256"] or example["selection_uses_model_predictions"]:
        raise RuntimeError("Training example does not match the repaired, prespecified training release")
    image_path = require_file(args.image_cache, EXPECTED_IMAGE)
    checkpoint = release_file(args.checkpoint, release)
    calibration_path = release_file(args.calibration_config, release)
    calibration = yaml.safe_load(calibration_path.read_text())
    pixel_size_um = float(calibration["inputs"]["source_pixel_size_um"])
    if not np.isclose(pixel_size_um, PIXEL_SIZE_UM, rtol=0, atol=1e-15):
        raise RuntimeError("Frozen IF source-pixel calibration changed")
    manifest = pd.read_csv(manifest_path)
    selected_villi = example["selected_villi"]
    rows = manifest.loc[manifest.villus_shape_id.astype(str).isin(selected_villi)].copy()
    if set(rows.villus_shape_id.astype(str)) != set(selected_villi) or not rows.split.astype(str).eq("train").all() or len(rows) != example["selected_cells"]:
        raise RuntimeError("Prespecified sparse-label example is not the locked three training villi")
    x0, x1, y0, y1 = example["windows"]["sparse_training"]["bounds_xxyy_px"]
    image = np.load(image_path, mmap_mode="r")
    crop = np.asarray(image[y0:y1, x0:x1])
    sample = crop[::4, ::4]
    low, high = np.percentile(sample[sample > 0], [1.0, 99.8])
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_B_sparse_IF_training_labels"
    table = out / "panel_B_displayed_training_cells.tsv"
    provenance = out / "panel_B_provenance.json"
    overwritten = protect_outputs([stem.with_suffix(".pdf"), stem.with_suffix(".png"), table, provenance], args.overwrite)
    figure, axis = plt.subplots(figsize=(5.0, 4.3))
    axis.imshow(crop, cmap="gray", vmin=low, vmax=high, extent=(x0, x1, y1, y0), interpolation="nearest", rasterized=True)
    artist = axis.scatter(rows.centroid_x_fullres_px, rows.centroid_y_fullres_px, c=rows.target_axis, cmap="viridis", vmin=0, vmax=1, s=6.0, linewidths=0, rasterized=True)
    axis.set_xlim(x0, x1); axis.set_ylim(y1, y0); axis.axis("off")
    axis.set_title(f"Sparse IF training labels · three villi · n = {len(rows):,}", fontweight="bold")
    colorbar = figure.colorbar(artist, ax=axis, fraction=0.034, pad=0.018)
    colorbar.set_label("Sparse IF reference\nCrypt-villus coordinate")
    # This footer is outside the microscopy axes: no training labels or image
    # pixels are obscured. Horizontal positions stay in full-resolution pixels.
    bar_length_px = SCALE_BAR_UM / pixel_size_um
    bar_x0 = x0 + 0.04 * (x1 - x0)
    bar_x1 = bar_x0 + bar_length_px
    footer_y = -0.045
    transform = axis.get_xaxis_transform()
    axis.plot([bar_x0, bar_x1], [footer_y, footer_y], transform=transform,
              color="black", linewidth=2.5, solid_capstyle="butt", clip_on=False)
    axis.text(bar_x1 + 0.025 * (x1 - x0), footer_y, "200 µm", transform=transform,
              color="black", fontsize=8.5, ha="left", va="center", clip_on=False)
    figure.tight_layout(pad=0.2)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    rows.to_csv(table, sep="\t", index=False)
    write_provenance(
        provenance, script=Path(__file__),
        inputs=[file_record(args.release_manifest, "repaired_IF_release"), file_record(example_path, "prespecified_training_example"), file_record(manifest_path, "IF_supervised_manifest"), file_record(image_path, "frozen_IF_DAPI_cache"), file_record(checkpoint, "released_IF_checkpoint"), file_record(calibration_path, "frozen_IF_source_pixel_calibration")],
        outputs=[*rendered, table],
        extra={
            "panel": "Figure 4b data component", "split": "train", "villi": selected_villi,
            "displayed_training_cells": len(rows), "strict_test_cells_used": False,
            "manual_workflow_artwork_generated": False, "inference_run": False,
            "full_resolution_bounds_xyxy_px": [x0, y0, x1, y1],
            "field_of_view_um": {"width": (x1 - x0) * pixel_size_um, "height": (y1 - y0) * pixel_size_um},
            "source_pixel_size_um": pixel_size_um,
            "display_normalization": {"percentiles": [1.0, 99.8], "subsample_step": 4, "vmin": float(low), "vmax": float(high)},
            "scale_bar": {
                "length_um": SCALE_BAR_UM, "length_full_resolution_px": bar_length_px,
                "pixels_per_um": 1.0 / pixel_size_um, "full_resolution_x_endpoints_px": [bar_x0, bar_x1],
                "position": "external_footer", "axes_fraction_y": footer_y,
                "label": "200 µm", "font_size_pt": 8.5,
                "vector_in_pdf": True, "obscures_image_or_cells": False,
            },
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
