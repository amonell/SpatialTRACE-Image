#!/usr/bin/env python3
"""Regenerate original Figure 4a microscopy crops, not whole-tissue DAPI."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_4.common import file_record, protect_outputs, require_file, write_provenance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmso-image", type=Path, required=True)
    parser.add_argument("--rari-image", type=Path, required=True)
    parser.add_argument("--region-lock", type=Path, required=True)
    parser.add_argument("--scale-calibration", type=Path, required=True)
    parser.add_argument("--style-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    lock_path = require_file(args.region_lock, "bf6c3064da8f7e0c97facb1bb4bac430c5d0fc1e58a88f57a37ad3e8af9720f5")
    lock = json.loads(lock_path.read_text())
    calibration_path = require_file(args.scale_calibration, "7acc25d5ae4bbdb3e4562aa91166ca67f50ded5798850eb046db6c8bcceec4e8")
    calibration = json.loads(calibration_path.read_text())
    reference = require_file(args.style_reference, lock["artwork_sha256"])
    sources = {
        "DMSO": require_file(args.dmso_image, lock["regions"]["DMSO"]["source_sha256"]),
        "RARi": require_file(args.rari_image, lock["regions"]["RARi"]["source_sha256"]),
    }
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")
    font_path = Path(findfont(FontProperties(family="Arial"), fallback_to_default=False))
    font_size_pt = 8.5
    font = ImageFont.truetype(str(font_path), round(font_size_pt * args.dpi / 72))
    out = args.output_dir.expanduser().resolve()
    stems = {condition: out / f"panel_A_{condition.lower()}_IF_overlay" for condition in sources}
    outputs = [stem.with_suffix(ext) for stem in stems.values() for ext in (".pdf", ".png")]
    provenance = out / "panel_A_provenance.json"
    overwritten = protect_outputs([*outputs, provenance], args.overwrite)
    records = {}
    for condition, path in sources.items():
        spec = lock["regions"][condition]
        source = Image.open(path).convert("RGB")
        if list(source.size) != spec["source_size_px"]:
            raise ValueError(f"Unexpected snapshot dimensions for {condition}")
        if calibration["conditions"][condition]["source_sha256"] != spec["source_sha256"]:
            raise ValueError(f"Scale calibration source does not match {condition} crop source")
        bx0, by0, bx1, by1 = calibration["source_bar_bounds_xyxy"]
        pixels = np.asarray(source)
        # The source scale is explicitly printed as 50 µm. Its same 79-pixel
        # stroke is present on three rows in both original snapshots.
        if not (pixels[by0:by1, bx0:bx1].min(axis=2) >= 126).all():
            raise ValueError(f"Original source scale-bar pixels no longer verify for {condition}")
        bar_length_px = calibration["rendered_bar_length_snapshot_px"]
        if bar_length_px != bx1 - bx0 or calibration["physical_length_um"] != 50:
            raise ValueError("Unexpected original snapshot scale calibration")
        crop = source.crop(tuple(spec["source_crop_xyxy"]))
        if spec["rotation_degrees"] == 180:
            crop = crop.transpose(Image.Transpose.ROTATE_180)
        elif spec["rotation_degrees"] != 0:
            raise ValueError("Only verified original 0/180-degree orientations are permitted")
        # No new contrast, channel recoloring, or fitted color transformation.
        width, height = crop.size
        footer_height = round(0.32 * args.dpi)
        total_height = height + footer_height
        label = calibration["source_bar_label"]
        label_width = font.getlength(label)
        margin = max(28, int((label_width - bar_length_px) / 2 + 12))
        sx1 = width - margin
        sx0 = sx1 - bar_length_px
        sy = round(0.10 * args.dpi)
        text_y = round(0.16 * args.dpi)
        stroke_pt = 2.0
        stroke_px = round(stroke_pt * args.dpi / 72)
        # Append a black rail: never paint the bar or text onto microscopy.
        annotated = Image.new("RGB", (width, total_height), "black")
        annotated.paste(crop, (0, 0))
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((sx0, height + sy, sx1 - 1, height + sy + stroke_px - 1), fill="white")
        draw.text(((sx0 + sx1) / 2, height + text_y), label, fill="white", font=font, anchor="mt")
        annotated.save(stems[condition].with_suffix(".png"), dpi=(args.dpi, args.dpi))
        figure = plt.figure(figsize=(width / args.dpi, total_height / args.dpi), facecolor="black")
        axis = figure.add_axes([0, footer_height / total_height, 1, height / total_height])
        axis.imshow(crop, interpolation="none", aspect="auto", extent=(0, width, height, 0))
        axis.axis("off")
        footer = figure.add_axes([0, 0, 1, footer_height / total_height])
        footer.set_xlim(0, width)
        footer.set_ylim(footer_height, 0)
        footer.axis("off")
        footer.plot([sx0, sx1], [sy + stroke_px / 2, sy + stroke_px / 2],
                    color="white", lw=stroke_pt, solid_capstyle="butt")
        footer.text((sx0 + sx1) / 2, text_y, label, color="white",
                    fontsize=font_size_pt, ha="center", va="top", fontfamily="Arial")
        figure.savefig(stems[condition].with_suffix(".pdf"), dpi=args.dpi, pad_inches=0)
        plt.close(figure)
        records[condition] = {**spec, "output_size_px": [width, total_height],
                              "microscopy_bounds_output_px": [0, 0, width, height],
                              "color_operation": "none: original RGB preserved",
                              "crop_pixels_preserved_exactly": True,
                              "scale_bar": {"length_um": 50, "length_snapshot_px": bar_length_px,
                                            "source_reference_bar_xyxy": [bx0, by0, bx1, by1],
                                            "placement": "external black footer", "pdf_vector": True,
                                            "label_font_pt": font_size_pt, "footer_height_px": footer_height}}
    write_provenance(
        provenance, script=Path(__file__),
        inputs=[file_record(path, f"original_{condition}_multichannel_snapshot") for condition, path in sources.items()]
        + [file_record(lock_path, "artwork_registered_region_lock"),
           file_record(calibration_path, "original_snapshot_scale_calibration"),
           file_record(font_path, "Arial_font_for_PNG_scale_label"),
           file_record(reference, "artwork_reference_only_not_render_source")],
        outputs=outputs, overwritten=overwritten,
        extra={"panel": "Figure 4a microscopy components", "records": records,
               "manual_artwork_generated": False, "model_inference": False,
               "random_seed": None, "dpi": args.dpi,
               "warnings": [
                   "Original RGB snapshot colors preserved. The assembled PNG differs; its unrecorded export color transform has not been inferred or applied.",
                   "Region bounds recovered from raster artwork with approximately one source-pixel registration uncertainty.",
                   "Snapshot-to-raw-slide transform not verified. These are not raw VSI crop coordinates.",
                   "Scale bar transfers the original 50 µm / 79 raster-pixel stroke without changing image scale; derived pixel calibration has source-raster precision."]},
    )


if __name__ == "__main__":
    main()
