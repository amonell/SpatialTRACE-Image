#!/usr/bin/env python3
"""Render real student/teacher views used by paired representation pretraining."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_3_Extended.common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402

EXPECTED_REGION = "1b58cdfe13909c7da88c225e9f98250b6bbf4bb485e05c7fdaeaa3900e77bbaa"
EXPECTED_METADATA = "55e893e1c53a07e9ac74a9f7e05c3617887e59edfe2cec10d2d1579f216d86d5"
EXPECTED_PRETRAIN_SUMMARY = "daecdd0e75be017a65bd231a062d2a7ca77960cf237a169ab4dca552ae023f1f"
EXPECTED_CENTER = "if:cont1:nucleus:352082"
EXPECTED_PAIRED_MANIFEST = "22c7cede602b605e8b3eed02aa0f4579988427b674bd5ccf8f1c1896758f2e7d"
MASK_DISPLAY_GRAY = 128


def apply_mask(image: np.ndarray, seed: int, fraction: float) -> tuple[np.ndarray, list[int]]:
    if image.shape != (256, 256):
        raise RuntimeError(f"Expected 256 x 256 transformer input; observed {image.shape}")
    rng = np.random.default_rng(seed)
    indices = sorted(map(int, rng.choice(256, size=int(round(256 * fraction)), replace=False)))
    output = image.copy()
    for index in indices:
        row, column = divmod(index, 16)
        output[row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16] = MASK_DISPLAY_GRAY
    return output, indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-data", type=Path, required=True)
    parser.add_argument("--region-metadata", type=Path, required=True)
    parser.add_argument("--pretraining-summary", type=Path, required=True)
    parser.add_argument("--paired-views", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    region = require_file(args.region_data, EXPECTED_REGION)
    metadata_path = require_file(args.region_metadata, EXPECTED_METADATA)
    summary_path = require_file(args.pretraining_summary, EXPECTED_PRETRAIN_SUMMARY)
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    if metadata.get("center_id") != EXPECTED_CENTER:
        raise RuntimeError(f"Expected exact historical center {EXPECTED_CENTER}")
    if summary.get("objective") != "masked_paired_scale_ema_feature_prediction":
        raise RuntimeError("Panel A requires the paired representation pretraining objective")
    fraction = float(summary["mask_fraction"])
    paired_path = require_file(args.paired_manifest, EXPECTED_PAIRED_MANIFEST)
    paired = json.loads(paired_path.read_text())
    if paired["center_id"] != EXPECTED_CENTER or not paired["same_geometry_and_intensity_parameters_for_both_student_scales"]:
        raise RuntimeError("Paired pretraining augmentation lineage mismatch")
    views_path = require_file(args.paired_views, paired["views"]["sha256"])
    arrays = np.load(views_path)
    local = arrays["local_teacher_image"] * 255
    context = arrays["context_teacher_image"] * 255
    student_local, local_indices = apply_mask(arrays["local_student_image"] * 255, args.seed, fraction)
    student_context, context_indices = apply_mask(arrays["context_student_image"] * 255, args.seed + 1, fraction)
    panels = (
        (student_local, "Masked student local view"),
        (local, "EMA-teacher local target"),
        (student_context, "Masked student context view"),
        (context, "EMA-teacher context target"),
    )
    out = args.output_dir.expanduser().resolve()
    stem = out / "panel_A_paired_pretraining_views"
    mask_table = out / "panel_A_masked_patch_indices.tsv"
    provenance = out / "panel_A_provenance.json"
    outputs = [stem.with_suffix(".pdf"), stem.with_suffix(".png"), mask_table, provenance]
    overwritten = protect_outputs(outputs, args.overwrite)
    figure, axes = plt.subplots(1, 4, figsize=(8.0, 2.2))
    for axis, (image, title) in zip(axes.ravel(), panels, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none")
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.10)
    rendered = save_pair(figure, stem, args.dpi)
    plt.close(figure)
    pd.DataFrame(
        [("local", args.seed, index) for index in local_indices]
        + [("context", args.seed + 1, index) for index in context_indices],
        columns=["scale", "mask_seed", "patch_index"],
    ).to_csv(mask_table, sep="\t", index=False)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[
            file_record(region, "exact_Figure3a_region_arrays"),
            file_record(metadata_path, "exact_region_metadata"),
            file_record(summary_path, "paired_representation_pretraining_summary"),
            file_record(paired_path, "exact_shared_augmentation_record"),
            file_record(views_path, "actual_dataset_paired_views"),
        ],
        outputs=[*rendered, mask_table],
        extra={
            "panel": "Extended Figure 3a component",
            "layout": "1x4",
            "component_order": ["masked_student_local", "EMA_teacher_local", "masked_student_context", "EMA_teacher_context"],
            "center_id": EXPECTED_CENTER,
            "objective": summary["objective"],
            "mask_fraction": fraction,
            "mask_seed_local": args.seed,
            "mask_seed_context": args.seed + 1,
            "model_inference_run": False,
            "student_variant_both_scales": paired["student_variant"],
            "teacher_variant_both_scales": paired["teacher_variant"],
            "shared_student_intensity_parameters": paired["shared_student_intensity_parameters"],
            "mask_visualization": "Gray squares illustrate learned-token masking; they are a display overlay, not the training pixel values.",
            "mask_color_hex": "#808080",
            "mask_gray_value_0_255": MASK_DISPLAY_GRAY,
            "mask_display_only": True,
            "frozen_source_mask_visualization": paired["mask_visualization"],
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
