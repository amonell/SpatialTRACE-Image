#!/usr/bin/env python3
"""Render Figure 3a's original IF pretraining and approved Xenium inference examples."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402
from figure_inputs import paired_views, locked_input, PAIRED_GEOMETRY  # noqa: E402
from Figure_3.supervised_example import supervised_example, prediction_background, add_prediction_overlay  # noqa: E402

EXPECTED_CHECKPOINT = "d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137"
EXPECTED_PRETRAIN = "d4698fe67251bb56b805624a1e810cacc86f1a553b969ae4b23d54ba5ff3c23a"
EXPECTED_CENTER_ID = "if:cont1:nucleus:352082"
BOX_COLORS = {"context": "#ff9f1c", "local": "#2ec4ff", "fine": "#ff4fd8"}
SCALE_BARS_UM = {"overview": 200.0, "context": 100.0, "local": 50.0, "fine": 10.0}


def add_scale_bar(axis: plt.Axes, image_size_px: int, pixel_size_um: float, bar_um: float) -> None:
    width = bar_um / pixel_size_um
    x1 = image_size_px * 0.94
    x0 = x1 - width
    y = image_size_px * 0.92
    axis.plot(
        [x0, x1], [y, y], color="white",
        linewidth=max(2.2, image_size_px / 850), solid_capstyle="butt",
        path_effects=[path_effects.Stroke(linewidth=max(3.5, image_size_px / 550), foreground="black"), path_effects.Normal()],
    )
    label = axis.text(
        (x0 + x1) / 2, y - image_size_px * 0.026, f"{int(bar_um)} μm",
        color="white", fontsize=max(7.5, image_size_px / 300), ha="center", va="bottom",
    )
    label.set_path_effects([path_effects.Stroke(linewidth=2.0, foreground="black"), path_effects.Normal()])


def add_target_outline(axis: plt.Axes, mask: np.ndarray, image_size_px: int) -> None:
    if np.any(mask):
        axis.contour(mask.astype(float), levels=[0.5], colors=["#00f0ff"], linewidths=max(0.8, image_size_px / 1300))


def add_center_box(axis: plt.Axes, outer_px: int, box_px: int, color: str, label: str) -> None:
    start = (outer_px - box_px) / 2
    axis.add_patch(Rectangle((start, start), box_px, box_px, fill=False, edgecolor=color, linewidth=max(1.4, outer_px / 520)))
    if label:
        text = axis.text(
            start + outer_px * 0.012, start - outer_px * 0.014, label,
            color=color, fontsize=max(8, outer_px / 230), fontweight="bold", ha="left", va="bottom",
        )
        text.set_path_effects([path_effects.Stroke(linewidth=2.2, foreground="black"), path_effects.Normal()])


def crop_figure(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    pixel_size_um: float,
    bar_um: float,
    boxes: tuple[tuple[int, str, str], ...] = (),
) -> plt.Figure:
    size = int(image.shape[0])
    figure, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none")
    add_target_outline(axis, mask, size)
    for box_px, color, label in boxes:
        add_center_box(axis, size, box_px, color, label)
    add_scale_bar(axis, size, pixel_size_um, bar_um)
    axis.set_xlim(0, size)
    axis.set_ylim(size, 0)
    axis.axis("off")
    figure.tight_layout(pad=0)
    return figure


def image_figure(image: np.ndarray) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(3.0, 3.0))
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none")
    axis.axis("off")
    figure.tight_layout(pad=0)
    return figure


def apply_dihedral(array: np.ndarray, index: int) -> np.ndarray:
    transformed = np.rot90(array, k=index % 4) if index % 4 else np.asarray(array)
    if index >= 4:
        transformed = np.fliplr(transformed)
    return np.ascontiguousarray(transformed)


def masked(image: np.ndarray, seed: int, fraction: float = 0.40) -> np.ndarray:
    result = np.asarray(image).copy()
    grid = 16
    patch = result.shape[0] // grid
    if result.shape != (256, 256) or patch != 16:
        raise RuntimeError(f"Expected a 256 × 256 model input; observed {result.shape}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(grid * grid, size=int(round(grid * grid * fraction)), replace=False)
    for index in indices:
        row, column = divmod(int(index), grid)
        result[row * patch : (row + 1) * patch, column * patch : (column + 1) * patch] = 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-data", type=Path, required=True)
    parser.add_argument("--region-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretraining-checkpoint", type=Path, required=True)
    parser.add_argument("--production-selection", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--pretraining-summary", type=Path, required=True)
    parser.add_argument("--supervised-manifest", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--paired-views", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    region_path = locked_input(args.region_data)
    region_metadata_path = locked_input(args.region_metadata)
    checkpoint = require_file(args.checkpoint, EXPECTED_CHECKPOINT)
    pretraining_checkpoint = require_file(args.pretraining_checkpoint, EXPECTED_PRETRAIN)
    selection_path = require_file(args.production_selection)
    training_summary_path = require_file(args.training_summary)
    pretraining_summary_path = require_file(args.pretraining_summary)
    metadata = json.loads(region_metadata_path.read_text())
    if metadata.get("center_id") != EXPECTED_CENTER_ID:
        raise RuntimeError(f"Figure 3a must use {EXPECTED_CENTER_ID}; observed {metadata.get('center_id')}")
    views, paired_record = paired_views(args.paired_manifest, args.paired_views, args.seed)
    arrays = np.load(region_path)
    pixel_size_um = float(metadata["pixel_size_um"])
    example=supervised_example(args.supervised_manifest)
    bounded=example['table'];bounded_record=example['manifest'];supervised_inputs=example['images']
    out = args.output_dir.expanduser().resolve()

    names = (
        "panel_A_overview_nested_fields",
        "panel_A_context_with_local_field",
        "panel_A_local_with_fine_field",
        "panel_A_context_input",
        "panel_A_local_input",
        "panel_A_fine_target_overlay",
        "panel_A_dihedral_augmentations",
        "panel_A_paired_pretraining_views",
        "panel_A_fine_sft_input",
        "panel_A_crypt_villus_head_prediction",
        "panel_A_epithelial_head_prediction",
        "panel_A_crypt_villus_regional_prediction",
        "panel_A_epithelial_regional_prediction",
    )
    table = out / "panel_A_architecture_data.tsv"
    provenance = out / "panel_A_provenance.json"
    outputs = [out / f"{name}.{suffix}" for name in names for suffix in ("pdf", "png")]
    overwritten = protect_outputs(outputs + [table, provenance], args.overwrite)
    rendered: list[Path] = []

    figure = crop_figure(
        arrays["overview"], arrays["overview_target_mask"],
        pixel_size_um=pixel_size_um, bar_um=SCALE_BARS_UM["overview"],
        boxes=((2048, BOX_COLORS["context"], "2048 px"), (512, BOX_COLORS["local"], "512 px"), (128, BOX_COLORS["fine"], "128 px")),
    )
    rendered += save_pair(figure, out / names[0], args.dpi); plt.close(figure)
    figure = crop_figure(
        arrays["context"], arrays["context_target_mask"],
        pixel_size_um=pixel_size_um, bar_um=SCALE_BARS_UM["context"],
        boxes=((512, BOX_COLORS["local"], "512 px"),),
    )
    rendered += save_pair(figure, out / names[1], args.dpi); plt.close(figure)
    figure = crop_figure(
        arrays["local"], arrays["local_target_mask"],
        pixel_size_um=pixel_size_um, bar_um=SCALE_BARS_UM["local"],
        boxes=((128, BOX_COLORS["fine"], "128 px"),),
    )
    rendered += save_pair(figure, out / names[2], args.dpi); plt.close(figure)

    for image, name in (
        (supervised_inputs["context_image"], names[3]),
        (supervised_inputs["local_image"], names[4]),
    ):
        figure = image_figure(image)
        rendered += save_pair(figure, out / name, args.dpi); plt.close(figure)
    figure = crop_figure(
        arrays["fine"], arrays["fine_target_mask"],
        pixel_size_um=pixel_size_um, bar_um=SCALE_BARS_UM["fine"],
    )
    rendered += save_pair(figure, out / names[5], args.dpi); plt.close(figure)

    transform_specs = ((0, "Original"), (1, "Rotation 90°"), (5, "Rotation + flip"))
    figure, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for axis, (index, label) in zip(axes.ravel(), transform_specs, strict=True):
        image = apply_dihedral(arrays["context"], index)
        target = apply_dihedral(arrays["context_target_mask"], index)
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none")
        add_target_outline(axis, target, 2048)
        add_center_box(axis, 2048, 512, BOX_COLORS["local"], "")
        axis.set_title(label)
        axis.axis("off")
    figure.tight_layout(pad=0.25)
    rendered += save_pair(figure, out / names[6], args.dpi); plt.close(figure)

    paired = ((views['student_local'], 'Masked student local view'),
              (views['student_context'], 'Masked student context view'),
              (views['teacher_local'], 'Clean teacher local view'),
              (views['teacher_context'], 'Clean teacher context view'))
    figure, axes = plt.subplots(2, 2, figsize=(4.8, 4.8))
    for axis, (image, label) in zip(axes.ravel(), paired, strict=True):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none")
        axis.set_title(label)
        axis.axis("off")
    figure.tight_layout(pad=0.3)
    rendered += save_pair(figure, out / names[7], args.dpi); plt.close(figure)

    figure = image_figure(supervised_inputs['fine_image'])
    rendered += save_pair(figure, out / names[8], args.dpi); plt.close(figure)
    for name, key in zip(names[9:], ('head_cv','head_epi','context_cv','context_epi'), strict=True):
        figure = image_figure(prediction_background(example,key))
        add_prediction_overlay(figure.axes[0],example,key)
        rendered += save_pair(figure, out / name, args.dpi); plt.close(figure)

    selection = json.loads(selection_path.read_text())
    training = json.loads(training_summary_path.read_text())
    pretraining = json.loads(pretraining_summary_path.read_text())
    pretraining_rows = {row["scale_name"]: row for row in metadata["pretraining_rows"]}
    rows = [
        ("Released model", "paired-representation-shared-scale-v2"),
        ("Production checkpoint SHA256", EXPECTED_CHECKPOINT),
        ("Pretraining checkpoint SHA256", EXPECTED_PRETRAIN),
        ("Figure 3a source", metadata["source_id"]),
        ("Figure 3a center ID", metadata["center_id"]),
        ("Nucleus label ID", str(metadata["nucleus_label_id"])),
        ("Center x, full-resolution pixels", str(metadata["center_x_fullres_px"])),
        ("Center y, full-resolution pixels", str(metadata["center_y_fullres_px"])),
        ("Overview crop", "4096 px"),
        ("Local crop", "512 px → 256 px model input"),
        ("Context crop", "2048 px → 256 px model input"),
        ("Fine crop", "128 px → 128 px model input"),
        ("Pretraining local prepared index", str(pretraining_rows["local"]["prepared_index"])),
        ("Pretraining context prepared index", str(pretraining_rows["context"]["prepared_index"])),
        ("Supervised-side source", bounded_record['stable_source_id']),
        ("Supervised-side illustrated row", bounded_record['illustrated_row_id']),
        ("Supervised-side prepared index", str(bounded_record['illustrated_prepared_index'])),
        ("Supervised-side native local/context/fine widths", "783 / 3132 / 196 px at 0.2125 µm/px"),
        ("Supervised-side released SFT assignment", bounded_record['released_sft_assignment']),
        ("Patch size", "16 px"),
        ("Token grid", "16 × 16"),
        ("Transformer readout", "central 4 × 4 output-token mean"),
        ("Encoder", "shared ViT with learned local/context scale embeddings"),
        ("Pretraining objective", "paired representation prediction with EMA teacher"),
        ("Pretraining mask fraction", "0.40"),
        ("SFT selected seed", str(selection["selected_seed"])),
        ("SFT selected epoch", str(selection["selected_epoch"])),
        ("SFT train rows", str(training["train_row_count"])),
        ("SFT validation rows", str(training["validation_row_count"])),
        ("Pretraining epochs completed", str(pretraining.get("completed_epochs", pretraining.get("epochs", 35)))),
    ]
    pd.DataFrame(rows, columns=["field", "value"]).to_csv(table, sep="\t", index=False)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[
            file_record(region_path, "exact_historical_figure3a_region"),
            file_record(region_metadata_path, "exact_historical_region_metadata"),
            file_record(checkpoint, "released_production_v2_checkpoint"),
            file_record(pretraining_checkpoint, "paired_representation_pretraining_checkpoint"),
            file_record(selection_path, "validation_only_production_selection"),
            file_record(training_summary_path, "sft_training_summary"),
            file_record(pretraining_summary_path, "pretraining_summary"),
            file_record(args.supervised_manifest, "frozen_Xenium_supervised_example"),
            file_record(args.paired_manifest, "actual_dataset_paired_view_manifest"),
            file_record(args.paired_views, "actual_dataset_paired_views"),
            *[file_record(record['path'], record['role']) for record in bounded_record['outputs']],
            file_record(ROOT / 'figure_inputs.py', 'shared_assembly_component_input_logic'),
            file_record(ROOT / 'supervised_example.py', 'shared_Xenium_prediction_renderer'),
        ],
        outputs=[*rendered, table],
        extra={
            "panel": "Figure 3a component assets only",
            "manual_architecture_artwork_generated": False,
            "exact_historical_region_recovered": True,
            "center_id": EXPECTED_CENTER_ID,
            "mask_seed": args.seed,
            "mask_fraction": 0.40,
            "augmentation_family": "D4 examples",
            "paired_geometry": PAIRED_GEOMETRY,
            "paired_source": {key: paired_record[key] for key in ('center_id', 'local_prepared_index', 'context_prepared_index', 'student_variant', 'teacher_variant', 'shared_student_intensity_parameters')},
            "mask_visualization": paired_record['mask_visualization'],
            "bounded_prediction_rows": len(bounded),
            "supervised_input_protocol": bounded_record['input_protocol'],
            "bounded_prediction_scope": bounded_record['evaluation_scope'],
            "supervised_source": bounded_record['stable_source_id'],
            "supervised_illustrated_row_id": bounded_record['illustrated_row_id'],
            "prediction_marker_semantics": bounded_record['display']['marker_semantics'],
            "prediction_color_limits": [0, 1],
        },
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
