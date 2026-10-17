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
from common import file_record, protect_outputs, sha256, write_provenance, verify_training_input_parity  # noqa: E402


def save(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return [pdf, png]


def image_panel(image: np.ndarray, stem: Path, dpi: int) -> list[Path]:
    nonzero = image[image > 0]
    low, high = np.percentile(nonzero if nonzero.size else image, [0.5, 99.8])
    aspect = image.shape[1] / image.shape[0]
    fig = plt.figure(figsize=(6.0, 6.0 / aspect), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    ax.axis("off")
    return save(fig, stem, dpi)


def coordinate_panel(
    cells: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    value_column: str,
    limits: tuple[float, float, float, float],
    stem: Path,
    dpi: int,
    point_size: float,
) -> list[Path]:
    x0, x1, y0, y1 = limits
    aspect = (x1 - x0) / (y1 - y0)
    fig = plt.figure(figsize=(6.0, 6.0 / aspect), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(
        cells[x_column],
        cells[y_column],
        c=cells[value_column],
        cmap="viridis",
        norm=Normalize(0, 1),
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.axis("off")
    return save(fig, stem, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the four image/data rectangles used in the supplied Figure 1d artwork."
    )
    parser.add_argument("--training-dapi", type=Path, required=True)
    parser.add_argument("--training-graph-cells", type=Path, required=True)
    parser.add_argument("--inference-dapi", type=Path, required=True)
    parser.add_argument("--image-model-predictions", type=Path, required=True)
    parser.add_argument("--image-model-provenance", type=Path, required=True)
    parser.add_argument("--inference-contract", type=Path, required=True)
    parser.add_argument("--region-lock", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--frozen-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output_dir.expanduser().resolve()
    names = (
        "panel_D_paired_training_graph_coordinate",
        "panel_D_paired_training_dapi",
        "panel_D_unseen_day90_dapi",
        "panel_D_released_image_model_coordinate_prediction",
    )
    outputs = [out / f"{name}{suffix}" for name in names for suffix in (".pdf", ".png")]
    provenance_path = out / "panel_D_provenance.json"
    overwritten = protect_outputs(outputs + [provenance_path], args.overwrite)

    lock = json.loads(args.region_lock.read_text())
    input_manifest = json.loads(args.input_manifest.read_text())
    registry = json.loads(args.frozen_registry.read_text())
    inference_provenance = json.loads(args.image_model_provenance.read_text())
    parity_path = verify_training_input_parity(inference_provenance, args.inference_contract)
    training_region = lock["regions"]["paired_training_example"]
    inference_region = lock["regions"]["unseen_image_example"]
    training_limits = tuple(map(float, training_region["limits_um_xyxy"]))
    inference_window = tuple(map(float, inference_region["pixel_window_xyxy"]))

    manifest_crops = input_manifest["crop_definitions"]
    if tuple(manifest_crops["day90_training_pixel_window"]) != tuple(training_region["pixel_window_xyxy"]):
        raise RuntimeError("Prepared training DAPI does not match the locked Figure 1d window.")
    if tuple(manifest_crops["day90_inference_pixel_window"]) != tuple(inference_region["pixel_window_xyxy"]):
        raise RuntimeError("Prepared inference DAPI does not match the locked Figure 1d window.")

    model = registry["models"]["xenium_image_coordinate_model"]
    checkpoint = model["checkpoint"]
    if sha256(checkpoint["path"]) != checkpoint["sha256"]:
        raise RuntimeError("Released Xenium checkpoint no longer matches the frozen registry.")
    if inference_provenance.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise RuntimeError("Figure 1d image-model predictions came from a different checkpoint.")
    prediction_record = next(
        record for record in inference_provenance["outputs"] if Path(record["path"]).name == "predictions.csv"
    )
    if sha256(args.image_model_predictions) != prediction_record["sha256"]:
        raise RuntimeError("Figure 1d image-model prediction table does not match its provenance.")

    training_dapi = np.load(args.training_dapi)
    inference_dapi = np.load(args.inference_dapi)
    training_cells = pd.read_parquet(args.training_graph_cells)
    predictions = pd.read_csv(args.image_model_predictions)
    if len(predictions) != int(inference_provenance["row_count"]):
        raise RuntimeError("Figure 1d prediction-table row count does not match its provenance.")

    rendered: list[Path] = []
    rendered += coordinate_panel(
        training_cells,
        x_column="x",
        y_column="y",
        value_column="crypt_villus_prediction",
        limits=training_limits,
        stem=out / names[0],
        dpi=args.dpi,
        point_size=2.4,
    )
    rendered += image_panel(training_dapi, out / names[1], args.dpi)
    rendered += image_panel(inference_dapi, out / names[2], args.dpi)
    rendered += coordinate_panel(
        predictions,
        x_column="centroid_x_fullres_px",
        y_column="centroid_y_fullres_px",
        value_column="predicted_axis_coordinate",
        limits=inference_window,
        stem=out / names[3],
        dpi=args.dpi,
        point_size=2.4,
    )

    inputs = [
        file_record(args.training_dapi, "locked_training_dapi_crop"),
        file_record(args.training_graph_cells, "frozen_graph_coordinate_cells"),
        file_record(args.inference_dapi, "locked_unseen_dapi_crop"),
        file_record(args.image_model_predictions, "released_image_model_predictions"),
        file_record(args.image_model_provenance, "released_image_model_inference_provenance"),
        file_record(args.inference_contract, "training_input_contract"),
        file_record(parity_path, "golden_training_input_parity"),
        file_record(args.region_lock, "figure1d_region_lock"),
        file_record(args.input_manifest, "figure1_input_manifest"),
        file_record(args.frozen_registry, "frozen_registry"),
    ]
    write_provenance(
        provenance_path,
        script=HERE,
        inputs=inputs,
        outputs=rendered,
        extra={
            "panel": "Figure 1d",
            "component_count": 4,
            "component_display_order": list(names),
            "source_batch": lock["source_section"]["batch"],
            "training_pixel_window_xyxy": training_region["pixel_window_xyxy"],
            "inference_pixel_window_xyxy": inference_region["pixel_window_xyxy"],
            "xenium_model_version": model["version"],
            "checkpoint": checkpoint,
            "warnings": [],
        },
        overwritten=overwritten,
    )
    print(provenance_path)


if __name__ == "__main__":
    main()
