#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, load_prepared_manifest, protect_outputs, require_file, save_pair, write_provenance


STEMS = {
    "morphology": "panel_C_heldout_xenium_morphology",
    "manual": "panel_C_heldout_manual_peyer_label",
    "probability": "panel_C_heldout_gat_peyer_probability",
    "class": "panel_C_heldout_gat_peyer_class",
}


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image)]
    values = values[values > 0]
    if values.size < 128:
        values = image[np.isfinite(image)]
    low, high = np.percentile(values, [0.5, 99.8])
    return np.clip((image.astype(np.float32) - float(low)) / float(high - low), 0, 1)


def style_axis(ax, bounds: tuple[float, float, float, float], title: str) -> None:
    xmin, xmax, ymin, ymax = bounds
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_scale_bar(ax, bounds: tuple[float, float, float, float], length_um: float, color: str, outline: str) -> None:
    xmin, xmax, ymin, ymax = bounds
    width, height = xmax - xmin, ymax - ymin
    x0, y0 = xmin + 0.07 * width, ymax - 0.08 * height
    line = ax.plot([x0, x0 + length_um], [y0, y0], color=color, linewidth=2.2, solid_capstyle="butt")[0]
    line.set_path_effects([pe.Stroke(linewidth=4.0, foreground=outline), pe.Normal()])
    label = ax.text(x0 + length_um / 2, y0 - 0.035 * height, f"{int(length_um)} µm", color=color, ha="center", va="bottom", fontsize=7)
    label.set_path_effects([pe.Stroke(linewidth=2.5, foreground=outline), pe.Normal()])


def save_single(fig, stem: Path) -> list[Path]:
    outputs = save_pair(fig, stem)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--window-metadata", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path, manifest = load_prepared_manifest(args.prepared_manifest)
    cells_path = require_file(args.cells, manifest["prepared_outputs"]["panel_c_window_cells"]["sha256"])
    morphology_path = require_file(args.morphology, manifest["prepared_outputs"]["panel_c_morphology"]["sha256"])
    metadata_path = require_file(args.window_metadata, manifest["prepared_outputs"]["panel_c_window_metadata"]["sha256"])
    cells = pd.read_parquet(cells_path)
    image = np.load(morphology_path, mmap_mode="r")
    metadata = json.loads(metadata_path.read_text())
    if len(cells) != metadata["window_cells"] or not cells["split"].astype(str).eq("test").all():
        raise RuntimeError("Panel C cell table no longer matches the locked test window")
    threshold = float(metadata["threshold"])
    recalculated = cells["gat_peyer_probability"].to_numpy() >= threshold
    if not np.array_equal(recalculated.astype(int), cells["gat_peyer_class"].to_numpy(dtype=int)):
        raise RuntimeError("Panel C hard calls do not match the frozen threshold")

    out = args.output_dir.expanduser().resolve()
    all_outputs = [out / f"{stem}.{suffix}" for stem in STEMS.values() for suffix in ("pdf", "png")]
    provenance = out / "panel_C_heldout_window_provenance.json"
    overwritten = protect_outputs(all_outputs + [provenance], args.overwrite)
    b = metadata["crop_bounds_um"]
    bounds = (float(b["x_min"]), float(b["x_max"]), float(b["y_min"]), float(b["y_max"]))
    scale_bar_um = float(metadata["scale_bar_um"])

    fig, ax = plt.subplots(figsize=(3.0, 3.0), constrained_layout=True)
    ax.imshow(contrast_stretch(image), cmap="gray", extent=(bounds[0], bounds[1], bounds[3], bounds[2]), interpolation="nearest")
    style_axis(ax, bounds, "Held-out Xenium morphology")
    add_scale_bar(ax, bounds, scale_bar_um, "white", "black")
    save_single(fig, out / STEMS["morphology"])

    truth = cells["peyer_manual_label"].to_numpy(dtype=float) >= 0.5
    fig, ax = plt.subplots(figsize=(3.0, 3.0), constrained_layout=True)
    ax.scatter(cells.loc[~truth, "x_spatial"], cells.loc[~truth, "y_spatial"], s=0.65, color="#C5C5C5", alpha=0.68, linewidths=0, rasterized=True)
    ax.scatter(cells.loc[truth, "x_spatial"], cells.loc[truth, "y_spatial"], s=2.2, color="#7652A7", alpha=0.98, linewidths=0, rasterized=True)
    style_axis(ax, bounds, "Held-out manual Peyer label")
    add_scale_bar(ax, bounds, scale_bar_um, "#333333", "white")
    save_single(fig, out / STEMS["manual"])

    fig, ax = plt.subplots(figsize=(3.25, 3.0), constrained_layout=True)
    scatter = ax.scatter(cells["x_spatial"], cells["y_spatial"], c=cells["gat_peyer_probability"], s=1.0, cmap="magma", vmin=0, vmax=1, linewidths=0, rasterized=True)
    style_axis(ax, bounds, "Held-out GAT Peyer’s patch probability")
    add_scale_bar(ax, bounds, scale_bar_um, "#333333", "white")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label("Peyer’s patch probability", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    save_single(fig, out / STEMS["probability"])

    fig, ax = plt.subplots(figsize=(3.0, 3.0), constrained_layout=True)
    ax.scatter(cells.loc[~recalculated, "x_spatial"], cells.loc[~recalculated, "y_spatial"], s=0.65, color="#C5C5C5", alpha=0.68, linewidths=0, rasterized=True)
    ax.scatter(cells.loc[recalculated, "x_spatial"], cells.loc[recalculated, "y_spatial"], s=2.2, color="#4F8A5B", alpha=0.98, linewidths=0, rasterized=True)
    style_axis(ax, bounds, "Held-out GAT Peyer class prediction")
    add_scale_bar(ax, bounds, scale_bar_um, "#333333", "white")
    save_single(fig, out / STEMS["class"])

    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(cells_path, "locked_test_window_cells"), file_record(morphology_path, "locked_morphology_crop"), file_record(metadata_path, "locked_window_metadata"), file_record(manifest_path, "prepared_input_manifest")],
        outputs=all_outputs,
        overwritten=overwritten,
        extra={"panel": "C", "section": metadata["section"], "split": metadata["split"], "cell_count": int(len(cells)), "hard_call_threshold": threshold, "display_downsampling": False},
    )
    for output in all_outputs:
        if output.suffix == ".pdf":
            print(output)


if __name__ == "__main__":
    main()
