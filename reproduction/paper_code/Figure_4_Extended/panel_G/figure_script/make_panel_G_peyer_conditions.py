#!/usr/bin/env python3
"""Render both fixed qualitative Peyer windows as components of displayed panel F."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_4_Extended.common import file_record, protect_outputs, require_file, write_provenance  # noqa: E402
from Figure_4_Extended.peyer import render_peyer  # noqa: E402
from Figure_4_Extended.probability_display import DISPLAY_CONFIG, probability_display, probability_range

PIXEL_SIZE_UM = 0.16247698804244626


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("predictions", "summary", "image-cache", "checkpoint", "output-dir", "release-manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("--dpi must be at least 300; PDF probability dots remain vector")
    from frozen_preprocessing.peyer_representation_v1.release import load_release, RELEASE_POINTER
    active_path, active = load_release()
    release_path = require_file(args.release_manifest, active["window_release"]["sha256"])
    release = json.loads(release_path.read_text())
    if release['status'] != 'FROZEN_REPRESENTATION_CLASSIFIER_WINDOWS' or not release['if_qualitative_only']:
        raise RuntimeError('The IF Peyer panels require the representation-pretrained, fixed-window release')
    require_file(release['label_lock']['path'], release['label_lock']['sha256'])
    window_path = require_file(release['window_lock']['path'], release['window_lock']['sha256'])
    windows = json.loads(window_path.read_text())["IF_windows"]
    if [window["condition"] for window in windows] != ["DMSO", "RARi"] or release["threshold"] != .5:
        raise RuntimeError("Unexpected qualitative window order or fixed classifier threshold")
    predictions = require_file(args.predictions, release['inference']['if']['predictions']['sha256'])
    summary = require_file(args.summary, release['inference']['if']['summary']['sha256'])
    image_path = require_file(args.image_cache, "0451386a72b456d622e3f2f595d13d6e7cd5bdea830743b034894be8afb5f6ec")
    checkpoint = require_file(args.checkpoint, release['checkpoint']['sha256'])
    rows = pd.read_csv(predictions)
    display, _ = probability_display()
    out = args.output_dir.expanduser().resolve()
    conditions = [window["condition"] for window in windows]
    stems = {condition: out / f"panel_G_peyer_classifier_{condition}" for condition in conditions}
    tables = {condition: out / f"panel_G_{condition}_window_cells.tsv" for condition in conditions}
    provenance = out / "panel_G_provenance.json"
    outputs = [path for condition in conditions for path in
               (stems[condition].with_suffix(".pdf"), stems[condition].with_suffix(".png"), tables[condition])]
    overwritten = protect_outputs([*outputs, provenance], args.overwrite)
    image = np.load(image_path, mmap_mode="r")
    component_records = []
    for window in windows:
        condition, center, expected_count = window["condition"], window["center_um"], window["cells"]
        half = window["width_um"] / 2
        bounds_um = (center[0] - half, center[0] + half, center[1] - half, center[1] + half)
        if [int(round(value / PIXEL_SIZE_UM)) for value in bounds_um] != window["bounds_px"]:
            raise RuntimeError("Locked microscopy pixel bounds and physical window disagree")
        locked_rows = rows.loc[rows.condition.eq(condition)]
        if len(locked_rows) != expected_count or set(locked_rows.row_id) != set(window["cell_ids"]):
            raise RuntimeError("Classifier predictions do not match the locked window identities")
        _, selected, bounds = render_peyer(
            image=image, rows=rows, condition=condition, center_um=center,
            zoom_um=window["width_um"], pixel_size_um=PIXEL_SIZE_UM, stem=stems[condition], dpi=args.dpi,
        )
        if len(selected) != expected_count:
            raise RuntimeError(f"Changed frozen {condition} window membership: {len(selected)}")
        selected.to_csv(tables[condition], sep="\t", index=False)
        component_records.append({"condition": condition, "fixed_center_um": list(center),
                                  "bounds_um": list(bounds), "cell_count": len(selected),
                                  "width_um": window["width_um"], "bounds_px": window["bounds_px"],
                                  "positive_calls_at_0p5": int(selected.peyer_probability.ge(0.5).sum()),
                                  "probability_range_audit": probability_range(selected.peyer_probability),
                                  "component_order": [f"{condition} IF image", f"{condition} Peyer probability"],
                                  "output_stem": str(stems[condition])})
    write_provenance(
        provenance, script=Path(__file__),
        inputs=[file_record(RELEASE_POINTER, "active_classifier_release_pointer"),
                file_record(active_path, "representation_classifier_release"),
                file_record(release_path, "representation_classifier_window_release"), file_record(predictions, "frozen_binary_classifier_predictions"),
                file_record(window_path, "frozen_window_geometry_and_cell_identities"),
                file_record(summary, "frozen_inference_summary"),
                file_record(image_path, "frozen_IF_DAPI_cache"),
                file_record(checkpoint, "representation_pretrained_Peyer_classifier_checkpoint"),
                file_record(REPO / "Figure_4_Extended/peyer.py", "Peyer_rendering_helper"),
                file_record(DISPLAY_CONFIG, "shared_Peyer_probability_display"),
                file_record(REPO / "Figure_4_Extended/probability_display.py", "shared_Peyer_probability_display_helper"),
                file_record(REPO / "Figure_4/common.py", "PDF_export_helper")],
        outputs=outputs,
        extra={"panel": "Extended Figure 4g", "lineage": "representation-pretrained axis-architecture Peyer classifier",
               "qualitative_only": True, "components": component_records,
               "component_order": ["DMSO IF image", "DMSO Peyer probability", "RARi IF image", "RARi Peyer probability"],
               "import_layout": "two separate importable pairs, DMSO then RARi; common probability limits",
               "window_widths_um": {w["condition"]: w["width_um"] for w in windows}, "source_pixel_size_um": PIXEL_SIZE_UM,
               "model_inference_run": False, "pdf_prediction_markers": "vector", "pdf_colorbar": "vector",
               "pdf_microscopy": "native source raster, unchanged", "png_dpi": args.dpi,
               "probability_colormap": display["colormap"], "probability_color_limits": display["limits"],
               "colorbar_ticklabels": display["ticklabels"], "color_saturation_explicitly_marked": True,
               "classifier_threshold": 0.5, "predictions_rescaled": False,
               "prediction_marker_area_pt2": 5.5, "random_seed": None,
               "warnings": ["Qualitative separate-classifier IF visualization; no direct IF Peyer labels."],
               "inference_summary": json.loads(summary.read_text())},
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
