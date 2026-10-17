#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from paper_paths import Path, lock_record, rendering_script

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, protect_outputs, require_file, write_provenance, repaired_reference  # noqa: E402
from spatial_maps import render_map  # noqa: E402
from figure_inputs import locked_zoom, DEFAULT_ZOOM_RECORD  # noqa: E402

EXPECTED_CHECKPOINT = "d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137"
EXPECTED_SPLIT = "f707d0598722c0381c552ad9f4803174de8652e56d8adf99d64cf8bed61994e2"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--reference-manifest", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--split-manifest", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--dpi", type=int, default=300); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    predictions = repaired_reference(args.predictions, args.reference_manifest, "predictions_current_references.parquet"); checkpoint = require_file(args.checkpoint, EXPECTED_CHECKPOINT); split = require_file(args.split_manifest, EXPECTED_SPLIT)
    frame = pd.read_parquet(predictions)
    if len(frame) != 50_000 or frame.row_id.nunique() != 50_000: raise RuntimeError("Panel E requires all 50,000 locked sample_008 cells")
    out = args.output_dir.expanduser().resolve(); names = ("panel_E_epithelial_distance_reference", "panel_E_epithelial_distance_prediction", "panel_E_epithelial_distance_reference_zoom", "panel_E_epithelial_distance_prediction_zoom"); outputs = [out / f"{name}.{suffix}" for name in names for suffix in ("pdf", "png")]; provenance = out / "panel_E_provenance.json"; overwritten = protect_outputs(outputs + [provenance], args.overwrite)
    zoom = locked_zoom(DEFAULT_ZOOM_RECORD); rendered = []
    rendered += render_map(frame, column="epithelial_distance_clipped_1p0", title="Graph-derived reference", cmap="coolwarm", vmax=.7, stem=out / names[0], dpi=args.dpi, limits=None, show_box=zoom, dot_size=1.3)
    rendered += render_map(frame, column="predicted_epithelial_distance_clipped_1p0", title="Image-model prediction", cmap="coolwarm", vmax=.7, stem=out / names[1], dpi=args.dpi, limits=None, show_box=zoom, dot_size=1.3)
    rendered += render_map(frame, column="epithelial_distance_clipped_1p0", title="Reference zoom", cmap="coolwarm", vmax=.7, stem=out / names[2], dpi=args.dpi, limits=zoom, show_box=None, dot_size=5)
    rendered += render_map(frame, column="predicted_epithelial_distance_clipped_1p0", title="Prediction zoom", cmap="coolwarm", vmax=.7, stem=out / names[3], dpi=args.dpi, limits=zoom, show_box=None, dot_size=5)
    write_provenance(provenance, script=Path(__file__), inputs=[file_record(DEFAULT_ZOOM_RECORD,"locked_zoom_geometry"), file_record(ROOT/'figure_inputs.py','shared_input_logic'), file_record(args.reference_manifest, "current_graph_reference_release"), file_record(predictions, "sample008_production_v2_predictions"), file_record(checkpoint, "released_production_v2_checkpoint"), file_record(split, "locked_sample008_source")], outputs=rendered, extra={"panel": "Figure 3e", "source_id": "sample_008", "cells_displayed_per_full_map": 50000, "display_downsampling": False, "zoom_limits_fullres_px": list(zoom), "shared_color_limits": [0, .7]}, overwritten=overwritten)


if __name__ == "__main__": main()
