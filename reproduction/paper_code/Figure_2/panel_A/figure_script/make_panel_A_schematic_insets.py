#!/usr/bin/env python3
"""Render each numerical Figure 2A inset from locked cell tables.

The component runner and complete-page adapter call this same renderer.
Manual schematic artwork is separate; existing plots are never input data.
"""
from __future__ import annotations

import argparse
import json
from paper_paths import Path, lock_record, rendering_script
import sys

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_2.common import file_record, protect_outputs, require_file, require_graph_inputs, write_provenance
from Figure_2_Extended_2.common import PREPARED_MANIFEST_SHA256

STEMS = ("panel_A_sparse_coordinate_example", "panel_A_dense_coordinate_example", "panel_A_peyer_classifier_example")


def coordinate_example(zoom, selected, bounds, *, predicted):
    figure = plt.figure(figsize=(3.8, 2.3))
    axis = figure.add_axes([.015, .04, .86, .92])
    if predicted:
        axis.scatter(zoom.x, zoom.y, c=zoom.crypt_villus_prediction, cmap="viridis", vmin=0, vmax=1, s=.12, linewidths=0, rasterized=True)
    else:
        axis.scatter(zoom.x, zoom.y, c="#CFCFCF", s=.09, linewidths=0, rasterized=True)
        axis.scatter(selected.x, selected.y, c=selected.crypt_villus_label, cmap="viridis", vmin=0, vmax=1, s=.45, linewidths=0, rasterized=True)
    axis.set_xlim(bounds["x_min"], bounds["x_max"])
    axis.set_ylim(bounds["y_max"], bounds["y_min"])
    axis.set_aspect("auto")
    axis.axis("off")
    cbar = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), cax=figure.add_axes([.91, .11, .025, .78]))
    cbar.set_ticks([])
    cbar.outline.set_linewidth(.4)
    return figure


def classifier_example(cells):
    figure = plt.figure(figsize=(3, 3))
    axis = figure.add_axes([.015, .015, .97, .97])
    positive = cells.gat_peyer_class.eq(1)
    axis.scatter(cells.loc[~positive, "x_spatial"], cells.loc[~positive, "y_spatial"], c="#BCB7C0", s=.07, linewidths=0, rasterized=True)
    axis.scatter(cells.loc[positive, "x_spatial"], cells.loc[positive, "y_spatial"], c="#7652A7", s=.11, linewidths=0, rasterized=True)
    axis.set_aspect("auto")
    axis.invert_yaxis()
    axis.axis("off")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("zoom-cells", "sparse-labels", "input-manifest", "peyer-cells", "peyer-manifest", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    graph = require_graph_inputs(args.input_manifest, [args.zoom_cells, args.sparse_labels])
    peyer_manifest = require_file(args.peyer_manifest, PREPARED_MANIFEST_SHA256)
    peyer = json.loads(peyer_manifest.read_text())
    require_file(args.peyer_cells, peyer["prepared_outputs"]["panel_c_window_cells"]["sha256"])
    zoom, labels = pd.read_parquet(args.zoom_cells), pd.read_parquet(args.sparse_labels)
    selected = labels[labels.batch.astype(str).eq("day90_SI_r2") & labels.training_villus_id.astype(str).isin(graph["zoom_villi"])]
    cells = pd.read_parquet(args.peyer_cells)
    threshold = float(peyer["model_architecture"]["hard_call_threshold"])
    if not (cells.gat_peyer_probability.ge(threshold).astype(int) == cells.gat_peyer_class).all():
        raise RuntimeError("Frozen schematic classifier calls do not match their threshold")
    output = args.output_dir.expanduser().resolve()
    files = [output / (stem + suffix) for stem in STEMS for suffix in (".pdf", ".png")]
    provenance = output / "panel_A_schematic_insets_provenance.json"
    overwritten = protect_outputs([*files, provenance], args.overwrite)
    figures = [coordinate_example(zoom, selected, graph["zoom_limits_um"], predicted=False),
               coordinate_example(zoom, selected, graph["zoom_limits_um"], predicted=True), classifier_example(cells)]
    for figure, stem in zip(figures, STEMS, strict=True):
        for suffix in (".pdf", ".png"):
            figure.savefig(output / (stem + suffix), transparent=True, bbox_inches="tight", dpi=args.dpi)
        plt.close(figure)
    write_provenance(provenance, script=__file__,
        inputs=[file_record(args.zoom_cells, "frozen_graph_coordinate_zoom"), file_record(args.sparse_labels, "manual_coordinate_labels"),
                file_record(args.input_manifest, "locked_graph_component_manifest"), file_record(args.peyer_cells, "frozen_corrected_GAT_test_window"),
                file_record(peyer_manifest, "locked_corrected_GAT_component_manifest")], outputs=files, overwritten=overwritten,
        extra={"panel": "Figure 2a", "component_order": list(STEMS), "coordinate_bounds_um": graph["zoom_limits_um"],
               "coordinate_zoom_villi": graph["zoom_villi"], "classifier_threshold": threshold,
               "classifier_window_cells": len(cells), "model_training_or_inference_run": False,
               "coordinate_reference": "sparse manual LabelMe coordinates", "coordinate_prediction": "frozen all-label graph visualization",
               "classifier_lineage": "separate frozen corrected-label BinaryGAT; calls from its locked test window"})


if __name__ == "__main__":
    main()
