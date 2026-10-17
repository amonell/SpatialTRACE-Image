#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from common import file_record, protect_outputs, write_provenance, verify_peyer_component, require_frozen_input  # noqa: E402


def save(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return [pdf, png]


def clean_spatial(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 1c graph-model input/output components.")
    parser.add_argument("--sparse-labels", type=Path, required=True)
    parser.add_argument("--training-graph-cells", type=Path, required=True)
    parser.add_argument("--training-region-lock", type=Path, required=True)
    parser.add_argument("--peyer-cells", type=Path, required=True)
    parser.add_argument("--peyer-manifest", type=Path, required=True)
    parser.add_argument("--frozen-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    peyer_release = verify_peyer_component(args.peyer_cells, args.peyer_manifest)
    out = args.output_dir.expanduser().resolve()
    names = ("panel_C_sparse_axis_graph", "panel_C_coordinate_output", "panel_C_peyer_probability_output")
    outputs = [out / f"{name}{suffix}" for name in names for suffix in (".pdf", ".png")]
    provenance = out / "panel_C_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)
    labels = pd.read_parquet(require_frozen_input(args.sparse_labels))
    villus = labels[(labels.batch.astype(str) == "day90_SI_r2") & labels.training_villus_id.astype(str).eq("day90_SI_r2__villus_004")].copy().reset_index(drop=True)
    training = pd.read_parquet(require_frozen_input(args.training_graph_cells))
    training_lock = json.loads(require_frozen_input(args.training_region_lock).read_text())
    limits = training_lock["regions"]["paired_training_example"]["limits_um_xyxy"]
    peyer = pd.read_parquet(args.peyer_cells)

    tree = cKDTree(villus[["x", "y"]])
    _, neighbors = tree.query(villus[["x", "y"]], k=min(4, len(villus)))
    segments = []
    for source, targets in enumerate(neighbors[:, 1:]):
        for target in targets:
            if source < target:
                segments.append([villus.loc[source, ["x", "y"]].to_numpy(float), villus.loc[int(target), ["x", "y"]].to_numpy(float)])
    fig, ax = plt.subplots(figsize=(3.0, 5.2)); ax.add_collection(LineCollection(np.asarray(segments), colors="#414141", linewidths=.25, alpha=.35)); ax.scatter(villus.x, villus.y, c=villus.crypt_villus_label, cmap="viridis", norm=Normalize(0, 1), s=7, linewidths=0); clean_spatial(ax); rendered = save(fig, out / names[0], args.dpi)

    fig, ax = plt.subplots(figsize=(4.2, 3.5)); ax.scatter(training.x, training.y, c=training.crypt_villus_prediction, cmap="viridis", norm=Normalize(0,1), s=1.8, linewidths=0, rasterized=True); clean_spatial(ax)
    x0,x1,y0,y1=limits; ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); rendered += save(fig, out / names[1], args.dpi)
    fig, ax = plt.subplots(figsize=(4.2, 4.2)); ax.scatter(peyer.x_spatial, peyer.y_spatial, c=peyer.gat_peyer_probability, cmap="magma", norm=Normalize(0,1), s=2.0, linewidths=0, rasterized=True); clean_spatial(ax); rendered += save(fig, out / names[2], args.dpi)
    inputs = [file_record(args.peyer_manifest, 'corrected_label_Peyer_window'),file_record(args.sparse_labels,"sparse_labels"),file_record(args.training_graph_cells,"paired_training_graph_reference"),file_record(args.training_region_lock,"shared_C_D_training_region"),file_record(args.peyer_cells,"peyer_region_cells"),file_record(args.frozen_registry,"frozen_registry")]
    write_provenance(provenance, script=HERE, inputs=inputs, outputs=rendered, extra={"panel":"Figure 1c","rendered_from_numeric_inputs":True,"model_outputs_are_frozen":True,"coordinate_limits_um_xyxy":limits,"coordinate_rows":len(training),"conceptual_diagrams":"The complete page preserves original manual neighbor/expression diagrams. Historical numeric alternatives are not active components and are not regenerated."}, overwritten=overwritten)
    print(provenance)


if __name__ == "__main__":
    main()
