#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from common import file_record, protect_outputs, write_provenance, verify_peyer_component, require_frozen_input, load_source_regions  # noqa: E402


def save(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return [pdf, png]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 1b sparse axis and region labels.")
    parser.add_argument("--zoom-cells", type=Path, required=True)
    parser.add_argument("--sparse-labels", type=Path, required=True)
    parser.add_argument("--peyer-cells", type=Path, required=True)
    parser.add_argument("--peyer-manifest", type=Path, required=True)
    parser.add_argument("--source-region-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    peyer_release = verify_peyer_component(args.peyer_cells, args.peyer_manifest)
    out = args.output_dir.expanduser().resolve()
    stems = [out / "panel_B_continuous_sparse_label", out / "panel_B_peyer_region_label"]
    outputs = [stem.with_suffix(suffix) for stem in stems for suffix in (".pdf", ".png")]
    provenance = out / "panel_B_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)
    regions = load_source_regions(args.source_region_lock)
    region = regions["regions"]["sparse_axis"]
    zoom = pd.read_parquet(require_frozen_input(args.zoom_cells))
    labels = pd.read_parquet(require_frozen_input(args.sparse_labels))
    labels = labels[(labels.batch.astype(str) == "day90_SI_r2") & labels.training_villus_id.astype(str).isin(region["visible_villus_ids"])]
    peyer = pd.read_parquet(args.peyer_cells)
    fig, ax = plt.subplots(figsize=(4.2, 3.5)); ax.scatter(zoom.x, zoom.y, c="#c8c8c8", s=1.2, linewidths=0, rasterized=True); ax.scatter(labels.x, labels.y, c=labels.crypt_villus_label, cmap="viridis", norm=Normalize(0, 1), s=8, linewidths=0, rasterized=True); ax.set_aspect("equal")
    x0,x1,y0,y1 = region["limits_um_xyxy"]; ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); ax.axis("off"); rendered = save(fig, stems[0], args.dpi)
    fig, ax = plt.subplots(figsize=(4.2, 4.2)); ax.scatter(peyer.x_spatial, peyer.y_spatial, c="#c8c8c8", s=1.2, linewidths=0, rasterized=True); positive = peyer.peyer_manual_label.fillna(0).astype(float) > 0; ax.scatter(peyer.loc[positive, "x_spatial"], peyer.loc[positive, "y_spatial"], c="#7E57C2", s=4, linewidths=0, rasterized=True); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); rendered += save(fig, stems[1], args.dpi)
    inputs = [file_record(args.peyer_manifest, 'corrected_label_Peyer_window'), file_record(args.zoom_cells, "zoom_cells"), file_record(args.sparse_labels, "sparse_labels"), file_record(args.peyer_cells, "peyer_region_cells")]
    inputs.append(file_record(args.source_region_lock, "artwork_registered_source_regions"))
    write_provenance(provenance, script=HERE, inputs=inputs, outputs=rendered, extra={"panel": "Figure 1b", "peyer_split": sorted(peyer.split.astype(str).unique().tolist()), "axis_limits_um_xyxy": region["limits_um_xyxy"], "displayed_annotation_villus_ids": region["visible_villus_ids"], "annotation_scope": "Only the original illustrated annotation is colored; other labeled villi can lie inside this example window."}, overwritten=overwritten)
    print(provenance)


if __name__ == "__main__":
    main()
