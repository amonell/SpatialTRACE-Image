#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import file_record, protect_outputs, require_file, save_pair, write_provenance  # noqa: E402

BATCHES = ("day6_SI_r2", "day8_SI_Ctrl", "day30_SI_r2")
TITLES = ("Day 6 SI", "Day 8 SI", "Day 30 SI")
EXPECTED_SECTIONS = "745647147c9de41821445cea5eeb08f13422594191f55589fc24bb7a1db3fde5"
EXPECTED_LABELS = "7fd1c37966880be09e6fd402e810b7199358b4f9e20bad31da54abbcca522843"
EXPECTED_ZOOMS = "23a62f690d722b1302667b4fed47742d5fe89487b270d47e46e119460f2b899e"
EXPECTED_REFERENCE_ZOOMS = "24814d120fd4a1e16d405af7bce599199101b7cbf88bcbaef1f5f5b33e9afbfb"


def add_coordinate_zoom(figure, axis, cells, window, *, column, cmap, norm, annotations=None):
    """A paired, equal-aspect view with vector callout and external scale bar."""
    box=axis.get_position()
    inset=figure.add_axes([box.x0+.58*box.width,box.y0+.22*box.height,
                           .44*box.width,.56*box.height],zorder=5)
    axis.set_position([box.x0,box.y0+.13*box.height,.53*box.width,.75*box.height])
    x0,x1,y0,y1=window['bounds_xxyy_um']
    if annotations is None:
        inset.scatter(cells.x,cells.y,c=cells[column],cmap=cmap,norm=norm,
                      s=1.1,linewidths=0,rasterized=True)
    else:
        inset.scatter(cells.x,cells.y,c='#d4d4d4',s=.85,linewidths=0,rasterized=True)
        inset.scatter(annotations.x,annotations.y,c=annotations[column],cmap=cmap,norm=norm,
                      s=4.,linewidths=0,rasterized=True)
    inset.set(xlim=(x0,x1),ylim=(y1,y0),aspect='equal',xticks=[],yticks=[])
    inset.set_facecolor('white')
    for spine in inset.spines.values():spine.set(color='#444444',linewidth=.6)
    outline=Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#444444',linewidth=.65,zorder=4)
    outline.set_path_effects([path_effects.Stroke(linewidth=1.4,foreground='white'),path_effects.Normal()])
    axis.add_patch(outline)
    for source,target in [((x1,y0),(0,1)),((x1,y1),(0,0))]:
        connector=ConnectionPatch(source,target,coordsA='data',coordsB='axes fraction',
                                  axesA=axis,axesB=inset,color='#999999',linewidth=.4,zorder=3)
        figure.add_artist(connector)
    # Scale bar is below the crop, leaving every plotted cell visible.
    transform=inset.get_xaxis_transform()
    start=x0+.06*(x1-x0);length=window['scale_bar_um']
    inset.plot([start,start+length],[-.07,-.07],transform=transform,color='#333333',
               linewidth=1.2,solid_capstyle='butt',clip_on=False)
    inset.text(start+length/2,-.11,'200 µm',transform=transform,ha='center',va='top',fontsize=6)
    return inset


def render(sections: pd.DataFrame, labels: pd.DataFrame, *, column: str, label_column: str | None, cmap: str, vmax: float, stem: Path, dpi: int, zoom_cells, zoom_manifest, zoom_labels=None) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(8.0, 2.65))
    norm = Normalize(0, vmax)
    artist = None
    for axis, batch, title in zip(axes, BATCHES, TITLES, strict=True):
        cells = sections[sections["batch"].astype(str).eq(batch)]
        if cells.empty:
            raise RuntimeError(f"Missing section {batch}")
        if label_column is None:
            artist = axis.scatter(cells.x, cells.y, c=cells[column], cmap=cmap, norm=norm, s=1.1, linewidths=0, rasterized=True)
        else:
            axis.scatter(cells.x, cells.y, c="#d4d4d4", s=0.85, linewidths=0, rasterized=True)
            annotated = labels[labels["batch"].astype(str).eq(batch)]
            artist = axis.scatter(annotated.x, annotated.y, c=annotated[label_column], cmap=cmap, norm=norm, s=4.0, linewidths=0, rasterized=True)
        axis.set_title(title, fontweight="bold")
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
    colorbar = figure.colorbar(artist,cax=figure.add_axes([.35,.055,.3,.025]),orientation='horizontal')
    colorbar.set_ticks([0, vmax / 2, vmax])
    figure.subplots_adjust(left=0.01, right=0.91, bottom=0.01, top=0.90, wspace=0.04)
    for axis,batch in zip(axes,BATCHES,strict=True):
        annotated=None if label_column is None else zoom_labels.loc[zoom_labels.batch.eq(batch)]
        add_coordinate_zoom(figure,axis,zoom_cells.loc[zoom_cells.batch.eq(batch)],
                            zoom_manifest['windows'][batch],column=label_column or column,
                            cmap=cmap,norm=norm,annotations=annotated)
    rendered = save_pair(figure, stem, dpi)
    plt.close(figure)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-cells", type=Path, required=True)
    parser.add_argument("--sparse-labels", type=Path, required=True)
    parser.add_argument("--frozen-registry", type=Path, required=True)
    parser.add_argument("--zoom-manifest", type=Path, required=True)
    parser.add_argument("--reference-zoom-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    section_path = require_file(args.section_cells, EXPECTED_SECTIONS)
    label_path = require_file(args.sparse_labels, EXPECTED_LABELS)
    registry_path = require_file(args.frozen_registry)
    registry = json.loads(registry_path.read_text())
    lineage = registry["models"]["graph_coordinate_all_label_visualization"]
    from checkpoint_lineage import validate_graph_checkpoint_reference
    for item in lineage['checkpoints']:
        validate_graph_checkpoint_reference(item['path'], item['sha256'])
    for item in lineage["outputs"]:
        require_file(item["path"], item["sha256"])
    sections, labels = pd.read_parquet(section_path), pd.read_parquet(label_path)
    zoom_path=require_file(args.zoom_manifest,EXPECTED_ZOOMS)
    zoom_manifest=json.loads(zoom_path.read_text())
    if (zoom_manifest['status']!='FROZEN_GEOMETRY_SELECTED_PREDICTION_ZOOMS'
            or zoom_manifest['selection_uses_prediction_values'] or zoom_manifest['predictions_modified']):
        raise RuntimeError('Zooms must preserve the geometry-only selection and frozen predictions')
    record=zoom_manifest['outputs']['cells'];zoom_table=require_file(record['path'],record['sha256'])
    zoom_cells=pd.read_parquet(zoom_table)
    reference_path=require_file(args.reference_zoom_manifest,EXPECTED_REFERENCE_ZOOMS)
    reference_manifest=json.loads(reference_path.read_text())
    if (reference_manifest['status']!='FROZEN_REFERENCE_LABELS_IN_LOCKED_ZOOM_WINDOWS'
            or reference_manifest['labels_modified'] or reference_manifest['new_window_selection']):
        raise RuntimeError('Reference zooms must retain sparse labels in the existing windows')
    reference_record=reference_manifest['outputs']['labels']
    reference_table=require_file(reference_record['path'],reference_record['sha256'])
    zoom_labels=pd.read_parquet(reference_table)
    for batch in BATCHES:
        selected=zoom_cells.loc[zoom_cells.batch.eq(batch)]
        if len(selected)!=zoom_manifest['windows'][batch]['n']:raise RuntimeError('Zoom cell inventory changed')
        reference=reference_manifest['windows'][batch]
        if reference['bounds_xxyy_um']!=zoom_manifest['windows'][batch]['bounds_xxyy_um']:
            raise RuntimeError('Reference and prediction fields differ')
        if len(zoom_labels.loc[zoom_labels.batch.eq(batch)])!=reference['reference_n']:
            raise RuntimeError('Sparse-reference zoom inventory changed')
    out = args.output_dir.expanduser().resolve()
    names = (
        "panel_A_crypt_villus_manual_labels",
        "panel_A_crypt_villus_dense_predictions",
        "panel_A_epithelial_distance_manual_labels",
        "panel_A_epithelial_distance_dense_predictions",
    )
    outputs = [out / f"{name}.{suffix}" for name in names for suffix in ("pdf", "png")]
    provenance = out / "panel_A_provenance.json"
    overwritten = protect_outputs(outputs + [provenance], args.overwrite)
    rendered = []
    rendered += render(sections, labels, column="crypt_villus_prediction", label_column="crypt_villus_label", cmap="viridis", vmax=1.0, stem=out / names[0], dpi=args.dpi, zoom_cells=zoom_cells, zoom_manifest=zoom_manifest, zoom_labels=zoom_labels)
    rendered += render(sections, labels, column="crypt_villus_prediction", label_column=None, cmap="viridis", vmax=1.0, stem=out / names[1], dpi=args.dpi, zoom_cells=zoom_cells, zoom_manifest=zoom_manifest)
    rendered += render(sections, labels, column="epithelial_distance_prediction", label_column="epithelial_distance_label", cmap="coolwarm", vmax=0.7, stem=out / names[2], dpi=args.dpi, zoom_cells=zoom_cells, zoom_manifest=zoom_manifest, zoom_labels=zoom_labels)
    rendered += render(sections, labels, column="epithelial_distance_prediction", label_column=None, cmap="coolwarm", vmax=0.7, stem=out / names[3], dpi=args.dpi, zoom_cells=zoom_cells, zoom_manifest=zoom_manifest)
    write_provenance(
        provenance,
        script=Path(__file__),
        inputs=[file_record(section_path, "frozen_dense_predictions"), file_record(label_path, "manual_coordinate_labels"), file_record(registry_path, "frozen_model_registry"), file_record(zoom_path,'frozen_geometry_selected_zoom_manifest'), file_record(zoom_table,'frozen_zoom_cells'), file_record(reference_path,'frozen_reference_zoom_manifest'), file_record(reference_table,'frozen_zoom_reference_labels')],
        outputs=rendered,
        extra={"panel": "Extended Figure 2a", "batches": list(BATCHES), "lineage_status": lineage["status"],
               "zoom_windows":zoom_manifest['windows'],"same_windows_and_cells_for_both_coordinates":True,
               "reference_zoom_windows":reference_manifest['windows'],"references_remain_sparse":True,
               "reference_zoom_background":"Same frozen spatial cells, gray; only original annotations colored",
               "zoom_normalization_matches_whole_tissue":True,"new_inference":False,"predictions_modified":False},
        overwritten=overwritten,
    )


if __name__ == "__main__":
    main()
