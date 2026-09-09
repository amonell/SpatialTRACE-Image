"""Regenerate main Figure4 scientific layers at measured September5 anchors."""
from __future__ import annotations

import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle

from Figure_4.assembly.common import (PageInputs, HOME, SATA, REPAIR, PAGE, PIXEL_UM,
                                      colorbar, imap, top_text)
from Figure_4.spatial import GATE_COLORS, GATE_ORDER, common_tissue_limits
from Figure_4.panel_D.figure_script.make_panel_D_whole_tissue_maps import load_zooms

LAYOUT = Path(__file__).with_name("layout.json")


def microscopy(page):
    lockpath = HOME / "Figure_4/panel_A/processing/figure4a_region_lock.json"
    lock = json.loads(page.read(lockpath, "bf6c3064da8f7e0c97facb1bb4bac430c5d0fc1e58a88f57a37ad3e8af9720f5", "microscopy_region_lock").read_text())
    calibration = json.loads(page.read(HOME / "Figure_4/panel_A/processing/figure4a_scale_calibration.json",
                            "7acc25d5ae4bbdb3e4562aa91166ca67f50ded5798850eb046db6c8bcceec4e8", "snapshot_scale_calibration").read_text())
    figure = page.figure()
    rects = {"DMSO": [28.15, 36.48, 146.49, 141.44], "RARi": [176.18, 36.89, 146.33, 140.62]}
    footer = page.axis(figure, "A_shared_footer", [28.15, 177.92, 294.36, 17.76])
    footer.set_facecolor("#000000")
    footer.add_patch(Rectangle((0,0),1,1,transform=footer.transAxes,facecolor="black",edgecolor="none"))
    footer.set(xticks=[], yticks=[])
    for spine in footer.spines.values():
        spine.set_visible(False)
    for condition, rect in rects.items():
        spec = lock["regions"][condition]
        source = page.read(SATA / "Data/Histology/histology_snapshots" / ("DMSO.png" if condition == "DMSO" else "RAR.png"),
                           spec["source_sha256"], condition + "_original_RGB")
        snapshot = Image.open(source).convert("RGB")
        if list(snapshot.size) != spec["source_size_px"]:
            raise RuntimeError("Snapshot dimensions changed")
        crop = snapshot.crop(spec["source_crop_xyxy"])
        if spec["rotation_degrees"] == 180:
            crop = crop.transpose(Image.Transpose.ROTATE_180)
        elif spec["rotation_degrees"] != 0:
            raise RuntimeError("Unregistered microscopy rotation")
        axis = page.axis(figure, f"A_{condition}_image", rect)
        axis.imshow(crop, interpolation="none", aspect="auto")
        axis.axis("off")
        bar_width = rect[2] * calibration["rendered_bar_length_snapshot_px"] / crop.width
        bx = rect[0] + 5
        figure.add_artist(Line2D([bx / PAGE[0], (bx + bar_width) / PAGE[0]],
                                 [1 - 181.4 / PAGE[1]] * 2, transform=figure.transFigure,
                                 lw=1.4, color="white", solid_capstyle="butt"))
        top_text(figure, bx + bar_width + 3, 184, "50 µm", color="white", fontsize=6)
        top_text(figure, 88.1172 if condition=="DMSO" else 241.7139, 29.6605,
                 condition, fontsize=8, fontweight="bold")
        page.details["microscopy_" + condition] = {"source_crop_xyxy": spec["source_crop_xyxy"],
            "rotation_degrees": spec["rotation_degrees"], "source_RGB_preserved": True,
            "snapshot_bar_native_pixels": 79, "bar_length_pt": bar_width, "bar_um": 50}
    for text, x, color in (("DAPI", 103, "#4658ff"), ("E-Cadherin", 130, "#fa2bee"),
                           ("CD8α", 183, "#63e61a"), ("CD45.1", 218, "#ffff00")):
        top_text(figure, x, 193.2, text, color=color, fontsize=8)
    page.warnings.append("Original RGB snapshots retained; unregistered Illustrator color conversion was not reproduced.")
    page.save(figure, "panel_A_microscopy")


def sparse_labels(page):
    example = json.loads(page.read(REPAIR / "illustration_inputs/manifest.json",
                  "7bb07db00ed5be33d127420f80f9c914f1925ff5dbc24246ef0bf3d29353f8a2", "prespecified_IF_illustration").read_text())
    if example["selection_uses_model_predictions"] or example["selected_cells"] != 749:
        raise RuntimeError("Unrecognized frozen training illustration")
    row_record = example["sparse_cells"]
    rows = pd.read_csv(page.read(row_record["path"], row_record["sha256"], "sparse_training_cells"))
    win = example["windows"]["sparse_training"]
    image = np.load(page.read(win["DAPI"]["path"], win["DAPI"]["sha256"], "sparse_training_DAPI"))
    if len(rows) != 749 or not rows.split.eq("train").all():
        raise RuntimeError("Sparse example is not the retained training set")
    x0, x1, y0, y1 = win["bounds_xxyy_px"]
    height = 153.13 * image.shape[0] / image.shape[1]
    figure = page.figure()
    axis = page.axis(figure, "B_image", [378.5, 35.21, 153.13, height])
    axis.imshow(image, extent=(x0, x1, y1, y0), cmap="gray", vmin=win["display_vmin"],
                vmax=win["display_vmax"], interpolation="none", aspect="auto")
    artist = axis.scatter(rows.centroid_x_fullres_px, rows.centroid_y_fullres_px,
                           c=rows.target_axis, cmap="viridis", vmin=0, vmax=1,
                           s=1.7, linewidths=0, rasterized=True)
    axis.set(xlim=(x0, x1), ylim=(y1, y0))
    axis.axis("off")
    top_text(figure, 455.1, 31.9, "Example IF training annotations", ha="center", fontsize=8.5, fontweight="bold")
    cax = page.axis(figure, "B_reference_bar", [534.25, 47.21, 5.62, height - 24])
    colorbar(figure, cax, artist, ticks=[0, 1], label="Crypt-villus coordinate")
    bar_pt = 153.13 * (200 / example["pixel_size_um"]) / (x1 - x0)
    bx, by = 384.5, 35.21 + height + 5.5
    figure.add_artist(Line2D([bx / PAGE[0], (bx + bar_pt) / PAGE[0]],
                             [1 - by / PAGE[1]] * 2, transform=figure.transFigure,
                             color="black", lw=1.8, solid_capstyle="butt"))
    top_text(figure, bx + bar_pt + 4, by + 2.2, "200 µm", fontsize=7)
    page.details["sparse_example"] = {"cells": 749, "villi": example["selected_villi"],
         "bounds_xxyy_px": win["bounds_xxyy_px"], "native_aspect_preserved": True,
         "authorized_image_height_pt": height, "bar_length_pt": bar_pt,
         "bar_um": 200, "external_bar": True}
    page.save(figure, "panel_B_sparse_labels")


def strict_scatter(page):
    rows = pd.read_parquet(page.frozen("strict_test/predictions.parquet"))
    metrics = json.loads(page.frozen("strict_test/metrics.json").read_text())
    if len(rows) != 4843 or rows.row_id.duplicated().any() or not rows.eligible_final_test.all():
        raise RuntimeError("Unexpected strict-test membership")
    figure = page.figure()
    specs = [("C_crypt_villus", [54.721, 233.018, 152.063, 152.064],
              "target_axis", "predicted_axis_coordinate", "Crypt-villus", "viridis", 213.48),
             ("C_epithelial_distance", [312.121, 233.018, 152.064, 152.064],
              "epithelial_distance_clipped_1p0", "predicted_epithelial_distance_clipped_1p0",
              "Epithelial distance", "coolwarm", 470.92)]
    for ident, rect, ref, pred, key, cmap, cbx in specs:
        axis = page.axis(figure, ident, rect)
        artist = axis.hexbin(rows[ref], rows[pred], gridsize=45, mincnt=1, cmap=cmap,
                             bins="log", rasterized=True)
        axis.plot([0, 1], [0, 1], color="#555555", ls="--", lw=.75)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Sparse IF reference", ylabel="Image-model prediction")
        top_text(figure,74.394 if key=="Crypt-villus" else 322.0791,227.0179,
                 f"Held-out {key.lower()} predictions",fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        result = metrics[key]
        axis.text(.04, .95, f"Pearson r = {result['pearson_r']:.3f}\nMAE = {result['mae']:.3f}\nR² = {result['r2']:.3f}",
                  transform=axis.transAxes, va="top", fontsize=7.5, linespacing=1.2,
                  bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": .96, "pad": 2})
        cax = page.axis(figure, ident + "_bar", [cbx, 232.87, 7.5, 152.16])
        colorbar(figure, cax, artist, label="Cell count")
    page.details["strict_test"] = {"n": len(rows), "metrics_source": "frozen_metrics_json",
                                   "displayed_metrics": ["pearson_r", "mae", "r2"]}
    page.save(figure, "panel_C_strict_scatter")


def whole_maps(page):
    page.read(HOME / "Figure_4/panel_D/figure_script/make_panel_D_whole_tissue_maps.py", role="locked_zoom_loading_helper")
    config = page.read(HOME / "Figure4/imap_if/config_all_cells_production_v2_if_sft.yaml",
                       "6447d3cc3ea0e4318be13db878fd0a68da0ad4981562721f33a7e885587dce3d", "locked_IF_zoom_geometry")
    pixel, zooms = load_zooms(config)
    frames = {c: pd.read_parquet(page.frozen(f"whole_tissue/{c.lower()}_predictions.parquet")) for c in ("DMSO", "RARi")}
    limits = common_tissue_limits(frames)
    if [len(frames[c]) for c in ("DMSO", "RARi")] != [163992, 136437]:
        raise RuntimeError("Whole-tissue cell counts changed")
    specs = [
        ("DMSO", "predicted_axis_coordinate", "viridis", 1, [20.37,445.78,74.5,87.56], [16.578,535.401,82.115,61.586], [21.29,603.59,69.6,3.57]),
        ("DMSO", "predicted_epithelial_distance_clipped_1p0", "coolwarm", .5, [108.54,445.04,76.24,87.51], [104.64,534.855,84.045,61.553], [113.68,602.56,69.59,3.57]),
        ("RARi", "predicted_axis_coordinate", "viridis", 1, [206.52,461.18,69.1,72.73], [201.165,535.294,79.834,59.875], [208.53,601.84,69.6,3.57]),
        ("RARi", "predicted_epithelial_distance_clipped_1p0", "coolwarm", .5, [292.16,460.87,69.1,72.73], [286.781,535.606,79.834,59.875], [292.68,601.16,69.6,3.57])]
    figure = page.figure()
    for index, (condition, column, cmap, vmax, wr, zr, cr) in enumerate(specs):
        frame = frames[condition]
        whole_slot = page.axis(figure, f"D_whole_frame_{index}", wr)
        whole_slot.axis("off")
        whole = page.axis(figure, f"D_whole_{index}", wr)
        whole.scatter(frame.x_um, frame.y_um, c=frame[column], cmap=cmap, vmin=0, vmax=vmax,
                      s=.055, linewidths=0, rasterized=True)
        xmin, xmax, ymin, ymax = limits[condition]
        whole.set(xlim=(xmin, xmax), ylim=(ymax, ymin), aspect="equal")
        whole.axis("off")
        x0, x1, y0, y1 = [v * pixel for v in zooms[condition]]
        whole.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, lw=.9, edgecolor="#333333"))
        # Keep the original frame fixed. A separate equal-aspect inner axis
        # contains the complete locked6000×4500 field, with no recropping.
        border = page.axis(figure, f"D_zoom_frame_{index}", zr)
        border.set(xticks=[], yticks=[])
        border.patch.set_alpha(0)
        zw, zh = min(zr[2], zr[3] * 4 / 3), min(zr[3], zr[2] * 3 / 4)
        inner = [zr[0] + (zr[2]-zw)/2, zr[1]+(zr[3]-zh)/2, zw, zh]
        zoom = page.axis(figure, f"D_zoom_content_{index}", inner)
        subset = frame.loc[frame.x_um.between(x0,x1) & frame.y_um.between(y0,y1)]
        artist = zoom.scatter(subset.x_um, subset.y_um, c=subset[column], cmap=cmap,
                               vmin=0, vmax=vmax, s=.65, linewidths=0, rasterized=True)
        zoom.set(xlim=(x0,x1), ylim=(y1,y0), aspect="equal")
        zoom.axis("off")
        for xx in (x0,x1):
            figure.add_artist(ConnectionPatch(xyA=(xx,y1), coordsA=whole.transData,
                xyB=(xx,y0), coordsB=zoom.transData, lw=1.25, color="#333333", clip_on=False))
        cax = page.axis(figure, f"D_colorbar_{index}", cr)
        cb = colorbar(figure,cax,artist,horizontal=True,ticks=[0,vmax])
        cb.ax.set_xticklabels(["0", "1.0" if vmax==1 else "≥0.5"],fontsize=6)
        cb.set_label("Predicted crypt-\nvillus coordinate" if vmax==1 else "Predicted epithelial-\ndistance coordinate", fontsize=6.5, labelpad=2)
        page.details[f"whole_map_{index}"] = {"cells":len(frame), "zoom_cells":len(subset),
          "bounds_xxyy_px":list(zooms[condition]), "physical_aspect_preserved":True,
          "frame_rect_pt":zr,"contained_image_rect_pt":inner,"no_display_downsampling":True}
    figure.canvas.draw()
    for index,x,y in [(0,22.4,449.0),(2,206.3,464.2)]:
        axis = next(a for a in figure.axes if getattr(a,"_assembly_id",None)==f"D_whole_{index}")
        p = axis.get_position()
        width = 1000 / (axis.get_xlim()[1]-axis.get_xlim()[0]) * p.width * PAGE[0]
        figure.add_artist(Line2D([x/PAGE[0],(x+width)/PAGE[0]], [1-y/PAGE[1]]*2,
                                 transform=figure.transFigure,color="black",lw=.7))
        top_text(figure,x+width/2,y+5,"1 mm",ha="center",fontsize=5)
    top_text(figure,93.918,449.1214,"DMSO",fontsize=6.3,fontweight="bold")
    top_text(figure,277.0742,449.2093,"RARi",fontsize=6.3,fontweight="bold")
    page.save(figure, "panel_D_whole_maps")


def abundance(page):
    a = pd.read_csv(page.frozen("imap/p14_abundance_by_gate.tsv"),sep="\t")
    c = pd.read_csv(page.frozen("imap/p14_composition_by_gate.tsv"),sep="\t")
    wide = a.pivot(index="gate",columns="condition",values="p14_per_10000_non_p14_same_gate").reindex(GATE_ORDER)
    counts = a.pivot(index="gate",columns="condition",values="p14_count").reindex(GATE_ORDER)
    comp = c.pivot(index="gate",columns="condition",values="percent").reindex(GATE_ORDER)
    delta = comp.DMSO-comp.RARi
    colors={"DMSO":"#4F7F8A","RARi":"#A55D54"}
    figure=page.figure()
    left=page.axis(figure,"E_abundance",[49.495,669.367,148.248,132.912])
    right=page.axis(figure,"E_composition",[206.117,669.366,148.248,132.912])
    maximum=float(wide.max().max())
    left.set_xlim(0,maximum*1.5)
    for y,gate in enumerate(GATE_ORDER):
        d,r=wide.loc[gate,"DMSO"],wide.loc[gate,"RARi"]
        left.plot([r,d],[y,y],color=GATE_COLORS[gate],lw=1.8)
        for condition,value in [("DMSO",d),("RARi",r)]:
            left.scatter(value,y,s=16,color=colors[condition],edgecolor="#222222",lw=.3,zorder=3)
        left.text(max(d,r)+maximum*.025,y,f"{counts.loc[gate,'DMSO']:,} / {counts.loc[gate,'RARi']:,}",fontsize=6,va="center")
    handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor=colors[c],markeredgecolor="#333333",markersize=3.5,label=c) for c in colors]
    left.legend(handles=handles,frameon=False,loc="lower right",fontsize=6.5,handletextpad=.4,labelspacing=.2)
    left.set_xticks([0,1000,2000])
    left.set_xlabel("P14 per 10,000 non-P14 cells\nin the same gate\n(raw counts: DMSO / RARi)",fontsize=6.5,labelpad=2)
    top_text(figure,70.3339,664.5699,"Gate-specific P14 abundance",fontweight="bold",fontsize=8)
    lim=max(5,float(abs(delta).max())*1.3)
    right.axvspan(-lim,0,color="#f6eeee");right.axvspan(0,lim,color="#eef4f5")
    right.axvline(0,color="#333333",lw=.7)
    for y,gate in enumerate(GATE_ORDER):
        value=delta[gate]
        right.plot([0,value],[y,y],color=GATE_COLORS[gate],lw=1.8)
        right.scatter(value,y,s=16,color=colors["DMSO"] if value>=0 else colors["RARi"],edgecolor="#222222",lw=.3,zorder=3)
        right.text(value+(lim*.025 if value>=0 else -lim*.025),y,f"{value:+.1f}",ha="left" if value>=0 else "right",va="center",fontsize=6.5)
    right.set_xlim(-lim,lim)
    right.set_xlabel("DMSO − RARi (percentage points)",fontsize=7,labelpad=2)
    top_text(figure,227.4109,664.5659,"Within-P14 composition shift",fontweight="bold",fontsize=8)
    for axis in [left,right]:
        labels=list(GATE_ORDER)
        axis.set_ylim(4.2,-.2);axis.set_yticks(range(5),labels if axis is left else [])
        axis.grid(axis="x",color="#dddddd",lw=.45);axis.set_axisbelow(True)
        axis.spines[["top","right","left"]].set_visible(False);axis.tick_params(axis="y",length=0)
    page.details["abundance"]={"denominator":"non-P14 cells in the same gate", "raw_P14_counts_displayed":True,
        "composition_delta":"DMSO minus RARi percentage points", "source":"frozen tables"}
    page.save(figure,"panel_E_abundance_composition")


def p14_imaps(page):
    page.frozen("imap/p14_gate_definitions.json")
    figure=page.figure()
    imap(page,figure,"DMSO","P14",[407.315,463.505,173.601,111.996],
         title_y=458.1957,title_x=493.5079,show_xticks=False)
    artist=imap(page,figure,"RARi","P14",[406.794,597.404,173.601,111.996],
         title_y=594.2054,title_x=496.0287,
         xlabel=True,xlabel_anchor=(448.1605,732.457))
    cax=page.axis(figure,"F_shared_density_bar",[447.572,743.835,98.556,4.928])
    colorbar(figure,cax,artist,horizontal=True,ticks=[0,5,10,15],label="Probability density")
    page.save(figure,"panel_F_P14_IMAPs")


def prepare(output_dir: Path, overwrite=False):
    page=PageInputs(output_dir,overwrite)
    layout=json.loads(LAYOUT.read_text())
    page.read(layout["reference"],layout["reference_sha256"],"September5_layout_reference_only")
    microscopy(page);sparse_labels(page);strict_scatter(page);whole_maps(page);abundance(page);p14_imaps(page)
    for letter,x,y in [("a.",11.16,21.2),("b.",363.79,21.31),("c.",10.63,216.25),
                        ("d.",14.94,438.61),("e.",15.23,654.59),("f.",367.52,438.72)]:
        page.text(letter,x,y,size=10.2,panel=letter[0].upper())
    return page.finish(LAYOUT,Path(__file__))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--overwrite",action="store_true")
    a=p.parse_args();prepare(a.output_dir,a.overwrite)
