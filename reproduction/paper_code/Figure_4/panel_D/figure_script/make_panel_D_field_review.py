#!/usr/bin/env python3
"""Render matched microscopy, coordinate and gate views for field selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from Figure_4.common import configure_figure_fonts
from Figure_4.spatial import GATE_COLORS, GATE_ORDER
from Figure_4.panel_D.processing.prepare_panel_D_field_candidates import record, verify, save_json


def composite(dapi, ecad, display):
    result = np.zeros((*dapi.shape, 3), dtype=np.float32)
    for name, array, color in (("DAPI", dapi, [.22,.35,.85]), ("E_cadherin", ecad, [.6,.6,.6])):
        d = display[name]
        signal = np.clip((array.astype(np.float32)-d["vmin"])/(d["vmax"]-d["vmin"]), 0, 1)**d["gamma"]
        result = 1-(1-result)*(1-signal[...,None]*np.asarray(color, dtype=np.float32))
    return np.rint(np.clip(result, 0, 1)*255).astype(np.uint8)


def draw_cells(ax, rows, kind, size=.35):
    kwargs = {"s": size, "linewidths": 0, "rasterized": kind != "gates"}
    if kind == "gates":
        kwargs["c"] = rows.coordinate_gate.map(GATE_COLORS)
        kwargs["rasterized"] = len(rows) > 20000
    else:
        kwargs.update(c=rows["predicted_axis_coordinate" if kind == "cv" else "predicted_epithelial_distance_clipped_1p0"],
                      cmap="viridis" if kind == "cv" else "coolwarm", vmin=0, vmax=1 if kind == "cv" else .5)
    return ax.scatter(rows.centroid_x_fullres_px, rows.centroid_y_fullres_px, **kwargs)


def overview(manifest, frames, windows, condition):
    raw = np.load(verify(manifest["overview"]))
    image = composite(raw["dapi"], raw["ecad"], manifest["display"])
    h, w = manifest["native_shape_yx"]
    frame = frames[condition]
    x = frame.centroid_x_fullres_px; y = frame.centroid_y_fullres_px
    limits = [x.min()-500, x.max()+500, y.min()-500, y.max()+500]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, kind, title in zip(axes.flat, ("image", "cv", "epi", "gates"),
                               ("DAPI / E-cadherin", "Crypt-villus prediction", "Epithelial-distance prediction", "Saved IMAP gates")):
        if kind == "image":
            ax.imshow(image, extent=[0,w,h,0])
        else:
            draw_cells(ax, frame, kind, .3)
        ax.set(xlim=limits[:2], ylim=limits[3:1:-1], aspect="equal", title=title,
               xlabel="Native x (pixels)", ylabel="Native y (pixels)")
        for window in windows:
            if window["condition"] != condition:
                continue
            x0,x1,y0,y1 = window["bounds_xxyy_px"]
            color = "#f54c4c" if window["id"].startswith("Current") else "#ffaa00"
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor="none",edgecolor=color,lw=1))
            ax.text(x0,y0,"Current" if window["id"].startswith("Current") else window["id"],color=color,fontsize=9,weight="bold",va="bottom",
                    bbox={"facecolor":"white","alpha":.8,"edgecolor":"none","pad":1})
    fig.suptitle(f"{condition} · original specimen · field locations", fontsize=16, y=.985)
    fig.subplots_adjust(left=.07,right=.98,top=.95,bottom=.06,hspace=.15,wspace=.12)
    return fig


def field_page(window, microscopy, frame, manifest):
    x0,x1,y0,y1 = window["bounds_xxyy_px"]
    rows = frame.loc[frame.centroid_x_fullres_px.ge(x0)&frame.centroid_x_fullres_px.lt(x1)&
                     frame.centroid_y_fullres_px.ge(y0)&frame.centroid_y_fullres_px.lt(y1)].copy()
    if rows.empty or not set(rows.coordinate_gate).issubset(GATE_ORDER):
        raise RuntimeError("Empty field or unknown gate")
    # Display decimation affects microscopy alone. Every saved cell is plotted.
    image = composite(microscopy["dapi"][::3,::3], microscopy["ecad"][::3,::3], manifest["display"])
    fig = plt.figure(figsize=(12,10.6), facecolor="white")
    label = "Current field" if window["id"].startswith("Current") else window["id"]
    fig.text(.045,.972,f"{label}  |  {window['condition']}",fontsize=19,weight="bold",va="top")
    fig.text(.045,.939, "Original specimen · matching field in all four views",fontsize=10,color="#555555")
    rects = [[.045,.535,.425,.36],[.53,.535,.425,.36],[.045,.105,.425,.36],[.53,.105,.425,.36]]
    pixel = manifest["pixel_size_um"]
    for rect,kind,title in zip(rects,("image","gates","cv","epi"),
                                ("DAPI / E-cadherin","Cells by IMAP gate","Crypt-villus prediction","Epithelial-distance prediction")):
        ax = fig.add_axes(rect)
        if kind in ("image","gates"):
            ax.imshow(image,extent=(x0,x1,y1,y0),interpolation="nearest")
        if kind != "image":
            artist = draw_cells(ax,rows,kind,size=3.0 if kind=="gates" else 4.2)
            if kind == "gates":
                p14=rows.loc[rows["class"].eq("Transferred")]
                draw_cells(ax,p14,kind,size=10)
        ax.set(xlim=(x0,x1),ylim=(y1,y0),aspect="equal")
        ax.axis("off"); ax.set_title(title,fontsize=12,pad=8)
        length = 200/pixel
        ax.plot([x0+.04*(x1-x0),x0+.04*(x1-x0)+length],[-.025,-.025],
                transform=ax.get_xaxis_transform(),clip_on=False,color="black",lw=1.5)
        ax.text(x0+.04*(x1-x0)+length/2,-.045,"200 µm",transform=ax.get_xaxis_transform(),
                ha="center",va="top",fontsize=8)
        if kind in ("cv","epi"):
            cax=fig.add_axes([rect[0]+.235,.078,.19,.01])
            bar=fig.colorbar(artist,cax=cax,orientation="horizontal",ticks=[0,1 if kind=="cv" else .5])
            bar.ax.set_xticklabels(["0","1" if kind=="cv" else "≥0.5"]);bar.outline.set_linewidth(.4)
            bar.ax.tick_params(labelsize=8,length=2,pad=2)
    handles=[Line2D([],[],ls="",marker="o",color=GATE_COLORS[g],markersize=6,label=g) for g in GATE_ORDER]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.5,.497),ncol=5,frameon=False,
               fontsize=10,handletextpad=.4,columnspacing=2)
    fig.text(.045,.028,"Illustration candidate; visually reviewed, not an independent accuracy test.",fontsize=8,color="#555555")
    fig.text(.955,.028,f"x {x0:,}–{x1:,}  ·  y {y0:,}–{y1:,} px",ha="right",fontsize=8,color="#555555")
    return fig,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--overview-manifest",type=Path,required=True)
    p.add_argument("--candidate-manifest",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--dpi",type=int,default=300)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    configure_figure_fonts()
    before={str(q):record(q)["sha256"] for q in sorted((Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")/"Regenerated_Figures/Current").glob("*.pdf"))}
    survey=json.loads(a.overview_manifest.read_text());inputs=[record(a.overview_manifest)]
    verify(survey["source_manifest"])
    frames={c:pd.read_parquet(verify(rec)) for c,rec in survey["predictions"].items()}
    inputs.extend(survey["predictions"].values())
    manifest=json.loads(a.candidate_manifest.read_text()) if a.candidate_manifest else survey
    if a.candidate_manifest:
        inputs.append(record(a.candidate_manifest))
        if manifest["source_manifest"]!=survey["source_manifest"] or manifest["display"]!=survey["display"]:
            raise RuntimeError("Candidate and overview lineages differ")
    windows=manifest.get("windows",[{"id":"Current_"+c,"condition":c,"bounds_xxyy_px":b} for c,b in survey["current_bounds"].items()])
    pdf=a.output_dir/"Figure4D_field_candidates.pdf"
    (a.output_dir/"previews").mkdir()
    if a.candidate_manifest:(a.output_dir/"individual_PDFs").mkdir()
    summary=[]
    with PdfPages(pdf) as packet:
        for condition in ("DMSO","RARi"):
            fig=overview(survey,frames,windows,condition)
            packet.savefig(fig,dpi=a.dpi)
            fig.savefig(a.output_dir/"previews"/f"{condition}_locations.png",dpi=180)
            plt.close(fig)
        for window in manifest.get("windows",[]):
            raw=np.load(verify(window["microscopy"]))
            inputs.append(window["microscopy"])
            fig,rows=field_page(window,raw,frames[window["condition"]],manifest)
            packet.savefig(fig,dpi=a.dpi)
            fig.savefig(a.output_dir/"individual_PDFs"/f"{window['id']}.pdf",dpi=a.dpi)
            fig.savefig(a.output_dir/"previews"/f"{window['id']}.png",dpi=a.dpi)
            rows.to_parquet(a.output_dir/f"{window['id']}_displayed_cells.parquet",index=False)
            plt.close(fig)
            summary.append({"candidate_id":window["id"],"condition":window["condition"],
                            **dict(zip(("x_min_px","x_max_px","y_min_px","y_max_px"),window["bounds_xxyy_px"])),
                            "n_cells":len(rows),"n_P14":int(rows["class"].eq("Transferred").sum()),
                            "all_cells_displayed":True,"gate_assignments_changed":False})
            print(f"Rendered {window['id']}: {len(rows)} unchanged cells",flush=True)
    pd.DataFrame(summary).to_csv(a.output_dir/"candidate_fields.csv",index=False)
    after={q:record(q)["sha256"] for q in before}
    if before!=after:raise RuntimeError("Current publication figures changed")
    save_json(a.output_dir/"provenance.json",{
        "status":"AUTHOR_CHOICE_PENDING_NOT_PUBLISHED","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "command":[sys.executable,*sys.argv],"python":platform.python_version(),
        "packages":{n:importlib.metadata.version(n) for n in ("numpy","pandas","matplotlib","pyarrow")},
        "code":[record(__file__),record(ROOT/"Figure_4/spatial.py")],"git_state":"No Git repository","random_seed":None,
        "inputs":inputs,"outputs":[record(q) for q in sorted(a.output_dir.rglob("*")) if q.is_file()],
        "current_figure_hashes_unchanged":before,"candidates":summary,
        "shared_color_ranges":{"cv":[0,1],"epi":[0,.5]},"native_coordinates_unchanged":True,
        "selection_scope":"Author-requested visual choice using morphology and visible prediction behavior; not unbiased performance evidence",
        "warnings":manifest["warnings"]})
    print(pdf,flush=True)


if __name__=="__main__":main()
