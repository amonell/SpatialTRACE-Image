#!/usr/bin/env python3
"""Render companion pages for all seven new sections and the nine-mouse cohort."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from render_early_preview import Snapshot, regional_comparison, imaps, compositions
from render_review_panels import COLORS, GATES, composite, config, gate_background, scalebar, sha


def new_maps(data):
    fields = json.loads(data.files["display_fields.json"].read_text())
    sections = data.sections.loc[data.sections.cohort == "new"]
    fig, axes = plt.subplots(len(sections), 3, figsize=(9, 2.2 * len(sections) + .6), squeeze=False)
    fig.subplots_adjust(left=.04, right=.97, top=.965, bottom=.05, hspace=.22, wspace=.03)
    for j, title in enumerate(("DAPI / E-cadherin", "Crypt–villus coordinate", "Epithelial-distance coordinate")):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold", pad=25)
    for i, (_, mouse) in enumerate(sections.iterrows()):
        field = fields[mouse.source_id]
        assets = np.load(data.files[f"{mouse.source_id}_microscopy.npz"])
        frame = pd.read_parquet(data.files[f"{mouse.source_id}.parquet"])
        height, width = field["shape_yx"]
        axes[i, 0].imshow(composite(assets["dapi_overview"], assets["ecad_overview"], field["levels"]), extent=(0,width,height,0))
        for j, column, cmap, vmax in ((1,"predicted_axis_coordinate","viridis",1), (2,"predicted_epithelial_distance_clipped_1p0","coolwarm",.5)):
            axes[i,j].scatter(frame.centroid_x_fullres_px, frame.centroid_y_fullres_px, c=frame[column],
                              s=.065, linewidths=0, cmap=cmap, vmin=0, vmax=vmax, rasterized=True)
        for j, ax in enumerate(axes[i]):
            ax.set(xlim=(0,width), ylim=(height,0), aspect="equal")
            ax.axis("off"); scalebar(ax,field["pixel_size_x_um"],1000,"white" if j==0 else "black")
        axes[i,0].text(.5,1.025,f"{mouse.condition} {mouse.source_id[-2:]}",transform=axes[i,0].transAxes,ha="center",fontsize=8)
    for j, cmap, vmax, title in ((1,"viridis",1,"Crypt–villus coordinate"),(2,"coolwarm",.5,"Epithelial-distance coordinate")):
        box=axes[-1,j].get_position(); cax=fig.add_axes([box.x0,.025,box.width,.007])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,vmax),cmap=cmap),cax=cax,orientation="horizontal",ticks=[0,vmax])
        cb.set_label(title)
        if vmax==.5: cb.ax.set_xticklabels(["0","≥0.5"])
    data.save(fig,"Additional_04_new_section_maps")


def fields_by_gate(data):
    fields=json.loads(data.files["display_fields.json"].read_text())
    sections=data.sections.loc[data.sections.cohort=="new"]
    fig,axes=plt.subplots(4,4,figsize=(12,12.7))
    fig.subplots_adjust(left=.025,right=.985,top=.955,bottom=.06,wspace=.04,hspace=.22)
    for i,(_,mouse) in enumerate(sections.iterrows()):
        field=fields[mouse.source_id]; assets=np.load(data.files[f"{mouse.source_id}_microscopy.npz"])
        image=composite(assets["dapi_crop"],assets["ecad_crop"],field["levels"])
        frame=pd.read_parquet(data.files[f"{mouse.source_id}.parquet"])
        x,y,w,h=field["bounds_xywh_px"]
        cells=frame.loc[(frame.centroid_x_fullres_px>=x)&(frame.centroid_x_fullres_px<x+w)&
                        (frame.centroid_y_fullres_px>=y)&(frame.centroid_y_fullres_px<y+h)]
        for j in range(2):
            ax=axes[i//2,2*(i%2)+j]
            ax.imshow(image,extent=(x,x+w,y+h,y))
            if j:
                ax.scatter(cells.centroid_x_fullres_px,cells.centroid_y_fullres_px,
                           c=cells.gate_name.map(dict(zip(GATES,COLORS))),s=np.where(cells.is_p14==1,9,1.8),
                           edgecolors="none",linewidths=0,rasterized=True)
            ax.set(xlim=(x,x+w),ylim=(y+h,y),aspect="equal");ax.axis("off")
            scalebar(ax,field["pixel_size_x_um"],100,"white")
            ax.set_title(f"{mouse.condition} {mouse.source_id[-2:]}\n"+("Cells by IMAP gate" if j else "DAPI / E-cadherin"),fontsize=8)
    axes[-1,-2].axis("off"); axes[-1,-1].axis("off")
    handles=[Line2D([],[],marker="o",ls="",color=c,label=g,markersize=5) for g,c in zip(GATES,COLORS)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.015),ncol=5,frameon=False)
    data.save(fig,"Additional_05_new_gate_backprojections")


def background_imaps(data):
    rows=data.sections
    fig,axes=plt.subplots(3,3,figsize=(10.5,10.7))
    fig.subplots_adjust(left=.07,right=.98,top=.90,bottom=.09,hspace=.35,wspace=.25)
    fig.suptitle("Individual-mouse non-P14 coordinate maps",fontsize=14)
    for i,(_,mouse) in enumerate(rows.iterrows()):
        ax=axes.flat[i]; gate_background(ax)
        frame=pd.read_parquet(data.files[f"{mouse.source_id}.parquet"])
        frame=frame.loc[frame.is_p14==0]
        ax.scatter(frame.predicted_epithelial_distance_clipped_1p0,frame.predicted_axis_coordinate,
                   s=.06,c="#4A426B",alpha=.12,linewidths=0,rasterized=True)
        table=data.gates.loc[data.gates.source_id==mouse.source_id].set_index("gate")
        for gate,x,y,ha in (("Top IE",.02,.8,"left"),("Top LP",.98,.8,"right"),("Crypt IE",.02,.08,"left"),
                           ("Crypt LP",.98,.325,"right"),("Muscularis",.98,.08,"right")):
            ax.text(x,y,f"{'Musc' if gate=='Muscularis' else gate}\n{table.loc[gate,'non_p14_composition_percent']:.1f}%",
                    ha=ha,va="center",fontsize=7,bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":1})
        ax.set_title(f"{mouse.condition} · {data.label(mouse)}\n{len(frame):,} non-P14 cells",fontsize=9)
        if i%3==0:ax.set_ylabel("Crypt–villus coordinate")
        if i>=6:ax.set_xlabel("Epithelial-distance coordinate")
    fig.text(.5,.025,data.footer(),ha="center",fontsize=7)
    data.save(fig,"Additional_06_individual_non_P14_IMAPs")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();config();data=Snapshot(a.input,a.output)
    if data.manifest["status"]!="frozen_nine_mouse_figure_release":raise RuntimeError("Expected the selected complete cohort")
    regional_comparison(data); imaps(data); compositions(data)
    new_maps(data); fields_by_gate(data); background_imaps(data)
    manifest={"status":"rendered_pending_visual_QA","timestamp_utc":datetime.now(timezone.utc).isoformat(),
              "command":[sys.executable,*sys.argv],"python":platform.python_version(),"random_seed":None,
              "git_state":"Project directory is not a Git repository","inputs":data.inputs,
              "release_manifest":{"path":str(data.source/'manifest.json'),"sha256":sha(data.source/'manifest.json')},
              "code":[{"path":str(path.resolve()),"sha256":sha(path)} for path in
                      (Path(__file__),Path(__file__).with_name('render_early_preview.py'),Path(__file__).with_name('render_review_panels.py'))],
              "outputs":data.outputs,"warnings":data.manifest["warnings"]}
    with (data.output/'rendering_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)


if __name__=="__main__":main()
