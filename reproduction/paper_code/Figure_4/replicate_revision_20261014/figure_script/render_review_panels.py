#!/usr/bin/env python3
"""Render new-section review figures from frozen inference; never fit or infer."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

GATES = ("Top IE","Top LP","Crypt IE","Crypt LP","Muscularis")
COLORS = ("#7CB7C9","#9FBF7A","#D68B81","#C8B96A","#B79572")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def config():
    for weight in ("normal","bold"):
        findfont(FontProperties(family="Arial",weight=weight),fallback_to_default=False)
    plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.titlesize":9,
        "axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"pdf.fonttype":42,
        "ps.fonttype":42,"axes.linewidth":.6,"savefig.facecolor":"white"})


def composite(d,e,levels):
    d = np.clip(np.asarray(d,dtype=np.float32)/max(levels["dapi_upper"],1),0,1)
    e = np.clip(np.asarray(e,dtype=np.float32)/max(levels["ecad_upper"],1),0,1)
    return np.clip(np.stack([.65*d,.65*d+.8*e,.9*d+.8*e],axis=-1),0,1)


class Inputs:
    def __init__(self,root,out):
        self.root,self.out = root,out
        self.manifest = json.loads((root/"analysis/manifest.json").read_text())
        if self.manifest["status"] != "frozen_condition_blind_analysis":
            raise RuntimeError("Unrecognized frozen analysis")
        self.files = {Path(r["path"]).name:r for r in self.manifest["outputs"]}
        for r in self.files.values():
            if sha(r["path"]) != r["sha256"]:
                raise RuntimeError(f"Frozen analysis changed: {r['path']}")
        self.sources = json.loads((root/"inspection.json").read_text())["sources"]
        self.levels = json.loads((root/"alignment_qa/display_levels.json").read_text())
        self.review = json.loads((root/"alignment_review.json").read_text())
        self.sections = pd.read_csv(self.files["sections.csv"]["path"])
        self.gates = pd.read_csv(self.files["gates.csv"]["path"])
        self.frames = {}
        self.outputs = []
        self.extra_inputs = []
        key_lock=json.loads((root/"cohort_key_manifest.json").read_text())
        if sha(key_lock["key"]["path"])!=key_lock["key"]["sha256"]:
            raise RuntimeError("Verified slide-label key changed")
        self.key=pd.read_csv(key_lock["key"]["path"]).set_index("source_id")
        self.extra_inputs.append(key_lock["key"])
        (out/"pdfs").mkdir(parents=True,exist_ok=True)
        (out/"previews").mkdir(exist_ok=True)

    def title(self,sid):
        return f"{self.key.loc[sid,'condition']} · section {sid[-2:]}"

    def frame(self,sid):
        if sid not in self.frames:
            self.frames[sid] = pd.read_parquet(self.files[f"{sid}.parquet"]["path"])
        return self.frames[sid]

    def images(self,source):
        sid = source["source_id"]
        manifest = json.loads((self.root/"images"/sid/"manifest.json").read_text())
        for entry in manifest["outputs"]:
            if not any(r["path"] == entry["path"] for r in self.extra_inputs):
                if sha(entry["path"]) != entry["sha256"]:
                    raise RuntimeError("Microscopy cache changed")
                self.extra_inputs.append(entry)
        return tuple(np.load(self.root/"images"/sid/f"{channel}.npy",mmap_mode="r") for channel in ("dapi","ecad"))

    def save(self,fig,stem):
        pdf = self.out/"pdfs"/(stem+".pdf")
        png = self.out/"previews"/(stem+".png")
        if pdf.exists() or png.exists():
            raise FileExistsError(pdf)
        fig.savefig(pdf,dpi=600)
        fig.savefig(png,dpi=150)
        plt.close(fig)
        self.outputs += [{"path":str(p),"sha256":sha(p)} for p in (pdf,png)]
        print(f"Rendered {pdf}",flush=True)


def scalebar(ax,pixel_um,length_um,color="black"):
    xmin,xmax = ax.get_xlim(); ymax,ymin = ax.get_ylim()
    x = xmin+.06*(xmax-xmin); y = ymax-.06*(ymax-ymin)
    length = length_um/pixel_um
    ax.plot([x,x+length],[y,y],color=color,lw=1.8,solid_capstyle="butt")
    label = "1 mm" if length_um==1000 else f"{length_um:g} µm"
    ax.annotate(label,(x+length/2,y),xytext=(0,4),textcoords="offset points",ha="center",color=color,fontsize=7)


def whole_sections(data):
    fig,axes = plt.subplots(8,3,figsize=(9,18),gridspec_kw={"hspace":.18,"wspace":.06})
    fig.subplots_adjust(left=.045,right=.95,top=.965,bottom=.055)
    for j,title in enumerate(("DAPI / E-cadherin","Crypt–villus coordinate","Epithelial-distance coordinate")):
        axes[0,j].set_title(title,pad=16,fontweight="bold")
    for i,source in enumerate(data.sources):
        sid = source["source_id"]; frame = data.frame(sid)
        d,e = data.images(source)
        extent = (0,d.shape[1],d.shape[0],0)
        axes[i,0].imshow(composite(d[::24,::24],e[::24,::24],data.levels[sid]),extent=extent)
        for j,column,cmap,vmax in ((1,"predicted_axis_coordinate","viridis",1),(2,"predicted_epithelial_distance_clipped_1p0","coolwarm",.5)):
            axes[i,j].scatter(frame.centroid_x_fullres_px,frame.centroid_y_fullres_px,
                            c=frame[column],cmap=cmap,vmin=0,vmax=vmax,s=.07,linewidths=0,rasterized=True)
        for j,ax in enumerate(axes[i]):
            ax.set(xlim=(0,d.shape[1]),ylim=(d.shape[0],0),aspect="equal")
            ax.axis("off")
            scalebar(ax,source["selected_series"]["pixel_size_x_um"],1000,"white" if j==0 else "black")
        axes[i,0].text(-.06,1.03,f"{chr(97+i)}",transform=axes[i,0].transAxes,fontsize=11,fontweight="bold")
        axes[i,0].text(.5,1.02,data.title(sid),transform=axes[i,0].transAxes,ha="center",fontsize=8)
    for j,cmap,vmax,label in ((1,"viridis",1,"Crypt–villus coordinate"),(2,"coolwarm",.5,"Epithelial-distance coordinate")):
        rect=axes[-1,j].get_position()
        cax=fig.add_axes([rect.x0,.028,rect.width,.007])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,vmax),cmap=cmap),cax=cax,orientation="horizontal",ticks=[0,vmax])
        cb.set_label(label)
        if vmax==.5: cb.ax.set_xticklabels(["0","≥0.5"])
    data.save(fig,"Additional_01_all_section_maps")


def gate_background(ax):
    rectangles=[(0,.4,.28,.6),( .28,.4,.72,.6),(0,0,.28,.4),(.28,.25,.72,.15),(.28,0,.72,.25)]
    for color,rect in zip(COLORS,rectangles):
        ax.add_patch(Rectangle(rect[:2],rect[2],rect[3],facecolor=color,alpha=.23,edgecolor="none",zorder=0))
    ax.axvline(.28,color=".4",lw=.6); ax.axhline(.4,color=".4",lw=.6)
    ax.plot([.28,1],[.25,.25],color=".4",lw=.6)
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1])


def imaps(data,p14):
    population="P14" if p14 else "non-P14"
    fig,axes=plt.subplots(2,4,figsize=(12,6.8))
    fig.subplots_adjust(left=.065,right=.985,top=.93,bottom=.13,wspace=.2,hspace=.34)
    for i,source in enumerate(data.sources):
        sid=source["source_id"]; frame=data.frame(sid); frame=frame.loc[frame.is_p14==int(p14)]
        ax=axes.flat[i]; gate_background(ax)
        ax.scatter(frame.predicted_epithelial_distance_clipped_1p0,frame.predicted_axis_coordinate,
                   s=1.5 if p14 else .08,c="#315778" if p14 else "#413268",alpha=.5 if p14 else .16,
                   linewidths=0,rasterized=not p14,zorder=1)
        values=data.gates.loc[data.gates.source_id==sid].set_index("gate")
        column="p14_composition_percent" if p14 else "non_p14_composition_percent"
        for gate,x,y,ha in (("Top IE",.03,.74,"left"),("Top LP",.97,.79,"right"),
                            ("Crypt IE",.03,.07,"left"),("Crypt LP",.97,.32,"right"),("Muscularis",.97,.08,"right")):
            label="Musc" if gate=="Muscularis" else gate
            ax.text(x,y,f"{label}\n{values.loc[gate,column]:.1f}%",ha=ha,va="center",fontsize=6.8,
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.85,pad=1),zorder=3)
        ax.set_title(f"{data.title(sid)}\n{len(frame):,} {population} cells",fontsize=8)
        ax.text(-.13,1.08,chr(97+i),transform=ax.transAxes,fontweight="bold",fontsize=11)
        if i%4==0: ax.set_ylabel("Crypt–villus coordinate")
        if i>=4: ax.set_xlabel("Epithelial-distance coordinate")
    data.save(fig,"Additional_02_P14_IMAPs" if p14 else "Additional_03_non_P14_IMAPs")


def gate_overlays(data):
    fig,axes=plt.subplots(4,4,figsize=(12,12.5))
    fig.subplots_adjust(left=.025,right=.985,top=.96,bottom=.05,wspace=.04,hspace=.2)
    for i,source in enumerate(data.sources):
        sid=source["source_id"]; d,e=data.images(source); frame=data.frame(sid)
        w=source["qa_windows"][data.review["display_window_indices_zero_based"][sid]]
        x,y,ww,hh=w["x"],w["y"],w["width"],w["height"]
        image=composite(d[y:y+hh,x:x+ww],e[y:y+hh,x:x+ww],data.levels[sid])
        mask=(frame.centroid_x_fullres_px>=x)&(frame.centroid_x_fullres_px<x+ww)&(frame.centroid_y_fullres_px>=y)&(frame.centroid_y_fullres_px<y+hh)
        sub=frame.loc[mask]
        row=i//2; col=2*(i%2)
        for j in range(2):
            ax=axes[row,col+j]
            ax.imshow(image,extent=(x,x+ww,y+hh,y))
            if j:
                for gate,color in zip(GATES,COLORS):
                    points=sub.loc[sub.gate_name==gate]
                    ax.scatter(points.centroid_x_fullres_px,points.centroid_y_fullres_px,
                               s=np.where(points.is_p14==1,7,1.1),c=color,linewidths=0,alpha=.9,rasterized=True)
            ax.set(xlim=(x,x+ww),ylim=(y+hh,y),aspect="equal"); ax.axis("off")
            scalebar(ax,source["selected_series"]["pixel_size_x_um"],100,"white")
            ax.set_title(data.title(sid)+"\n"+("Cells by coordinate gate" if j else "DAPI / E-cadherin"),fontsize=8)
        axes[row,col].text(-.03,1.05,chr(97+i),transform=axes[row,col].transAxes,fontweight="bold",fontsize=11)
    handles=[Line2D([],[],linestyle="none",marker="o",color=c,markersize=5,label=g) for g,c in zip(GATES,COLORS)]
    fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,.005))
    data.save(fig,"Additional_04_gate_backprojection")


def summaries(data):
    fig=plt.figure(figsize=(11,7.5))
    grid=fig.add_gridspec(2,5,left=.07,right=.98,bottom=.1,top=.93,hspace=.55,wspace=.65,height_ratios=[1,1.25])
    for j,gate in enumerate(GATES):
        ax=fig.add_subplot(grid[0,j])
        frame=data.gates.loc[data.gates.gate==gate].set_index("source_id").loc[data.sections.source_id]
        ax.scatter(np.arange(8),frame.p14_per_10000_non_p14,s=26,c=COLORS[j],edgecolors="#333333",linewidths=.4)
        ax.set(xticks=np.arange(8),xticklabels=[f"{i:02d}" for i in range(1,9)],xlabel="Section",title=gate,ylim=(0,None))
        ax.spines[["top","right"]].set_visible(False)
        if j==0:
            ax.set_ylabel("P14 per 10,000 non-P14 cells\nin the same gate")
            ax.text(-.42,1.15,"a",transform=ax.transAxes,fontweight="bold",fontsize=12)
    ax=fig.add_subplot(grid[1,:3]); bottom=np.zeros(8)
    for gate,color in zip(GATES,COLORS):
        frame=data.gates.loc[data.gates.gate==gate].set_index("source_id").loc[data.sections.source_id]
        values=frame.p14_composition_percent.to_numpy()
        ax.bar(np.arange(8),values,bottom=bottom,color=color,label=gate,width=.7)
        bottom+=values
    ax.set(xticks=np.arange(8),xticklabels=[f"{i:02d}" for i in range(1,9)],xlabel="Section",ylabel="Within-P14 composition (%)",ylim=(0,100))
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,ncol=3,loc="lower left",bbox_to_anchor=(0,1.02),fontsize=7)
    ax.text(-.12,1.16,"b",transform=ax.transAxes,fontweight="bold",fontsize=12)
    ax=fig.add_subplot(grid[1,3:])
    ax.scatter(np.arange(8),data.sections.p14_percent_all_cells,s=30,c="#315778")
    for i,value in enumerate(data.sections.n_p14):
        ax.annotate(f"{int(value):,}",(i,data.sections.p14_percent_all_cells.iloc[i]),xytext=(0,7),textcoords="offset points",ha="center",fontsize=7)
    ax.set(xticks=np.arange(8),xticklabels=[f"{i:02d}" for i in range(1,9)],xlabel="Section",ylabel="P14 among all cells (%)",ylim=(0,max(data.sections.p14_percent_all_cells)*1.2))
    ax.spines[["top","right"]].set_visible(False)
    ax.text(-.17,1.16,"c",transform=ax.transAxes,fontweight="bold",fontsize=12)
    data.save(fig,"Additional_05_section_abundance_composition")


def cohort_comparison(data):
    path=data.root/"cohort/manifest.json"
    cohort=json.loads(path.read_text())
    if cohort["status"]!="frozen_ten_mouse_cohort":
        raise RuntimeError("Ten-mouse cohort is not ready")
    inputs={Path(r["path"]).name:r for r in cohort["outputs"]}
    for name in ("gates.csv","group_summary.csv"):
        r=inputs[name]
        if sha(r["path"])!=r["sha256"]: raise RuntimeError("Cohort table changed")
        data.extra_inputs.append(r)
    points=pd.read_csv(inputs["gates.csv"]["path"])
    stats=pd.read_csv(inputs["group_summary.csv"]["path"])
    stats=stats.loc[stats.subset=="all_ten_mice"]
    fig,axes=plt.subplots(2,5,figsize=(12,6.5))
    fig.subplots_adjust(left=.085,right=.985,bottom=.12,top=.9,wspace=.52,hspace=.32)
    colors={"DMSO":"#4F7F8A","RARi":"#A55D54"}
    for row,(metric,label) in enumerate((("p14_per_10000_non_p14","P14 per 10,000 non-P14 cells\nin the same gate"),
                                        ("p14_composition_percent","Within-P14 composition (%)"))):
        for j,gate in enumerate(GATES):
            ax=axes[row,j]
            for x,condition in enumerate(("DMSO","RARi")):
                selected=points.loc[(points.gate==gate)&(points.condition==condition)].sort_values("source_id")
                assert len(selected)==5 and selected.mouse_id.nunique()==5
                for offset,(_,r) in zip(np.linspace(-.14,.14,5),selected.iterrows()):
                    old=r.cohort=="original"
                    ax.scatter(x+offset,r[metric],s=30,marker="s" if old else "o",
                               facecolors="white" if old else colors[condition],edgecolors=colors[condition],linewidths=1,zorder=3)
                summary=stats.loc[(stats.gate==gate)&(stats.condition==condition)&(stats.metric==metric)].iloc[0]
                ax.errorbar(x,summary["mean"],yerr=summary["sem"],fmt="_",color="#222222",markersize=15,
                            capsize=4,elinewidth=1,zorder=4)
            ax.set(xticks=[0,1],xticklabels=["DMSO","RARi"],xlim=(-.45,1.45),ylim=(0,None))
            ax.spines[["top","right"]].set_visible(False)
            if row==0: ax.set_title(gate,fontweight="bold")
            if j==0:
                ax.set_ylabel(label)
                ax.text(-.36,1.1,"a" if row==0 else "b",transform=ax.transAxes,fontweight="bold",fontsize=12)
    handles=[Line2D([],[],marker="o",linestyle="none",color="#555555",label="New mice"),
             Line2D([],[],marker="s",linestyle="none",markerfacecolor="white",markeredgecolor="#555555",label="Original mice"),
             Line2D([],[],marker="_",linestyle="none",color="black",markersize=12,label="Mean ± SEM")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False)
    data.save(fig,"Additional_06_ten_mouse_comparison")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); config(); data=Inputs(a.root,a.output)
    whole_sections(data); imaps(data,True); imaps(data,False); gate_overlays(data); summaries(data); cohort_comparison(data)
    manifest={"command":sys.argv,"code":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)},
              "timestamp_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
              "packages":{"matplotlib":matplotlib.__version__,"pandas":pd.__version__,"numpy":np.__version__},
              "analysis_manifest":{"path":str(a.root/"analysis/manifest.json"),"sha256":sha(a.root/"analysis/manifest.json")},
              "inputs":list(data.files.values())+data.extra_inputs,"outputs":data.outputs,
              "random_seed":None,"git_state":"not a Git repository","training_or_inference":False,
              "display_downsampling":"microscopy overview stride24; all prediction cells shown",
              "selection":data.review["display_selection"],
              "warnings":["Mouse identities confirmed by author; new treatment identities transcribed from scanned labels", "Original manuscript figures remain unchanged", "First five pages describe the new eight sections; page six includes all ten mice"]}
    with (a.output/"rendering_manifest.json").open("x") as f: json.dump(manifest,f,indent=2)


if __name__=="__main__":
    main()
