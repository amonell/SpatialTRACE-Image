"""Small fixed-page primitives shared by the Figure 1 and Figure 3 adapters."""
from __future__ import annotations
import hashlib
import json
import os
from paper_paths import Path, lock_record, rendering_script
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="figure13-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

WIDTH, HEIGHT = 595.276, 841.89


def configure():
    for family in ("Arial", "Arial:style=Bold"):
        font=Path(subprocess.check_output(["fc-match","-f","%{file}",family],text=True).strip())
        if "Arial" not in font_manager.FontProperties(fname=font).get_name():
            raise RuntimeError("Arial is required; font substitution is forbidden")
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family":"Arial","font.size":8,"pdf.fonttype":42,
                         "ps.fonttype":42,"axes.linewidth":.65,"xtick.labelsize":7.8,
                         "ytick.labelsize":7.8,"axes.labelsize":8,"axes.titlesize":8,
                         "figure.dpi":300,"savefig.dpi":300})


def figure():
    configure()
    return plt.figure(figsize=(WIDTH/72,HEIGHT/72),frameon=False)


def axis(fig,rect,*,off=True):
    x,y,w,h=map(float,rect)
    ax=fig.add_axes([x/WIDTH,1-(y+h)/HEIGHT,w/WIDTH,h/HEIGHT])
    if off:ax.set_axis_off()
    ax._assembly_rect_pt=list(rect)
    return ax


def image(fig,rect,array,*,gray=True,normalize=False):
    ax=axis(fig,rect)
    kwargs={"cmap":"gray","vmin":0,"vmax":255} if gray else {}
    if normalize:
        positive=np.asarray(array)[np.asarray(array)>0]
        kwargs.update(vmin=float(np.percentile(positive,.5)),vmax=float(np.percentile(positive,99.8)))
    ax.imshow(array,interpolation="nearest",aspect="auto",**kwargs)
    return ax


def scatter(fig,rect,frame,x,y,*,value=None,color=None,cmap="viridis",vmin=0,vmax=1,size=.25,limits=None,alpha=1):
    ax=axis(fig,rect)
    artist=ax.scatter(frame[x],frame[y],c=frame[value] if value else color,s=size,linewidths=0,
                      cmap=cmap if value else None,vmin=vmin if value else None,
                      vmax=vmax if value else None,rasterized=True,alpha=alpha)
    if limits is None:
        xx=frame[x].to_numpy();yy=frame[y].to_numpy();px=(xx.max()-xx.min())*.015;py=(yy.max()-yy.min())*.015
        limits=(xx.min()-px,xx.max()+px,yy.min()-py,yy.max()+py)
    x0,x1,y0,y1=limits;ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
    return ax,artist


def colorbar_strip(fig,rect,cmap,vmin=0,vmax=1,vertical=True):
    arr=np.linspace(vmin,vmax,256)[:,None] if vertical else np.linspace(vmin,vmax,256)[None,:]
    ax=axis(fig,rect)
    # Same-color half-point edge overlap prevents PDF-viewer stitching seams;
    # clipping retains the exact measured outer colorbar rectangle.
    ax.pcolormesh(arr,cmap=cmap,vmin=vmin,vmax=vmax,shading="flat",edgecolors="face",linewidths=.5,antialiased=False,rasterized=False)
    ax.set_xlim(0,arr.shape[1]);ax.set_ylim(0,arr.shape[0])
    return ax


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def record(path,role,expected=None):
    path=Path(path);actual=hash_file(path)
    if expected and actual!=expected:raise RuntimeError(f"Hash mismatch: {path}")
    return {"path":str(path),"sha256":actual,"role":role}


def save_panel_layers(fig,output_dir,anchors,panel_axes):
    """Export each panel from the same live artists used by the complete page."""
    if set(anchors)!=set(panel_axes) or set(anchors.values())!=set(fig.axes):
        raise RuntimeError('Every scientific axis must have exactly one panel owner')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    visibility={ax:ax.get_visible() for ax in fig.axes};layers={}
    try:
        for panel in sorted(set(panel_axes.values())):
            for name,ax in anchors.items():ax.set_visible(visibility[ax] and panel_axes[name]==panel)
            name=f'scientific_panel_{panel}.pdf'
            fig.savefig(output_dir/name,transparent=True,dpi=300)
            layers[panel]=name
    finally:
        for ax,visible in visibility.items():ax.set_visible(visible)
    return layers


def save(fig,path,anchors):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,transparent=True,dpi=300)
    fig.canvas.draw()
    actual=[]
    for key,ax in anchors.items():
        bb=ax.get_position()
        actual.append({"id":key,"rect_pt":[bb.x0*WIDTH,(1-bb.y1)*HEIGHT,bb.width*WIDTH,bb.height*HEIGHT]})
    plt.close(fig)
    return actual
