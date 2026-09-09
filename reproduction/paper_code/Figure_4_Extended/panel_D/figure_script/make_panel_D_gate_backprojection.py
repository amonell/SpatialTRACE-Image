#!/usr/bin/env python3
"""Show DAPI/E-cadherin beside unchanged cell-gate assignments in both tissues."""
import argparse
import json
import shutil
import sys
from datetime import datetime,timezone
from paper_paths import Path, lock_record, rendering_script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from Figure_4.common import configure_figure_fonts,require_file,protect_outputs,save_pair,file_record,write_provenance
from Figure_4.spatial import GATE_ORDER,GATE_COLORS


DEFAULT_CELL_MARKER_AREA_PT2 = 6.075
DEFAULT_P14_MARKER_AREA_PT2 = 27


def plot_gate_cells(ax, cells, cell_area, p14_area):
    """Filled gate colors, with larger P14 markers and no cell outlines."""
    ordinary=ax.scatter(cells.x_um,cells.y_um,c=cells.coordinate_gate.map(GATE_COLORS),
        s=cell_area,edgecolors='none',linewidths=0,zorder=2)
    p14=cells.loc[cells.is_P14]
    highlighted=ax.scatter(p14.x_um,p14.y_um,c=p14.coordinate_gate.map(GATE_COLORS),
        s=p14_area,edgecolors='none',linewidths=0,zorder=3)
    return ordinary, highlighted


def composite(window,display):
    rgb=None
    for key,color in [('DAPI',np.array([.22,.35,.85],dtype=np.float32)),('E_cadherin',np.array([.6,.6,.6],dtype=np.float32))]:
        rec=window[key];raw=np.load(require_file(rec['path'],rec['sha256']),mmap_mode='r')
        limits=display[key];image=np.asarray(raw,dtype=np.float32)
        image=np.clip((image-limits['vmin'])/(limits['vmax']-limits['vmin']),0,1)**limits['gamma']
        layer=image[...,None]*color
        rgb=layer if rgb is None else 1-(1-rgb)*(1-layer)
    return np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--page-layout',type=Path,help='Approved whole-page layout for a four-across panel D')
    p.add_argument('--dpi',type=int,default=600);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    manifest=json.loads(require_file(a.manifest).read_text())
    if manifest['status']!='FROZEN_REGISTERED_GATE_BACKPROJECTION' or manifest['gate_assignments_changed'] or manifest['boundaries_changed']:
        raise RuntimeError('Gate projection is not a fixed, registered release')
    for key in ['release','channel_manifest','channel_key','gates','checkpoint']:
        rec=manifest[key];require_file(rec['path'],rec['sha256'])
    out=a.output_dir.resolve();stem=out/'panel_D_gate_backprojection';prov=out/'panel_D_gate_backprojection_provenance.json'
    outputs=[stem.with_suffix('.pdf'),stem.with_suffix('.png')];overwritten=protect_outputs(outputs+[prov],a.overwrite)
    if overwritten:
        archive=out.parent/'polished_panel_archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_before_gate_backprojection')
        archive.mkdir(parents=True)
        for path in outputs+[prov]:
            if path.exists():shutil.copy2(path,archive/path.name)
    configure_figure_fonts()
    layout=json.loads(require_file(a.page_layout).read_text())['gate_projection_layout'] if a.page_layout else None
    if layout:
        width,height=layout['page_size_pt']
        fig=plt.figure(figsize=(width/72,height/72),facecolor='none')
        axes=np.array([fig.add_axes([x/width,1-(y+h)/height,w/width,h/height])
            for x,y,w,h in layout['axes_rect_pt']]).reshape(2,2)
    else:
        fig,axes=plt.subplots(2,2,figsize=(7.3,9.1))
    inputs=[file_record(a.manifest,'registered_gate_projection')]
    for row,condition in enumerate(['DMSO','RARi']):
        window=manifest['windows'][condition];image=composite(window,manifest['display'])
        rec=window['cells'];cells=pd.read_parquet(require_file(rec['path'],rec['sha256']))
        if set(cells.coordinate_gate)-set(GATE_ORDER):raise RuntimeError('Unexpected saved gate name')
        if not np.array_equal(cells.is_P14,cells['class'].eq('Transferred')):raise RuntimeError('P14 identities changed')
        x0,x1,y0,y1=window['bounds_xxyy_px'];pixel=manifest['pixel_size_um']
        extent=(x0*pixel,x1*pixel,y1*pixel,y0*pixel)
        for col,ax in enumerate(axes[row]):
            ax.imshow(image,extent=extent,interpolation='nearest')
            if col==1:
                plot_gate_cells(ax,cells,
                    layout['cell_marker_area_pt2'] if layout else DEFAULT_CELL_MARKER_AREA_PT2,
                    layout['P14_marker_area_pt2'] if layout else DEFAULT_P14_MARKER_AREA_PT2)
            ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal');ax.axis('off')
            title=f'{condition} · '+('DAPI / E-cadherin' if col==0 else 'Cells by IMAP gate')
            ax.set_title(title,fontsize=layout['title_size_pt'] if layout else 9,pad=6)
            xbar=extent[0]+.04*(extent[1]-extent[0]);trans=ax.get_xaxis_transform()
            ax.plot([xbar,xbar+200],[-.035,-.035],transform=trans,clip_on=False,color='black',lw=1.4 if layout else 2)
            ax.text(xbar+100,-.05,'200 µm',ha='center',va='top',fontsize=layout['scale_label_size_pt'] if layout else 7,transform=trans)
        inputs.extend([file_record(window[k]['path'],f'{condition}_{k}') for k in ['DAPI','E_cadherin','cells']])
    handles=[Line2D([],[],marker='o',ls='',color=GATE_COLORS[g],label=g,markersize=5) for g in GATE_ORDER]
    if layout:
        x,y=layout['legend_anchor_pt']
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(x/width,1-y/height),
            ncol=5,frameon=False,fontsize=layout['legend_size_pt'],columnspacing=1.2,handletextpad=.35,borderaxespad=0)
        inputs.append(file_record(a.page_layout,'approved_page_layout'))
        for suffix in ('.pdf','.png'):
            fig.savefig(stem.with_suffix(suffix),dpi=a.dpi,transparent=True,bbox_inches=None,pad_inches=0)
        fig.canvas.draw()
        geometry=[]
        for index,ax in enumerate(axes.flat):
            box=ax.get_position();condition=['DMSO','DMSO','RARi','RARi'][index]
            geometry.append({'id':f"D_{condition}_{'image' if index%2==0 else 'gates'}",
                'rect_pt':[box.x0*width,(1-box.y1)*height,box.width*width,box.height*height]})
        stem.with_suffix('.geometry.json').write_text(json.dumps({'rendered_anchors':geometry},indent=2)+'\n')
    else:
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.004),ncol=3,frameon=False,fontsize=8)
        fig.subplots_adjust(left=.015,right=.995,top=.95,bottom=.13,wspace=.07,hspace=.23)
        save_pair(fig,stem,a.dpi)
    plt.close(fig)
    write_provenance(prov,script=Path(__file__),inputs=inputs,outputs=outputs,overwritten=overwritten,
        extra={'panel':'Extended Figure 4D','microscopy':'DAPI blue; confirmed E-cadherin gray',
               'approved_page_layout':layout,
               'gate_colors':GATE_COLORS,'gate_assignments_changed':False,'scope':manifest['scope'],
               'P14_rule':manifest['P14_rule'],'P14_highlight':'Larger gate-colored filled markers, without outlines',
               'cell_marker_area_pt2':layout['cell_marker_area_pt2'] if layout else DEFAULT_CELL_MARKER_AREA_PT2,
               'P14_marker_area_pt2':layout['P14_marker_area_pt2'] if layout else DEFAULT_P14_MARKER_AREA_PT2,
               'cell_marker_outlines':False,
               'scale_bar_um':200,'same_display_limits_both_conditions':True,
               'only_microscopy_is_raster':True,'new_field_selection':False})


if __name__=='__main__':main()
