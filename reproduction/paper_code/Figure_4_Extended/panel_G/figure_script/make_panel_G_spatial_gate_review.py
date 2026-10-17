#!/usr/bin/env python3
"""Render fixed-window qualitative spatial checks of the unchanged IMAP gates."""
import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from Figure_4.common import require_file,protect_outputs,save_pair,file_record,write_provenance
from Figure_4.release import load_release,release_file
from Figure_4.spatial import GATE_ORDER,GATE_COLORS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['release-manifest','illustration-manifest','dmso-predictions','rari-predictions','checkpoint','gate-definitions','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dpi',type=int,default=600);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    release=load_release(a.release_manifest);ck=release_file(a.checkpoint,release);gates=release_file(a.gate_definitions,release)
    illustration_path=require_file(a.illustration_manifest);illustrations=json.loads(illustration_path.read_text())
    if illustrations['selection_uses_model_predictions'] or illustrations['lock']['sha256']!=release['lock']['sha256']:
        raise RuntimeError('Spatial review does not match the locked repair')
    out=a.output_dir.resolve();stems={c:out/f'panel_G_{c}_spatial_gate_review' for c in ['DMSO','RARi']}
    tables={c:out/f'panel_G_{c}_display_cells.tsv' for c in stems};provenance=out/'panel_G_provenance.json'
    outputs=[path for c in stems for path in [stems[c].with_suffix('.pdf'),stems[c].with_suffix('.png'),tables[c]]]
    overwritten=protect_outputs(outputs+[provenance],a.overwrite);inputs=[file_record(a.release_manifest,'repaired_IF_release'),file_record(ck,'IF_checkpoint'),file_record(gates,'unchanged_gates'),file_record(illustration_path,'fixed_microscopy_windows')];records=[]
    for condition,path in [('DMSO',a.dmso_predictions),('RARi',a.rari_predictions)]:
        path=release_file(path,release);rows=pd.read_parquet(path);window=illustrations['windows'][condition]
        x0,x1,y0,y1=window['bounds_xxyy_px'];pixel=illustrations['pixel_size_um']
        cells=rows.loc[rows.centroid_x_fullres_px.between(x0,x1)&rows.centroid_y_fullres_px.between(y0,y1)].copy()
        if cells.empty or set(cells.coordinate_gate)-set(GATE_ORDER):raise RuntimeError('Invalid fixed-window gate calls')
        image_path=require_file(window['DAPI']['path'],window['DAPI']['sha256']);image=np.load(image_path)
        inputs += [file_record(path,condition+'_whole_predictions'),file_record(image_path,condition+'_DAPI_window')]
        p14=cells.loc[cells['class'].eq('Transferred')];extent=(x0*pixel,x1*pixel,y1*pixel,y0*pixel)
        fig,axes=plt.subplots(1,3,figsize=(8.4,2.75));titles=['DAPI','Coordinate-gate assignments','P14 cells by coordinate gate']
        for i,ax in enumerate(axes):
            if i in [0,2]:
                ax.imshow(image,cmap='gray',vmin=window['display_vmin'],vmax=window['display_vmax'],extent=extent,interpolation='nearest')
            if i==1:
                ax.scatter(cells.x_um,cells.y_um,c=cells.coordinate_gate.map(GATE_COLORS),s=1.3,linewidths=0,rasterized=False)
                ax.scatter(p14.x_um,p14.y_um,s=8,facecolors='none',edgecolors='#222222',linewidths=.4,rasterized=False)
            if i==2:
                ax.scatter(p14.x_um,p14.y_um,c=p14.coordinate_gate.map(GATE_COLORS),s=13,edgecolors='white',linewidths=.3,rasterized=False)
            ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal');ax.axis('off');ax.set_title(titles[i],fontsize=8)
            bar_x=extent[0]+.05*(extent[1]-extent[0]);transform=ax.get_xaxis_transform()
            ax.plot([bar_x,bar_x+200],[-.05,-.05],transform=transform,clip_on=False,color='black',lw=2)
            ax.text(bar_x+100,-.07,'200 µm',transform=transform,clip_on=False,ha='center',va='top',fontsize=7.5)
        handles=[Line2D([],[],marker='o',ls='',color=GATE_COLORS[g],label=g,markersize=5) for g in GATE_ORDER]
        fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,-.07),frameon=False,fontsize=7.5)
        fig.suptitle(f'{condition} · {len(cells):,} cells · {len(p14):,} P14 cells',fontsize=9,y=.98)
        fig.subplots_adjust(left=.01,right=.99,bottom=.19,top=.84,wspace=.05)
        save_pair(fig,stems[condition],a.dpi);plt.close(fig);cells.to_csv(tables[condition],sep='\t',index=False)
        records.append({'condition':condition,'bounds_xxyy_px':[x0,x1,y0,y1],'cells':len(cells),'P14_cells':len(p14)})
    write_provenance(provenance,script=Path(__file__),inputs=inputs,outputs=outputs,overwritten=overwritten,
        extra={'panel':'Extended Figure 4g new components','scope':'Qualitative registered-microscopy review; no independent compartment labels or accuracy estimate',
               'gate_boundaries_changed':False,'region_selection':'Original Figure4D locked geometry, not model-error selection','windows':records,
               'P14_detection_rule':"source QuPath class == 'Transferred'",'density_estimator_used':False,'pdf_points':'vector',
               'scale_bar_um':200,'microscopy_channel':'DAPI; no unsupported E-cadherin channel assignment',
               'component_order':['DMSO DAPI','DMSO all-cell gates with P14 outlines','DMSO P14 gates over DAPI','RARi DAPI','RARi all-cell gates with P14 outlines','RARi P14 gates over DAPI']})


if __name__=='__main__':main()
