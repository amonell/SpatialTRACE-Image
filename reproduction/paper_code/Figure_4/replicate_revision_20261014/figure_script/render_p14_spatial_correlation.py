#!/usr/bin/env python3
"""Render the frozen tissue-by-tissue 2D P14 distribution correlation heatmap."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'frozen_preprocessing/if_replicates_20261014'))
from audit_regional_statistics import record, verified


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    release=json.loads(verified({'path':str(a.manifest),'sha256':a.manifest_sha256}).read_text())
    assert release['status']=='frozen_p14_joint_spatial_correlation'
    files={Path(r['path']).name:verified(r) for r in release['outputs']}
    definition=json.loads(files['definition.json'].read_text())
    order=definition['row_and_column_order']
    rows=pd.read_csv(files['rows.csv']).set_index('mouse_id').loc[order]
    table=pd.read_csv(files['correlation.csv'],index_col='mouse_id').loc[order,order]
    values=table.to_numpy()
    assert values.shape==(9,9) and np.isfinite(values).all()
    np.testing.assert_allclose(values,values.T); np.testing.assert_allclose(np.diag(values),1)
    for weight in ['normal','bold']: findfont(FontProperties(family='Arial',weight=weight),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(9.2,8.6))
    ax=fig.add_axes([.25,.30,.57,.61])
    cmap=plt.get_cmap('RdBu_r'); norm=Normalize(vmin=-1,vmax=1)
    mesh=ax.pcolormesh(np.arange(10),np.arange(10),values,cmap=cmap,norm=norm,edgecolors='white',linewidth=.65)
    ax.set(xlim=(0,9),ylim=(9,0),xticks=np.arange(9)+.5,yticks=np.arange(9)+.5,
           xticklabels=rows.display_label,yticklabels=rows.display_label)
    ax.tick_params(length=0,pad=8); ax.spines[:].set_visible(False)
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor',fontsize=9)
    for ticks in [ax.get_xticklabels(),ax.get_yticklabels()]:
        for tick,condition in zip(ticks,rows.condition): tick.set_color({'DMSO':'#37788F','RARi':'#B05B49'}[condition])
    ax.axhline(4,color='white',lw=2.5); ax.axvline(4,color='white',lw=2.5)
    for i in range(9):
        for j in range(9):
            rgb=np.asarray(cmap(norm(values[i,j]))[:3])
            light=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)@np.array([.2126,.7152,.0722])
            ax.text(j+.5,i+.5,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9.2,
                    color='white' if light<.35 else '#202020')
    cb=fig.colorbar(mesh,cax=fig.add_axes([.85,.30,.024,.61]),ticks=[-1,-.5,0,.5,1])
    cb.set_label('Pearson r',labelpad=9); cb.outline.set_visible(False)
    title=fig.text(.5,.978,'Similarity of 2D P14 spatial distributions',ha='center',va='top',fontsize=15,weight='bold')
    fig.text(.5,.94,'Crypt–villus × epithelial-distance space · one row and column per SI',ha='center',fontsize=9.5)
    method=fig.text(.5,.105,'Pearson correlation between unit-mass P14 IMAPs on the same 100 × 100 grid.',ha='center',fontsize=9)
    detail=fig.text(.5,.066,'Original IMAP smoothing retained. Original-experiment mice are labeled; all tissue removals applied.',ha='center',fontsize=8)
    fig.text(.5,.028,'Values describe similarity of spatial distributions, independent of total P14 abundance.',ha='center',fontsize=8.5)
    a.output.mkdir(parents=True,exist_ok=False)
    outputs=[]
    for ext in ['pdf','png']:
        path=a.output/f'Additional_12_P14_2D_spatial_correlation.{ext}'
        fig.savefig(path,dpi=300); outputs.append(record(path))
    supported=pd.read_csv(files['correlation_occupied_support.csv'],index_col='mouse_id').loc[order,order].to_numpy()
    np.testing.assert_allclose(supported,supported.T); np.testing.assert_allclose(np.diag(supported),1)
    mesh.set_array(supported)
    for annotation,value in zip(ax.texts,supported.ravel()):
        rgb=np.asarray(cmap(norm(value))[:3])
        light=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)@np.array([.2126,.7152,.0722])
        annotation.set_text(f'{value:.2f}'); annotation.set_color('white' if light<.35 else '#202020')
    title.set_text('2D P14 correlation: occupied-bin sensitivity')
    method.set_text(f'Pearson correlation restricted to {definition["n_sensitivity_bins"]:,} bins containing observed P14 cells in any SI.')
    detail.set_text('One common mask for every pair; the original smoothed maps and color scale are unchanged.')
    for ext in ['pdf','png']:
        path=a.output/f'Additional_13_P14_2D_correlation_occupied_bins.{ext}'
        fig.savefig(path,dpi=300); outputs.append(record(path))
    plt.close(fig)
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
                'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','matplotlib']},
                'git_state':'Not a Git worktree','random_seed':None,
                'inputs':[record(a.manifest),*release['outputs'],record(__file__),record(ROOT/'frozen_preprocessing/if_replicates_20261014/audit_regional_statistics.py')],
                'outputs':outputs,'warnings':release['warnings'],'color_scale':[-1,1],
                'correlations_computed_during_render':False,'paper_figures_modified':False}
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'outputs':outputs}))


if __name__=='__main__': main()
