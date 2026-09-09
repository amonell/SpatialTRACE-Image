#!/usr/bin/env python3
"""Render audited, separately retrained image-branch comparisons."""
import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from Figure_4.common import configure_figure_fonts,require_file,protect_outputs,save_pair,file_record,write_provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    release=json.loads(require_file(a.release).read_text())
    if release['status']!='FROZEN_EVALUATED_COMPARISON' or release['test_n']!=5107:
        raise RuntimeError('Incomplete matched comparison')
    audit=release['audit'];require_file(audit['path'],audit['sha256'])
    rec=release['outputs']['metrics'];table=pd.read_csv(require_file(rec['path'],rec['sha256']),sep='\t')
    order=['full','no_local','no_context','no_fine']
    if len(table)!=8 or not table.n.eq(5107).all():raise RuntimeError('Incomplete coordinate/condition table')
    out=a.output_dir.resolve();stem=out/'panel_F_matched_branch_retraining'
    provenance=out/'panel_F_provenance.json'
    outputs=[stem.with_suffix('.pdf'),stem.with_suffix('.png')]
    overwritten=protect_outputs(outputs+[provenance],a.overwrite)
    if overwritten:
        import shutil
        from datetime import datetime,timezone
        archive=out.parent/'polished_panel_archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_before_F_retraining')
        archive.mkdir(parents=True)
        for path in outputs+[provenance]:
            if path.exists():shutil.copy2(path,archive/path.name)
    configure_figure_fonts();fig,axes=plt.subplots(2,2,figsize=(7.1,4.5))
    colors=['#242424','#3b78a5','#b06c31','#69814f']
    labels=['Local + context\n+ fine','Without\nlocal','Without\ncontext','Without\nfine']
    for col,(axis,title) in enumerate([('crypt_villus','Crypt–villus coordinate'),('epithelial_distance','Epithelial-distance coordinate')]):
        frame=table.loc[table.axis.eq(axis)].set_index('condition').loc[order]
        for row,(metric,ylabel) in enumerate([('pearson_r','Pearson r'),('mae','MAE')]):
            ax=axes[row,col];values=frame[metric].to_numpy();spread=max(np.ptp(values),.004)
            ax.scatter(range(4),values,c=colors,s=37,zorder=3)
            for i,value in enumerate(values):ax.text(i,value+spread*.13,f'{value:.3f}',ha='center',fontsize=8)
            ax.set(xticks=range(4),xticklabels=labels,xlim=(-.5,3.5),
                   ylim=(values.min()-.35*spread,values.max()+.5*spread),ylabel=ylabel)
            if row==0:ax.set_title(title,fontsize=9)
            ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',length=0,labelsize=7)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4));ax.grid(axis='y',color='#eeeeee',lw=.5,zorder=0)
    fig.suptitle('Matched branch retraining',fontsize=10,y=.995)
    fig.subplots_adjust(top=.88,bottom=.15,wspace=.28,hspace=.65)
    save_pair(fig,stem,a.dpi);plt.close(fig)
    write_provenance(provenance,script=Path(__file__),inputs=[file_record(a.release,'matched_retraining_release'),
        file_record(audit['path'],'matching_audit'),file_record(rec['path'],'frozen_metrics')],outputs=outputs,
        overwritten=overwritten,extra={'panel':'Extended Figure 3F',
        'scope':release['scope'],'no_error_bars':'One matched training seed; cells are not independent replicates',
        'seed':release['seed'],'training_performed_by_renderer':False})


if __name__=='__main__':main()
