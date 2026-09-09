#!/usr/bin/env python3
"""Render matched annotation-budget curves from frozen section-level scores."""
import sys
from paper_paths import Path, lock_record, rendering_script
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from Figure_2_Extended_3.common import *


def validate_budget_table(table):
    keys=['variant','axis','test_section','n_train_villi']
    sections=sorted(table.test_section.unique())
    expected=pd.MultiIndex.from_product([
        ['full','cell_only','uniform'],[axis for axis,_ in AXES],sections,
        [1,5,10,15,20,25,30,35,40,45]],names=keys)
    actual=pd.MultiIndex.from_frame(table[keys])
    if (len(sections)!=8 or len(table)!=480 or actual.has_duplicates
            or len(expected.difference(actual)) or len(actual.difference(expected))
            or not np.isfinite(table[['pearson_r','mae']].to_numpy()).all()):
        raise RuntimeError('Expected matched scores for three models, ten budgets, eight sections and two coordinates')


def main(panel='B',release_sha256=None):
    a=parser(__doc__).parse_args()
    if release_sha256:require_file(a.release,release_sha256)
    release,table=load_table(a.release,'budget_metrics')
    validate_budget_table(table)
    configure_figure_fonts();fig,axes=plt.subplots(2,2,figsize=(7.1,4.9))
    for col,(axis,title) in enumerate(AXES):
        for row,(metric,label) in enumerate([('pearson_r','Pearson r'),('mae','MAE')]):
            ax=axes[row,col]
            for variant in ['full','cell_only','uniform']:
                frame=table.loc[table.axis.eq(axis)&table.variant.eq(variant)]
                grouped=frame.groupby('n_train_villi')[metric].agg(['mean','sem','count']).sort_index()
                if not grouped['count'].eq(8).all():raise RuntimeError('Incomplete matched budget')
                xs=grouped.index.to_numpy(dtype=float);mean=grouped['mean'].to_numpy();sem=grouped['sem'].to_numpy()
                ax.plot(xs,mean,'o-',color=COLORS[variant],lw=1.5,ms=3,label=NAMES[variant])
                ax.fill_between(xs,mean-sem,mean+sem,color=COLORS[variant],alpha=.13,linewidth=0)
            ax.set(xlabel='Annotated training villi',ylabel=label,xticks=[1,10,20,30,40,45])
            if row==0:ax.set_title(title,fontsize=9)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#eeeeee',lw=.5)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig.suptitle('Matched annotation requirements',fontsize=10)
    fig.subplots_adjust(top=.87,bottom=.19,wspace=.3,hspace=.48)
    export(fig,a,f'panel_{panel}_matched_annotation_budgets',panel,release,Path(__file__));plt.close(fig)


if __name__=='__main__':main()
