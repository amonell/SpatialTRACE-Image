#!/usr/bin/env python3
"""Compare component contributions with paired held-out section points."""
import sys
from paper_paths import Path, lock_record, rendering_script
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from Figure_2_Extended_3.common import *


def main(panel='A'):
    a=parser(__doc__).parse_args();release,table=load_table(a.release,'component_metrics')
    order=['full','cell_only','uniform','randomized']
    configure_figure_fonts();fig,axes=plt.subplots(2,2,figsize=(7.1,5.2))
    sections=sorted(table.test_section.unique())
    if len(sections)!=8 or len(table)!=64:raise RuntimeError('Expected all eight sections and four comparators')
    for col,(axis,title) in enumerate(AXES):
        data=table.loc[table.axis.eq(axis)]
        for row,(metric,label) in enumerate([('pearson_r','Pearson r'),('mae','MAE')]):
            ax=axes[row,col]
            pairs=data.pivot(index='test_section',columns='variant',values=metric).loc[sections,order]
            if pairs.isna().any().any():raise RuntimeError('Missing section pair')
            for i,section in enumerate(sections):
                offset=(i-3.5)*.016
                ax.plot(np.arange(4)+offset,pairs.loc[section],color='#bdbdbd',lw=.55,alpha=.75,zorder=1)
                ax.scatter(np.arange(4)+offset,pairs.loc[section],c=[COLORS[x] for x in order],s=16,alpha=.8,zorder=2)
            for i,variant in enumerate(order):
                ax.plot([i-.2,i+.2],[pairs[variant].mean()]*2,color=COLORS[variant],lw=2.5,zorder=3)
            ax.set(xticks=range(4),xticklabels=['Full graph','Cell only','Uniform\nneighbors','Randomized\nconnections'],ylabel=label)
            if row==0:ax.set_title(title,fontsize=9)
            ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',length=0,labelsize=7.5)
            ax.grid(axis='y',color='#eeeeee',lw=.5)
    fig.suptitle('Spatial information and learned attention',fontsize=10)
    fig.subplots_adjust(top=.87,bottom=.14,wspace=.32,hspace=.65)
    export(fig,a,f'panel_{panel}_graph_component_comparison',panel,release,Path(__file__));plt.close(fig)


if __name__=='__main__':main()
