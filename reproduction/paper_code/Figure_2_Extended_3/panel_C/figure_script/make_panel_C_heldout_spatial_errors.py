#!/usr/bin/env python3
"""Show manual reference, matched predictions and errors in one fixed field."""
import sys
from paper_paths import Path, lock_record, rendering_script
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from Figure_2_Extended_3.common import *


def main():
    a=parser(__doc__).parse_args();release,table=load_table(a.release,'spatial_field')
    spec=release['spatial_example'];baseline=NAMES[spec['selected_baseline']]
    configure_figure_fonts();fig,axes=plt.subplots(2,5,figsize=(10.6,4.5))
    x0,y0,x1,y1=spec['bounds_xyxy']
    titles=['Manual reference','Full graph',baseline,'Full graph\nabsolute error',baseline+'\nabsolute error']
    for row,(coordinate,title) in enumerate(AXES):
        images=[]
        for col,suffix in enumerate(['reference','full','baseline','full_absolute_error','baseline_absolute_error']):
            ax=axes[row,col];field=f'{coordinate}_{suffix}'
            cmap=('viridis' if row==0 else 'coolwarm') if col<3 else 'magma'
            artist=ax.scatter(table.x,table.y,c=table[field],cmap=cmap,norm=Normalize(0,1),s=5,linewidths=0,rasterized=False)
            ax.set(xlim=(x0,x1),ylim=(y1,y0),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if row==0:ax.set_title(titles[col],fontsize=8)
            if col==0:ax.set_ylabel(title,fontsize=8)
            images.append(artist)
        fig.colorbar(images[0],ax=axes[row,:3],fraction=.012,pad=.02,ticks=[0,.5,1],label='Coordinate')
        fig.colorbar(images[3],ax=axes[row,3:],fraction=.017,pad=.02,ticks=[0,.5,1],label='Absolute error')
    fig.suptitle(f'Held-out annotated field · {spec["budget"]} training villi',fontsize=10,y=.99)
    fig.subplots_adjust(top=.81,bottom=.10,left=.04,right=.97,hspace=.17,wspace=.1)
    export(fig,a,'panel_C_heldout_spatial_errors','C',release,Path(__file__));plt.close(fig)


if __name__=='__main__':main()
