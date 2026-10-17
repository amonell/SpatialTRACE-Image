#!/usr/bin/env python3
"""Render numbered RARi options without changing either published Figure 4."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from Figure_4.common import configure_figure_fonts
from Figure_4.spatial import GATE_COLORS, GATE_ORDER
from Figure_4.panel_D.figure_script.make_panel_D_field_review import composite, draw_cells, field_page
from Figure_4.panel_D.processing.prepare_panel_D_field_candidates import record, verify, save_json

DATA = Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript')

# Scatter sizes are areas in pt². Scale diameters, preserving the larger P14
# markers and leaving positions, colors, normalization and microscopy intact.
CONTACT_DOT_DIAMETER_MULTIPLIER = 2.5
DETAIL_DOT_DIAMETER_MULTIPLIER = 2.0
LOCATOR_DOT_DIAMETER_MULTIPLIER = 2.0


def enlarge_scatter_dots(figure, diameter_multiplier):
    count = 0
    for axis in figure.axes:
        for artist in axis.collections:
            if isinstance(artist, PathCollection):
                artist.set_sizes(artist.get_sizes() * diameter_multiplier**2)
                count += 1
    if count != 4:
        raise RuntimeError(f'Expected gate cells, P14, CV and epithelial scatter layers; found {count}')


def selected_rows(frame, window):
    x0,x1,y0,y1 = window['bounds_xxyy_px']
    return frame.loc[frame.centroid_x_fullres_px.ge(x0)&frame.centroid_x_fullres_px.lt(x1)&
                     frame.centroid_y_fullres_px.ge(y0)&frame.centroid_y_fullres_px.lt(y1)].copy()


def contact(windows, frame, manifest, offset):
    fig = plt.figure(figsize=(11,14.7))
    fig.text(.035,.978,f'RARi fields · options {offset+1}–{offset+len(windows)}',fontsize=17,weight='bold',va='top')
    for col,title in enumerate(('DAPI / E-cadherin','Crypt-villus prediction','Cells by IMAP gate')):
        fig.text(.215+col*.31,.943,title,ha='center',fontsize=11)
    last = None
    for row,window in enumerate(windows):
        x0,x1,y0,y1 = window['bounds_xxyy_px']
        with np.load(verify(window['microscopy'])) as raw:
            image = composite(raw['dapi'][::6,::6],raw['ecad'][::6,::6],manifest['display'])
        cells = selected_rows(frame,window)
        top = .925-row*.167
        fig.text(.023,top-.007,str(offset+row+1),fontsize=12,weight='bold',va='top')
        for col,kind in enumerate(('image','cv','gates')):
            ax = fig.add_axes([.06+col*.31,top-.15,.29,.15])
            if kind in ('image','gates'):
                ax.imshow(image,extent=(x0,x1,y1,y0),interpolation='nearest')
            if kind != 'image':
                artist = draw_cells(ax,cells,kind,size=(.7 if kind=='gates' else .9)*CONTACT_DOT_DIAMETER_MULTIPLIER**2)
                if kind == 'cv':last = artist
                else:
                    draw_cells(ax,cells.loc[cells['class'].eq('Transferred')],kind,size=2.5*CONTACT_DOT_DIAMETER_MULTIPLIER**2)
            ax.set(xlim=(x0,x1),ylim=(y1,y0),aspect='equal');ax.axis('off')
            xbar = x0+.035*(x1-x0);length=200/manifest['pixel_size_um']
            ax.plot([xbar,xbar+length],[-.025,-.025],transform=ax.get_xaxis_transform(),clip_on=False,lw=1,color='black')
            ax.text(xbar+length/2,-.04,'200 µm',ha='center',va='top',fontsize=6.5,transform=ax.get_xaxis_transform())
    cax=fig.add_axes([.40,.062,.22,.008])
    bar=fig.colorbar(last,cax=cax,orientation='horizontal',ticks=[0,.5,1])
    bar.ax.tick_params(labelsize=8,length=2,pad=2);bar.set_label('Crypt-villus coordinate',fontsize=9,labelpad=2)
    handles=[Line2D([],[],ls='',marker='o',color=GATE_COLORS[g],markersize=5,label=g) for g in GATE_ORDER]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=5,frameon=False,fontsize=9)
    return fig


def locator(frame, windows, current_bounds, first_option=1):
    fig,ax=plt.subplots(figsize=(9,9))
    draw_cells(ax,frame,'cv',size=.18*LOCATOR_DOT_DIAMETER_MULTIPLIER**2)
    for i,window in enumerate(windows,first_option):
        x0,x1,y0,y1=window['bounds_xxyy_px']
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#b34728',lw=1.2))
        ax.text(x0,y0,str(i),ha='left',va='bottom',fontsize=11,weight='bold',color='#873117',
                bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
    x0,x1,y0,y1=current_bounds
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#444444',linestyle='--',lw=1.1))
    ax.text(x1,y0,'Current R1',fontsize=9,color='#444444',va='bottom',ha='right',
            bbox=dict(facecolor='white',edgecolor='none',pad=1))
    ax.invert_yaxis();ax.set_aspect('equal')
    ax.set(xlabel='Native x (pixels)',ylabel='Native y (pixels)',title=f'RARi · locations of {len(windows)} new fields')
    fig.tight_layout(pad=1.5)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--expected-count',type=int,default=10)
    p.add_argument('--first-option',type=int,default=1)
    p.add_argument('--dpi',type=int,default=300)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    manifest=json.loads(a.candidate_manifest.read_text())
    if manifest['status']!='QUALITATIVE_FIELD_REVIEW_NOT_SELECTED_FOR_PUBLICATION':raise RuntimeError('Not a review release')
    windows=manifest['windows']
    if a.expected_count<=0 or a.first_option<=0:raise ValueError('Counts and numbering must be positive')
    if len(windows)!=a.expected_count or any(w['condition']!='RARi' for w in windows):raise RuntimeError('Unexpected RARi field count')
    if len({tuple(w['bounds_xxyy_px']) for w in windows})!=len(windows):raise RuntimeError('Duplicate fields')
    if manifest['selection_plan'].get('first_option',1)!=a.first_option:raise RuntimeError('Numbering differs from prepared plan')
    source=json.loads(verify(manifest['source_manifest']).read_text())
    if (manifest['predictions']['RARi']!=source['windows']['RARi']['source_predictions'] or
            manifest['display']!=source['display'] or manifest['pixel_size_um']!=source['pixel_size_um']):
        raise RuntimeError('Review differs from frozen prediction/channel lineage')
    frame=pd.read_parquet(verify(manifest['predictions']['RARi']))
    configure_figure_fonts()
    before=[record(q) for q in sorted((DATA/'Regenerated_Figures/Current').glob('*.pdf'))]
    pointer=record(ROOT/'Figure_4/panel_D/processing/active_fields.json')
    if len(before)!=8:raise RuntimeError('Expected eight current figure PDFs')
    a.output_dir.mkdir(parents=True)
    for directory in ('previews','individual_PDFs','displayed_cells'):(a.output_dir/directory).mkdir()
    inputs=[record(a.candidate_manifest),manifest['source_manifest'],manifest['predictions']['RARi']]
    summary=[]
    pdf_name=f'Figure4D_RARi_{len(windows)}_options.pdf'
    with PdfPages(a.output_dir/pdf_name) as packet:
        for offset in range(0,len(windows),5):
            first=a.first_option+offset
            last=min(first+4,a.first_option+len(windows)-1)
            fig=contact(windows[offset:offset+5],frame,manifest,first-1)
            packet.savefig(fig,dpi=a.dpi)
            fig.savefig(a.output_dir/'previews'/f'Options_{first:02d}_{last:02d}.png',dpi=180)
            plt.close(fig)
        fig=locator(frame,windows,manifest['selection_plan']['current_RARi_not_a_candidate'],a.first_option)
        packet.savefig(fig,dpi=a.dpi);fig.savefig(a.output_dir/'previews'/'Field_locations.png',dpi=180);plt.close(fig)
        for number,window in enumerate(windows,a.first_option):
            with np.load(verify(window['microscopy'])) as raw:
                fig,cells=field_page(window,raw,frame,manifest)
            enlarge_scatter_dots(fig, DETAIL_DOT_DIAMETER_MULTIPLIER)
            fig.texts[0].set_text(f'Option {number} | RARi')
            packet.savefig(fig,dpi=a.dpi)
            stem=f'Option_{number:02d}'
            fig.savefig(a.output_dir/'individual_PDFs'/(stem+'.pdf'),dpi=a.dpi)
            fig.savefig(a.output_dir/'previews'/(stem+'.png'),dpi=a.dpi)
            plt.close(fig)
            cells.to_parquet(a.output_dir/'displayed_cells'/(stem+'.parquet'),index=False)
            summary.append({'option':number,'id':window['id'],'bounds_xxyy_px':window['bounds_xxyy_px'],
                            'cells':len(cells),'P14_cells':int(cells['class'].eq('Transferred').sum())})
            inputs.append(window['microscopy'])
            print(f'Option {number}: {len(cells)} unchanged cells',flush=True)
    for rec in [*before,pointer]:verify(rec)
    pd.DataFrame(summary).to_csv(a.output_dir/'candidate_fields.csv',index=False)
    save_json(a.output_dir/'provenance.json',{
        'status':'AUTHOR_CHOICE_PENDING_NOT_PUBLISHED','created_utc':datetime.now(timezone.utc).isoformat(),
        'command':[sys.executable,*sys.argv],'python':platform.python_version(),
        'packages':{n:importlib.metadata.version(n) for n in ('numpy','pandas','matplotlib','pyarrow')},
        'git_state':'No Git repository','random_seed':None,
        'code':[record(__file__),record(ROOT/'Figure_4/panel_D/figure_script/make_panel_D_field_review.py'),
                record(ROOT/'Figure_4/panel_D/processing/prepare_panel_D_field_candidates.py'),
                record(ROOT/'Figure_4/spatial.py')],
        'inputs':inputs,'outputs':[record(q) for q in sorted(a.output_dir.rglob('*')) if q.is_file()],
        'current_figures_unchanged':before,'active_field_pointer_unchanged':pointer,
        'candidates':summary,'warnings':manifest['warnings'],'no_inference':True,'no_gate_changes':True,
        'first_option':a.first_option,'n_options':len(windows),'combined_pdf':pdf_name,
        'shared_color_ranges':{'cv':[0,1],'epi':[0,.5]},'shared_channel_display':manifest['display'],
        'marker_revision':{
            'reference':'figure4D_RARi_options_20261016_v2',
            'diameter_multipliers':{'contact':CONTACT_DOT_DIAMETER_MULTIPLIER,
                                   'detail':DETAIL_DOT_DIAMETER_MULTIPLIER,
                                   'locator':LOCATOR_DOT_DIAMETER_MULTIPLIER},
            'areas_pt2':{'contact_gate':.7*CONTACT_DOT_DIAMETER_MULTIPLIER**2,
                         'contact_P14':2.5*CONTACT_DOT_DIAMETER_MULTIPLIER**2,
                         'contact_cv':.9*CONTACT_DOT_DIAMETER_MULTIPLIER**2,
                         'detail_gate':3*DETAIL_DOT_DIAMETER_MULTIPLIER**2,
                         'detail_P14':10*DETAIL_DOT_DIAMETER_MULTIPLIER**2,
                         'detail_cv_and_epi':4.2*DETAIL_DOT_DIAMETER_MULTIPLIER**2,
                         'locator':.18*LOCATOR_DOT_DIAMETER_MULTIPLIER**2},
            'P14_relative_size_preserved':True,'positions_colors_and_values_unchanged':True}})
    print(a.output_dir/pdf_name,flush=True)


if __name__=='__main__':main()
