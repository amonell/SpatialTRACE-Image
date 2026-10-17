#!/usr/bin/env python3
"""Twenty alternative displays of the same frozen nine-mouse P14 dataset."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from paper_paths import Path, lock_record, rendering_script
import platform
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'frozen_preprocessing/if_replicates_20261014'))
from audit_regional_statistics import record, verified

C = {'DMSO': '#37788F', 'RARi': '#B05B49'}
E = 'observed_over_expected_p14_density'
P = 'p14_composition_percent'
TARGETS = ['Top LP', 'Muscularis']
GATE_COLORS = {'Top LP': '#C48B35', 'Muscularis': '#81579E', 'Top IE': '#DADADA',
               'Crypt IE': '#ABABAB', 'Crypt LP': '#737373'}


class Suite:
    def __init__(self, files, output):
        self.plan = json.loads(files['suite_plan.json'].read_text())
        self.gates = self.plan['gates']
        self.data = pd.read_csv(files['density_long.csv'])
        self.mice = pd.read_csv(files['mice.csv'])
        self.summary = pd.read_csv(files['group_summaries.csv'])
        self.composition = pd.read_csv(files['composition_statistics.csv'])
        self.enrichment = pd.read_csv(files['enrichment_statistics.csv'])
        self.fold = pd.read_csv(files['density_fold_change.csv']).set_index('gate')
        self.changes = pd.read_csv(files['experiment_composition_changes.csv'])
        self.output, self.outputs, self.pages = output, [], []
        (output/'pdfs').mkdir(parents=True, exist_ok=False)
        (output/'previews').mkdir()

    def figure(self, number, subtitle, note='', legend=True):
        option = self.plan['options'][number-1]
        fig = plt.figure(figsize=(9.6, 6.5))
        fig.text(.5, .955, f'{number:02d}  {option["title"]}', ha='center', va='top', fontsize=14, weight='bold')
        fig.text(.5, .888, subtitle, ha='center', fontsize=9.5)
        if legend:
            self.legend(fig)
        fig.text(.5, .115, note, ha='center', fontsize=8.2)
        fig.text(.5, .065, '4 DMSO and 5 RARi mice · all tissue removals applied · tissue 08 excluded', ha='center', fontsize=8, color='#555555')
        return fig

    def legend(self, fig, original=True):
        handles = [Line2D([], [], marker='o', ls='', color=C[c], markersize=5, label=c) for c in C]
        if original:
            handles.append(Line2D([], [], marker='s', ls='', color='#555555', markerfacecolor='white', markersize=5, label='Original experiment'))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .856), frameon=False, ncol=3, fontsize=9)

    def ax(self, fig, rect=(.12, .25, .83, .52)):
        ax = fig.add_axes(rect)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E7E7E7', lw=.6)
        ax.set_axisbelow(True)
        return ax

    def point(self, ax, x, y, row, size=35, **kwargs):
        old = row.cohort == 'original'
        ax.scatter(x, y, s=size, marker='s' if old else 'o',
                   facecolor='white' if old else C[row.condition], edgecolor=C[row.condition],
                   linewidth=1, zorder=4, **kwargs)

    def means(self, metric, gate, condition, cohort='all'):
        s = self.summary
        return s.loc[s.metric.eq(metric) & s.gate.eq(gate) & s.condition.eq(condition) & s.cohort.eq(cohort)].iloc[0]

    def dots(self, ax, metric, gates, ylabel, cohort='all', null=None):
        frame = self.data if cohort == 'all' else self.data.loc[self.data.cohort.eq(cohort)]
        for i, gate in enumerate(gates):
            for j, condition in enumerate(C):
                group = frame.loc[frame.gate.eq(gate) & frame.condition.eq(condition)].sort_values('mouse_id')
                center = i+(-.17 if j == 0 else .17)
                offsets = np.linspace(-.06, .06, len(group)) if len(group)>1 else [0]
                for offset, row in zip(offsets, group.itertuples()):
                    self.point(ax, center+offset, getattr(row, metric), row)
                s = self.means(metric, gate, condition, cohort)
                if s['n']>1:
                    ax.errorbar(center, s['mean'], yerr=s['sem'], fmt='_', color='#222222',
                                markersize=15, capsize=4, lw=1.3, zorder=5)
        ax.set(xticks=np.arange(len(gates)), xticklabels=gates, ylabel=ylabel,
               xlim=(-.55, len(gates)-.45), ylim=(0, frame.loc[frame.gate.isin(gates), metric].max()*1.15))
        if null is not None:
            ax.axhline(null, color='#777777', lw=.8, ls='--')

    def heatmap(self, number, metric, subtitle, label, raw=False):
        fig = self.figure(number, subtitle, 'Rows are individual SIs; original-experiment mice are labeled.', legend=False)
        ax = fig.add_axes([.24, .225, .57, .60])
        frame = self.data.pivot(index='mouse_id', columns='gate', values=metric).loc[self.plan['mouse_order'], self.gates]
        values = frame.to_numpy()
        if raw:
            cmap, norm, ticks = plt.get_cmap('viridis'), LogNorm(vmin=10, vmax=3000), [10, 30, 100, 300, 1000, 3000]
        else:
            maximum = max(1.5, math.ceil(float(values.max())*2)/2)
            cmap, norm, ticks = plt.get_cmap('RdBu_r'), TwoSlopeNorm(vmin=0, vcenter=1, vmax=maximum), np.arange(0, maximum+.01, .5)
        assert values.min()>=norm.vmin and values.max()<=norm.vmax
        mesh = ax.pcolormesh(np.arange(6), np.arange(10), values, norm=norm, cmap=cmap, edgecolors='white', linewidth=.8)
        rows = self.mice.set_index('mouse_id').loc[frame.index]
        ax.set(xticks=np.arange(5)+.5, yticks=np.arange(9)+.5, xticklabels=self.gates,
               yticklabels=rows.display_label, xlim=(0,5), ylim=(9,0))
        ax.xaxis.tick_top(); ax.tick_params(length=0, pad=8)
        ax.spines[:].set_visible(False)
        ax.axhline(4, color='white', lw=4)
        for tick, condition in zip(ax.get_yticklabels(), rows.condition):
            tick.set_color(C[condition])
        for i in range(9):
            for j in range(5):
                rgb = np.asarray(cmap(norm(values[i,j]))[:3])
                light = np.where(rgb<=.04045, rgb/12.92, ((rgb+.055)/1.055)**2.4) @ np.array([.2126,.7152,.0722])
                ax.text(j+.5, i+.5, f'{values[i,j]:.0f}' if raw else f'{values[i,j]:.2f}',
                        ha='center', va='center', fontsize=9, color='white' if light<.35 else '#222222')
        cb = fig.colorbar(mesh, cax=fig.add_axes([.845,.225,.02,.60]), ticks=ticks)
        cb.set_label(label, labelpad=9); cb.outline.set_visible(False)
        if raw:
            cb.ax.set_yticklabels([str(v) for v in ticks]); cb.ax.minorticks_off()
        return fig

    def save(self, number, fig):
        option = self.plan['options'][number-1]
        stem = f'Option_{number:02d}_{option["stem"]}'
        for suffix, folder in [('pdf','pdfs'),('png','previews')]:
            path = self.output/folder/f'{stem}.{suffix}'
            fig.savefig(path, dpi=300)
            self.outputs.append(record(path))
        plt.close(fig)
        self.pages.append({'number':number,'title':option['title'],'path':str(self.output/'pdfs'/f'{stem}.pdf')})

    def render(self):
        fig=self.figure(1,'Observed/expected P14 density; 1 = proportional to other-cell availability',
                        'Points: mice. Mean ± SEM. Holm-adjusted P: Top LP 0.229; muscularis 0.143 (experiment-stratified).')
        self.dots(self.ax(fig),E,TARGETS,'P14 enrichment (observed / expected)',null=1)
        self.save(1,fig)

        fig=self.figure(2,'Observed/expected P14 density in each anatomical gate','Points: mice. Black marks: mean ± SEM.')
        self.dots(self.ax(fig),E,self.gates,'P14 enrichment (observed / expected)',null=1)
        self.save(2,fig)

        fig=self.figure(3,'Percentage of each mouse’s P14 population in the indicated gate',
                        'Points: mice. Mean ± SEM. Experiment-stratified Holm-adjusted P = 0.143 for both gates.')
        self.dots(self.ax(fig),P,TARGETS,'P14 cells in gate (% of all P14 cells)')
        self.save(3,fig)

        fig=self.figure(4,'Top LP and muscularis highlighted; the remaining gates retain their full proportions',
                        'Each bar sums to 100% of that mouse’s P14 cells.',legend=False)
        ax=self.ax(fig, [.1,.32,.86,.46]); bottom=np.zeros(9)
        order=['Top LP','Top IE','Crypt IE','Crypt LP','Muscularis']
        table=self.data.pivot(index='mouse_id',columns='gate',values=P).loc[self.plan['mouse_order']]
        for gate in order:
            ax.bar(np.arange(9),table[gate],bottom=bottom,width=.72,color=GATE_COLORS[gate],edgecolor='white',lw=.5,label=gate)
            bottom+=table[gate].to_numpy()
        ax.set(ylim=(0,100),ylabel='P14 composition (%)',xticks=np.arange(9),xticklabels=self.mice.display_label)
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right',fontsize=8)
        ax.axvline(3.5,color='#777777',lw=.8)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.85),ncol=5,frameon=False,fontsize=9)
        self.save(4,fig)

        fig=self.figure(5,'Observed/expected enrichment: DMSO and RARi means','Lines join condition means, not paired mice. Whiskers: SEM.',legend=False)
        ax=self.ax(fig,[.18,.25,.75,.52])
        for i,gate in enumerate(self.gates):
            d,r=[self.means(E,gate,c) for c in C]
            ax.plot([d['mean'],r['mean']],[i,i],color='#BBBBBB',lw=2)
            for s,c in [(d,'DMSO'),(r,'RARi')]:
                ax.errorbar(s['mean'],i,xerr=s['sem'],fmt='o',color=C[c],capsize=3,ms=6,lw=1)
        ax.set(yticks=np.arange(5),yticklabels=self.gates,xlabel='P14 enrichment (observed / expected)',xlim=(0,3.2),ylim=(4.5,-.5))
        ax.axvline(1,color='#777777',lw=.8,ls='--')
        self.legend(fig,original=False); self.save(5,fig)

        fig=self.figure(6,'RARi minus DMSO, comparing treatments within each experiment',
                        'Experiment-size-weighted mean differences. Printed P values are Holm-adjusted across all five gates.',legend=False)
        ax=self.ax(fig,[.19,.25,.73,.52]); s=self.enrichment.loc[self.enrichment.analysis.eq('cohort_stratified')].set_index('gate').loc[self.gates]
        effects=-s.test_statistic.to_numpy()
        ax.barh(np.arange(5),effects,color=[GATE_COLORS[g] for g in self.gates],height=.58)
        ax.axvline(0,color='#333333',lw=.9)
        for i,(v,p) in enumerate(zip(effects,s.holm_adjusted_p)):
            ax.text(v+(.025 if v>=0 else -.025),i,f'{v:+.2f}  (P = {p:.3f})',ha='left' if v>=0 else 'right',va='center',fontsize=8.5)
        ax.set(yticks=np.arange(5),yticklabels=self.gates,ylim=(4.5,-.5),xlim=(-1.3,1.05),xlabel='Change in P14 enrichment (RARi − DMSO)')
        self.save(6,fig)

        fig=self.figure(7,'Muscularis density divided by Top LP density within the same mouse',
                        'One point per mouse. A value of 1 indicates equal cell-count-normalized density in the two gates.')
        ax=self.ax(fig,[.19,.25,.65,.52])
        for i,c in enumerate(C):
            group=self.mice.loc[self.mice.condition.eq(c)]
            for offset,row in zip(np.linspace(-.09,.09,len(group)),group.itertuples()):
                self.point(ax,i+offset,row.muscularis_top_lp_density_ratio,row,45)
        ax.set(xticks=[0,1],xticklabels=list(C),xlim=(-.5,1.5),ylim=(0,1.65),ylabel='Muscularis / Top LP density')
        ax.axhline(1,color='#777777',ls='--',lw=.8)
        self.save(7,fig)

        fig=self.figure(8,'Each point locates one mouse by its two gate enrichments',
                        'Labels identify SIs; OD and OR identify the original DMSO and RARi mice.')
        ax=self.ax(fig,[.17,.25,.68,.52])
        for row in self.mice.itertuples():
            self.point(ax,row.top_lp_enrichment,row.muscularis_enrichment,row,45)
            label=('OD' if row.condition=='DMSO' else 'OR') if row.cohort=='original' else row.source_id.rsplit('_',1)[-1]
            ax.annotate(label,(row.top_lp_enrichment,row.muscularis_enrichment),xytext=(5,5),textcoords='offset points',fontsize=8,color=C[row.condition])
        ax.axhline(1,color='#999999',lw=.7,ls='--'); ax.axvline(1,color='#999999',lw=.7,ls='--')
        ax.set(xlim=(0,3.7),ylim=(0,2),xlabel='Top LP enrichment (observed / expected)',ylabel='Muscularis enrichment (observed / expected)')
        self.save(8,fig)

        self.save(9,self.heatmap(9,E,'P14 fraction in gate divided by non-P14 fraction in gate','Observed / expected'))
        self.save(10,self.heatmap(10,'row_normalized_density','Gate density divided by the mean density across that mouse’s five gates','Density / row mean'))
        self.save(11,self.heatmap(11,'density','P14 cells per 10,000 non-P14 cells in the same gate; shared logarithmic color scale','P14 per 10,000 non-P14',raw=True))

        fig=self.figure(12,'Lower density can coexist with a larger muscularis share of P14 cells',
                        'Left: experiment-standardized density ratio. Right: experiment-stratified percentage-point difference.',legend=False)
        a=self.ax(fig,[.16,.25,.32,.52]); b=self.ax(fig,[.65,.25,.30,.52])
        f=self.fold.loc[self.gates,'fold_change_rari_over_dmso']
        a.barh(np.arange(5),f,color=[GATE_COLORS[g] for g in self.gates],height=.55)
        a.axvline(1,color='#555555',ls='--',lw=.8)
        a.set(yticks=np.arange(5),yticklabels=self.gates,ylim=(4.5,-.5),xlim=(0,1.1),xlabel='Density ratio (RARi / DMSO)')
        s=self.composition.loc[self.composition.analysis.eq('cohort_stratified')].set_index('gate').loc[self.gates]
        b.barh(np.arange(5),-s.test_statistic,color=[GATE_COLORS[g] for g in self.gates],height=.55)
        b.axvline(0,color='#555555',lw=.8)
        b.set(yticks=np.arange(5),yticklabels=self.gates,ylim=(4.5,-.5),xlim=(-12,17),xlabel='P14 share change (percentage points)')
        self.save(12,fig)

        fig=self.figure(13,'Observed/expected P14 enrichment in the two target gates',
                        'Each line joins Top LP and muscularis measurements from the same mouse.')
        ax=self.ax(fig,[.18,.25,.67,.52])
        for row in self.mice.itertuples():
            ax.plot([0,1],[row.top_lp_enrichment,row.muscularis_enrichment],color=C[row.condition],lw=1.1,alpha=.6)
            self.point(ax,0,row.top_lp_enrichment,row); self.point(ax,1,row.muscularis_enrichment,row)
        ax.axhline(1,color='#999999',ls='--',lw=.8)
        ax.set(xticks=[0,1],xticklabels=TARGETS,xlim=(-.25,1.25),ylim=(0,3.5),ylabel='P14 enrichment (observed / expected)')
        self.save(13,fig)

        fig=self.figure(14,'Both treatments are shown separately in the original and new experiments',
                        'Original: one mouse per condition. New: three DMSO and four RARi mice; mean ± SEM.')
        for rect,cohort,title in [([.10,.25,.37,.48],'original','Original experiment'),([.59,.25,.37,.48],'new','New experiment')]:
            ax=self.ax(fig,rect); self.dots(ax,E,TARGETS,'P14 enrichment (observed / expected)' if cohort=='original' else '',cohort,null=1)
            ax.set_ylim(0,3.5); ax.set_title(title,fontsize=11,pad=12)
        self.save(14,fig)

        fig=self.figure(15,'Muscularis enrichment minus Top LP enrichment, within each mouse',
                        'Rows are ordered within treatment by the displayed contrast. Zero indicates equal enrichment.',legend=False)
        ax=self.ax(fig,[.25,.23,.64,.57])
        mice=pd.concat([self.mice.loc[self.mice.condition.eq(c)].sort_values('muscularis_minus_top_lp_enrichment') for c in C])
        ax.barh(np.arange(9),mice.muscularis_minus_top_lp_enrichment,color=[C[c] for c in mice.condition],height=.65,alpha=.7)
        ax.set(yticks=np.arange(9),yticklabels=mice.display_label,ylim=(8.5,-.5),xlim=(-2.8,.65),xlabel='Muscularis − Top LP enrichment')
        ax.axvline(0,color='#555555',lw=.9); ax.axhline(3.5,color='#BBBBBB',lw=.8)
        self.save(15,fig)

        fig=self.figure(16,'Each mouse’s P14 population partitioned into three compartments',
                        'Top LP + muscularis + all other gates = 100% for each mouse.')
        ax=fig.add_axes([.18,.24,.64,.55]); h=np.sqrt(3)/2
        ax.plot([0,1,.5,0],[0,0,h,0],color='#777777',lw=1)
        for q in [.25,.5,.75]:
            ax.plot([q/2,1-q/2],[q*h,q*h],color='#DDDDDD',lw=.7)
            ax.plot([q,q+(1-q)/2],[0,(1-q)*h],color='#DDDDDD',lw=.7)
            ax.plot([1-q,(1-q)/2],[0,(1-q)*h],color='#DDDDDD',lw=.7)
        for row in self.mice.itertuples():
            self.point(ax,row.muscularis_composition/100+.5*row.other_composition/100,h*row.other_composition/100,row,50)
        ax.text(0,-.07,'Top LP 100%',ha='center',fontsize=10)
        ax.text(1,-.07,'Muscularis 100%',ha='center',fontsize=10)
        ax.text(.5,h+.035,'Other gates 100%',ha='center',fontsize=10)
        ax.set(xlim=(-.15,1.15),ylim=(-.11,h+.1)); ax.set_aspect('equal'); ax.axis('off')
        self.save(16,fig)

        fig=self.figure(17,'RARi minus DMSO in each experiment, as percentage points of total P14',
                        'Each point is an experiment-specific difference between condition means.',legend=False)
        ax=self.ax(fig,[.18,.26,.70,.50])
        for j,cohort in enumerate(['original','new']):
            for i,gate in enumerate(TARGETS):
                val=self.changes.loc[self.changes.cohort.eq(cohort)&self.changes.gate.eq(gate),'difference_rari_minus_dmso'].iloc[0]
                y=i+(-.09 if j==0 else .09)
                ax.plot([0,val],[y,y],color=GATE_COLORS[gate],lw=2,alpha=.5)
                ax.scatter(val,y,s=55,marker='s' if j==0 else 'o',facecolor='white' if j==0 else GATE_COLORS[gate],edgecolor=GATE_COLORS[gate],zorder=3)
        ax.axvline(0,color='#777777',lw=.8)
        ax.set(yticks=[0,1],yticklabels=TARGETS,ylim=(1.4,-.4),xlim=(-18,21),xlabel='P14 share change (RARi − DMSO; percentage points)')
        fig.legend(handles=[Line2D([],[],marker='s',ls='',color='#555555',markerfacecolor='white',label='Original experiment'),
                            Line2D([],[],marker='o',ls='',color='#555555',label='New experiment')],loc='upper center',bbox_to_anchor=(.5,.85),ncol=2,frameon=False,fontsize=9)
        self.save(17,fig)

        fig=self.figure(18,'P14 cells per 10,000 non-P14 cells in the same gate',
                        'Points: mice. Mean ± SEM. This abundance measure is lower under RARi in both gates.')
        self.dots(self.ax(fig),'density',TARGETS,'P14 per 10,000 non-P14 cells')
        self.save(18,fig)

        fig=self.figure(19,'P14 fraction versus non-P14 fraction in the same gate',
                        'Dashed line: equal fractions. Above the line indicates enrichment relative to non-P14 cells.')
        for rect,gate in [([.10,.25,.37,.48],'Top LP'),([.59,.25,.37,.48],'Muscularis')]:
            ax=self.ax(fig,rect)
            for row in self.data.loc[self.data.gate.eq(gate)].itertuples():
                self.point(ax,row.non_p14_composition_percent,row.p14_composition_percent,row,40)
            ax.plot([0,55],[0,55],ls='--',lw=.8,color='#777777')
            ax.set(xlim=(0,55),ylim=(0,55),xlabel='Non-P14 cells in gate (%)',ylabel='P14 cells in gate (%)' if gate=='Top LP' else '')
            ax.set_title(gate,fontsize=11,pad=12)
        self.save(19,fig)

        fig=self.figure(20,'Color: observed/expected enrichment. Circle area: fraction of the P14 population.',
                        'Each row is one mouse; original-experiment mice are labeled.',legend=False)
        ax=self.ax(fig,[.24,.22,.55,.58]); cmap=plt.get_cmap('RdBu_r'); norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=3.5)
        for i,mouse in enumerate(self.plan['mouse_order']):
            for j,gate in enumerate(self.gates):
                row=self.data.loc[self.data.mouse_id.eq(mouse)&self.data.gate.eq(gate)].iloc[0]
                mesh=ax.scatter(j,i,s=row[P]*9,c=[row[E]],cmap=cmap,norm=norm,edgecolor='#777777',linewidth=.4)
        ax.set(xticks=np.arange(5),xticklabels=self.gates,yticks=np.arange(9),yticklabels=self.mice.display_label,xlim=(-.6,4.6),ylim=(8.6,-.6))
        ax.xaxis.tick_top(); ax.tick_params(length=0,pad=8); ax.spines[:].set_visible(False)
        ax.axhline(3.5,color='#BBBBBB',lw=.8)
        cb=fig.colorbar(mesh,cax=fig.add_axes([.835,.42,.022,.38])); cb.set_label('Observed / expected',labelpad=8)
        handles=[ax.scatter([],[],s=v*9,facecolor='#CCCCCC',edgecolor='#777777',lw=.4,label=f'{v}%') for v in [10,25,50]]
        fig.legend(handles=handles,title='P14 share',loc='center',bbox_to_anchor=(.875,.285),frameon=False,fontsize=8,title_fontsize=9,labelspacing=1.4)
        self.save(20,fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    release=json.loads(verified({'path':str(a.manifest),'sha256':a.manifest_sha256}).read_text())
    assert release['status']=='frozen_twenty_p14_visualization_options'
    files={Path(r['path']).name:verified(r) for r in release['outputs']}
    for weight in ['normal','bold']: findfont(FontProperties(family='Arial',weight=weight),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,'savefig.facecolor':'white'})
    suite=Suite(files,a.output); suite.render(); assert len(suite.pages)==20
    writer=PdfWriter()
    for page in suite.pages: writer.append(PdfReader(page['path']),outline_item=f'{page["number"]:02d} {page["title"]}')
    writer.add_metadata({'/Title':'Twenty P14 regional visualization options','/Author':'SpatialTRACE'})
    book=a.output/'Twenty_P14_visualization_options.pdf'
    with book.open('xb') as handle: writer.write(handle)
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
                'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','matplotlib','pypdf']},
                'git_state':'Not a Git worktree','random_seed':None,
                'inputs':[record(a.manifest),*release['outputs'],record(__file__),record(ROOT/'frozen_preprocessing/if_replicates_20261014/audit_regional_statistics.py')],
                'outputs':suite.outputs,'review_pdf':record(book),'page_order':suite.pages,'warnings':release['warnings'],
                'tests_computed_during_render':False,'paper_figures_modified':False}
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'pdf':str(book),'individual_pdfs':str(a.output/'pdfs'),'pages':20}))


if __name__=='__main__': main()
