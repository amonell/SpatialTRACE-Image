#!/usr/bin/env python3
"""Check frozen values and all review pages; export contact sheets for inspection."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import sys
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
from extend_potential_plot_set import check_pdf, record, verified


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rendering',type=Path,required=True)
    p.add_argument('--data-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    rendering=json.loads(a.rendering.read_text())
    release=json.loads(a.data_manifest.read_text())
    for rec in rendering['inputs']+rendering['outputs']+release['inputs']+release['outputs']:
        verified(rec)
    files={Path(r['path']).name:Path(r['path']) for r in release['outputs']}
    long=pd.read_csv(files['density_long.csv']); mice=pd.read_csv(files['mice.csv'])
    summary=pd.read_csv(files['group_summaries.csv']); changes=pd.read_csv(files['experiment_composition_changes.csv'])
    assert len(long)==45 and mice.mouse_id.nunique()==len(mice)==9
    for mouse in mice.itertuples():
        sub=long.loc[long.mouse_id.eq(mouse.mouse_id)].set_index('gate')
        np.testing.assert_allclose(sub.density,10000*sub.n_p14/sub.n_non_p14)
        np.testing.assert_allclose(sub.p14_composition_percent,100*sub.n_p14/sub.n_p14.sum())
        np.testing.assert_allclose(sub.non_p14_composition_percent,100*sub.n_non_p14/sub.n_non_p14.sum())
        np.testing.assert_allclose(sub.observed_over_expected_p14_density,
                                   (sub.n_p14/sub.n_p14.sum())/(sub.n_non_p14/sub.n_non_p14.sum()))
        np.testing.assert_allclose(sub.row_normalized_density,sub.density/sub.density.mean())
        ratio=sub.loc['Muscularis','density']/sub.loc['Top LP','density']
        np.testing.assert_allclose([mouse.muscularis_top_lp_density_ratio,mouse.log2_muscularis_top_lp_density_ratio],
                                   [ratio,np.log2(ratio)])
        np.testing.assert_allclose(mouse.muscularis_minus_top_lp_enrichment,
                                   sub.loc['Muscularis','observed_over_expected_p14_density']-sub.loc['Top LP','observed_over_expected_p14_density'])
        np.testing.assert_allclose(mouse.other_composition,100-mouse.top_lp_composition-mouse.muscularis_composition)
    for row in summary.itertuples():
        sub=long.loc[long.gate.eq(row.gate)&long.condition.eq(row.condition)]
        if row.cohort!='all': sub=sub.loc[sub.cohort.eq(row.cohort)]
        x=sub[row.metric].to_numpy()
        assert len(x)==row.n
        np.testing.assert_allclose(x.mean(),row.mean)
        if len(x)>1: np.testing.assert_allclose(x.std(ddof=1)/np.sqrt(len(x)),row.sem)
        else: assert np.isnan(row.sem)
    for row in changes.itertuples():
        sub=long.loc[long.gate.eq(row.gate)&long.cohort.eq(row.cohort)]
        expected=sub.loc[sub.condition.eq('RARi'),'p14_composition_percent'].mean()-sub.loc[sub.condition.eq('DMSO'),'p14_composition_percent'].mean()
        np.testing.assert_allclose(expected,row.difference_rari_minus_dmso)
    # Explicitly check the fixed display bounds used by the non-heatmap panels.
    assert 0<=mice.muscularis_top_lp_density_ratio.min()<=mice.muscularis_top_lp_density_ratio.max()<1.65
    assert mice.top_lp_enrichment.max()<3.5 and mice.muscularis_enrichment.max()<2
    assert -2.8<mice.muscularis_minus_top_lp_enrichment.min()<=mice.muscularis_minus_top_lp_enrichment.max()<.65
    assert long.p14_composition_percent.max()<55 and long.non_p14_composition_percent.max()<55
    assert long.density.min()>10 and long.density.max()<3000
    assert long.observed_over_expected_p14_density.max()<3.5
    book_path=verified(rendering['review_pdf'])
    font_checks=[check_pdf(Path(page['path'])) for page in rendering['page_order']]
    checks=[]; previews=[]
    with pymupdf.open(book_path) as book:
        assert len(book)==len(book.get_toc())==20
        for i,page in enumerate(rendering['page_order']):
            with pymupdf.open(page['path']) as source:
                assert len(source)==1
                original=source[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
                copied=book[i].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
                assert (original.width,original.height,original.samples)==(copied.width,copied.height,copied.samples)
                checks.append({'page':i+1,'source':record(page['path']),'status':'PIXEL_IDENTICAL'})
                pix=book[i].get_pixmap(matrix=pymupdf.Matrix(1.4,1.4),alpha=False)
                previews.append(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
    a.output.mkdir(parents=True,exist_ok=False)
    contacts=[]
    for first in range(0,20,4):
        group=previews[first:first+4]; w,h=group[0].size
        sheet=Image.new('RGB',(w*2,h*2),'#EEEEEE')
        for j,im in enumerate(group): sheet.paste(im,((j%2)*w,(j//2)*h))
        path=a.output/f'contact_{first+1:02d}_{first+4:02d}.png'; sheet.save(path); contacts.append(record(path))
    report={'status':'AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_PENDING','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,*sys.argv],'code':record(__file__),
            'inputs':[record(a.rendering),record(a.data_manifest)],'review_pdf':record(book_path),
            'data_checks':'All densities, normalizations, per-mouse contrasts, means, SEM and experiment changes independently reproduced.',
            'fixed_axis_bounds':'All plotted values inside declared bounds.','page_checks':checks,'font_checks':font_checks,
            'contact_sheets':contacts,'paper_figures_modified':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'pages':len(checks),'contact_sheets':[r['path'] for r in contacts]}))


if __name__=='__main__': main()
