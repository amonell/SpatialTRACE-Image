#!/usr/bin/env python3
"""Verify exact data identity and unchanged figures for numbered RARi options."""
import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import subprocess
import sys

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Figure_4.panel_D.processing.prepare_panel_D_field_candidates import verify, record, save_json


def check(ok,message):
    if not ok:raise RuntimeError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-manifest',type=Path,required=True)
    p.add_argument('--render-dir',type=Path,required=True)
    p.add_argument('--expected-count',type=int,default=10)
    p.add_argument('--first-option',type=int,default=1)
    p.add_argument('--exclude-plan',type=Path,action='append',default=[],help='Additional earlier fields that must not be repeated')
    a=p.parse_args()
    manifest=json.loads(a.candidate_manifest.read_text())
    provenance=json.loads((a.render_dir/'provenance.json').read_text())
    for rec in provenance['inputs']+provenance['outputs']+provenance['current_figures_unchanged']:
        verify(rec)
    verify(provenance['active_field_pointer_unchanged'])
    check(len(provenance['current_figures_unchanged'])==8,'Expected eight unchanged figures')
    check(len(manifest['windows'])==a.expected_count,'Unexpected option count')
    check(manifest['selection_plan'].get('first_option',1)==a.first_option,'Incorrect first option')
    check([w['option'] for w in provenance['candidates']]==list(range(a.first_option,a.first_option+a.expected_count)),
          'Incorrect exported option numbering')
    frame=pd.read_parquet(verify(manifest['predictions']['RARi']))
    previous=json.loads((ROOT/'Figure_4/panel_D/processing/field_review_plan_20261016_v2.json').read_text())
    old={tuple(w['bounds_xxyy_px']) for w in previous['windows'] if w['condition']=='RARi'}
    for plan_path in a.exclude_plan:
        old.update(tuple(w['bounds_xxyy_px']) for w in json.loads(plan_path.read_text())['windows'] if w['condition']=='RARi')
    checks=[]
    for number,window in enumerate(manifest['windows'],a.first_option):
        check(window['condition']=='RARi','Incorrect condition')
        x0,x1,y0,y1=window['bounds_xxyy_px']
        check((x1-x0,y1-y0)==(6000,4500),'Changed field size')
        check(tuple(window['bounds_xxyy_px']) not in old,'Repeated old field')
        expected=frame.loc[frame.centroid_x_fullres_px.ge(x0)&frame.centroid_x_fullres_px.lt(x1)&
                           frame.centroid_y_fullres_px.ge(y0)&frame.centroid_y_fullres_px.lt(y1)].reset_index(drop=True)
        actual=pd.read_parquet(a.render_dir/'displayed_cells'/f'Option_{number:02d}.parquet')
        pd.testing.assert_frame_equal(actual,expected,check_exact=True)
        with np.load(verify(window['microscopy'])) as raw:
            for channel in ('dapi','ecad'):
                check(raw[channel].shape==(4500,6000),'Incorrect native shape')
                check(raw[channel].dtype==np.uint16,'Incorrect native type')
        pdf=a.render_dir/'individual_PDFs'/f'Option_{number:02d}.pdf'
        pages=PdfReader(pdf).pages;check(len(pages)==1,'Candidate must be one page')
        text=pages[0].extract_text()
        check(f'Option {number} | RARi' in text,'Incorrect candidate page heading')
        for title in ('DAPI / E-cadherin','Cells by IMAP gate','Crypt-villus prediction','Epithelial-distance prediction'):
            check(title in text,'Missing candidate view')
        font_lines=subprocess.run(['pdffonts',str(pdf)],check=True,capture_output=True,text=True).stdout.splitlines()[2:]
        check(bool(font_lines),'Missing vector text')
        for line in font_lines:
            check('Arial' in line and line.rsplit(None,5)[1]=='yes','Unembedded or non-Arial font')
        checks.append({'option':number,'cells':len(actual),'all_source_columns_exact':True,
                       'native_DAPI_Cy5_shape_and_type_verified':True,'new_field':True,'four_views_present':True})
    pages=PdfReader(a.render_dir/f'Figure4D_RARi_{a.expected_count}_options.pdf').pages
    check(len(pages)==(a.expected_count+4)//5+1+a.expected_count,'Incorrect contact/locator/detail page count')
    save_json(a.render_dir/'data_identity_audit.json',{
        'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
        'command':[sys.executable,*sys.argv],'script':record(__file__),
        'candidate_manifest':record(a.candidate_manifest),'provenance':record(a.render_dir/'provenance.json'),
        'checks':checks,'pdf_pages':len(pages),'all_current_figures_unchanged':True,
        'excluded_prior_plans':[record(path) for path in a.exclude_plan],
        'active_field_choice_unchanged':True,'scope':'Data identity and rendering QA, not measured anatomical accuracy'})
    print(f'PASS: {a.expected_count} RARi options; exact frozen cell values; native DAPI/Cy5; eight current PDFs and field pointer unchanged')


if __name__=='__main__':main()
