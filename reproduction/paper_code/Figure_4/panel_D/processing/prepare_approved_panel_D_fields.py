#!/usr/bin/env python3
"""Freeze the author's two chosen fields; subset saved predictions without inference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Figure_4.panel_D.processing.prepare_panel_D_field_candidates import record, verify, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    choice=json.loads(a.selection.read_text())
    if choice['status']!='AUTHOR_APPROVED' or choice['selected']!={'DMSO':'Current_DMSO','RARi':'R1'}:
        raise RuntimeError('Expected explicit author-approved pair')
    review=json.loads(verify(choice['review_manifest']).read_text())
    prior=json.loads(verify(review['source_manifest']).read_text())
    inputs=[record(a.selection),choice['review_manifest'],review['source_manifest']]
    for key in ('release','channel_manifest','channel_key','gates','checkpoint'):
        verify(prior[key]);inputs.append(prior[key])
    a.output_dir.mkdir(parents=True)
    windows={};outputs=[]
    for condition,ident in choice['selected'].items():
        selected=next(w for w in review['windows'] if w['id']==ident)
        bounds=selected['bounds_xxyy_px']
        if selected['condition']!=condition or bounds!=choice['bounds_xxyy_px'][condition]:
            raise RuntimeError('Selection bounds/condition changed')
        raw=np.load(verify(selected['microscopy']))
        inputs.append(selected['microscopy'])
        source=review['predictions'][condition]
        full=pd.read_parquet(verify(source));inputs.append(source)
        x0,x1,y0,y1=bounds
        cells=full.loc[full.centroid_x_fullres_px.ge(x0)&full.centroid_x_fullres_px.lt(x1)&
                       full.centroid_y_fullres_px.ge(y0)&full.centroid_y_fullres_px.lt(y1)].copy()
        if cells.empty or cells.duplicated(['source_id','object_id']).any():
            raise RuntimeError('Empty field or duplicate cells')
        cells['is_P14']=cells['class'].eq('Transferred')
        target=a.output_dir/f'{condition}_cells.parquet'
        cells.to_parquet(target,index=False)
        item={'bounds_xxyy_px':bounds,'selected_review_id':ident,'source_predictions':source,
              'cells':record(target),'n':len(cells),'P14_n':int(cells.is_P14.sum()),
              'gate_counts':{str(k):int(v) for k,v in cells.coordinate_gate.value_counts().items()}}
        outputs.append(item['cells'])
        for source_key,target_key in (('dapi','DAPI'),('ecad','E_cadherin')):
            image=raw[source_key]
            assert image.shape==(y1-y0,x1-x0) and image.dtype==np.uint16
            target=a.output_dir/f'{condition}_{target_key}.npy'
            np.save(target,image)
            item[target_key]=record(target);outputs.append(item[target_key])
        # The entire DMSO field must be byte-exact to the previous original crop.
        if condition=='DMSO':
            ox0,ox1,oy0,oy1=prior['windows'][condition]['bounds_xxyy_px']
            assert (x0,x1)==(ox0,ox1) and oy0<=y0<y1<=oy1
            for source_key,target_key in (('dapi','DAPI'),('ecad','E_cadherin')):
                old=np.load(verify(prior['windows'][condition][target_key]),mmap_mode='r')
                np.testing.assert_array_equal(raw[source_key],old[y0-oy0:y1-oy0])
        windows[condition]=item
    result={key:prior[key] for key in ('release','channel_manifest','channel_key','gates','checkpoint','pixel_size_um','display','P14_rule')}
    result.update(status='FROZEN_REGISTERED_GATE_BACKPROJECTION',
        created_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ('numpy','pandas','pyarrow')},
        git_state='Project is not a Git repository',random_seed=None,
        code=[record(__file__)],inputs=inputs,outputs=outputs,selection=choice,
        selection_record=record(a.selection),windows=windows,
        gate_assignments_changed=False,boundaries_changed=False,separate_condition_boundaries=False,
        same_bounds_in_main_and_extended=True,new_field_selection=True,model_inference_run=False,
        scope=choice['selection_scope'],
        warnings=['Qualitative fields were selected after reviewing predictions and microscopy.',
                  'No independent muscularis annotations; these images do not establish compartment accuracy.',
                  'Original IF specimens contributed to model development; cohort summaries and evaluations are unchanged.'])
    save_json(a.output_dir/'manifest.json',result)
    print(json.dumps(record(a.output_dir/'manifest.json'),indent=2))


if __name__=='__main__':main()
