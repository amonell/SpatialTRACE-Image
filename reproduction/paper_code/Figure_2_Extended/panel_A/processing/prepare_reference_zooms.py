#!/usr/bin/env python3
"""Freeze sparse reference labels inside the already locked prediction windows."""
import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from Figure_2_Extended.common import require_file, file_record

EXPECTED_LABELS='7fd1c37966880be09e6fd402e810b7199358b4f9e20bad31da54abbcca522843'
EXPECTED_WINDOWS='23a62f690d722b1302667b4fed47742d5fe89487b270d47e46e119460f2b899e'


def labels_in_window(labels,batch,bounds):
    x0,x1,y0,y1=bounds
    return labels.loc[labels.batch.eq(batch)&labels.x.between(x0,x1)&labels.y.between(y0,y1)].copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sparse-labels',type=Path,required=True)
    p.add_argument('--prediction-zoom-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    label_path=require_file(a.sparse_labels,EXPECTED_LABELS)
    window_path=require_file(a.prediction_zoom_manifest,EXPECTED_WINDOWS)
    prediction=json.loads(window_path.read_text())
    labels=pd.read_parquet(label_path);frames=[];windows={}
    for batch,window in prediction['windows'].items():
        frame=labels_in_window(labels,batch,window['bounds_xxyy_um'])
        if frame.empty:raise RuntimeError('Locked prediction window has no reference labels: '+batch)
        windows[batch]={'bounds_xxyy_um':window['bounds_xxyy_um'],
                        'scale_bar_um':window['scale_bar_um'],'reference_n':len(frame),
                        'training_villus_ids':sorted(frame.training_villus_id.unique().tolist())}
        frames.append(frame)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    table=out/'reference_labels.parquet';pd.concat(frames,ignore_index=True).to_parquet(table,index=False)
    manifest={'schema':1,'status':'FROZEN_REFERENCE_LABELS_IN_LOCKED_ZOOM_WINDOWS',
              'created_at':datetime.now(timezone.utc).isoformat(),'windows':windows,
              'inputs':[file_record(label_path,'frozen_sparse_reference_labels'),file_record(window_path,'unchanged_prediction_zoom_windows')],
              'outputs':{'labels':file_record(table,'frozen_zoom_reference_labels')},
              'background':'The same frozen spatial cells as the prediction zooms, shown in gray.',
              'reference_values':'Only original sparse annotations are colored; missing labels are not filled or predicted.',
              'new_window_selection':False,'labels_modified':False,'new_inference':False,
              'script':file_record(__file__,'explicit_reference_subsetting'),
              'command':' '.join([sys.executable,*sys.argv]),'python':sys.version,'pandas':pd.__version__}
    target=out/'manifest.json';target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest':file_record(target,'frozen_reference_zoom_manifest'),'windows':windows},indent=2))


if __name__=='__main__':main()
