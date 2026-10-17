#!/usr/bin/env python3
"""Freeze geometry-selected zoom subsets of the existing graph display table.

Selection uses annotation identities and spatial bounds only, never prediction
values or errors. Run explicitly before rendering; it performs no inference.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Figure_2_Extended.common import require_file, file_record

SECTIONS_SHA='745647147c9de41821445cea5eeb08f13422594191f55589fc24bb7a1db3fde5'
LABELS_SHA='7fd1c37966880be09e6fd402e810b7199358b4f9e20bad31da54abbcca522843'
BATCHES=('day6_SI_r2','day8_SI_Ctrl','day30_SI_r2')
FIELDS=['global_index','batch','x','y','crypt_villus_prediction','epithelial_distance_prediction']


def select_bounds(labels, batch, width=800.):
    """A common square centered on the first registered annotated villus."""
    identity=batch+'__villus_000'
    geometry=labels.loc[labels.batch.eq(batch)&labels.training_villus_id.eq(identity),['x','y']]
    if geometry.empty:raise RuntimeError('Prespecified annotation missing: '+identity)
    low,high=geometry.min(),geometry.max()
    center=(low+high)/2
    if (high-low).max()>=width:raise RuntimeError('Annotation exceeds the prespecified zoom width')
    return identity,[float(center.x-width/2),float(center.x+width/2),
                     float(center.y-width/2),float(center.y+width/2)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--section-cells',type=Path,required=True)
    p.add_argument('--sparse-labels',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    sections=require_file(a.section_cells,SECTIONS_SHA);labels=require_file(a.sparse_labels,LABELS_SHA)
    # Only geometry and annotation identity are read while selecting windows.
    annotations=pd.read_parquet(labels,columns=['batch','training_villus_id','x','y'])
    windows={}
    for batch in BATCHES:
        identity,bounds=select_bounds(annotations,batch)
        windows[batch]={'annotation_id':identity,'bounds_xxyy_um':bounds,'width_um':800.,'scale_bar_um':200.}
    table=pd.read_parquet(sections,columns=FIELDS)
    subsets=[]
    for batch,window in windows.items():
        x0,x1,y0,y1=window['bounds_xxyy_um']
        cells=table.loc[table.batch.eq(batch)&table.x.between(x0,x1)&table.y.between(y0,y1)].copy()
        if len(cells)<100:raise RuntimeError('Insufficient spatial coverage in the prespecified window')
        window['n']=len(cells)
        window['cell_identity_sha256']=hashlib.sha256(cells.global_index.to_numpy(dtype='<i8').tobytes()).hexdigest()
        subsets.append(cells)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    zoom=pd.concat(subsets,ignore_index=True);path=out/'zoom_cells.parquet';zoom.to_parquet(path,index=False)
    manifest={'schema':1,'status':'FROZEN_GEOMETRY_SELECTED_PREDICTION_ZOOMS',
              'created_at':datetime.now(timezone.utc).isoformat(),
              'selection':'First registered annotated villus (villus_000), centered on its bounding box; common 800 µm square.',
              'selection_uses_prediction_values':False,'same_windows_for_both_coordinates':True,
              'coordinate_units':'µm, from the frozen X_spatial table',
              'scope':'Descriptive all-label graph visualization, using the same presampled cells as the whole-tissue maps; not held-out performance.',
              'new_downsampling':False,'new_inference':False,'predictions_modified':False,
              'windows':windows,'outputs':{'cells':file_record(path,'frozen_zoom_cells')},
              'inputs':[file_record(sections,'frozen_dense_predictions'),file_record(labels,'manual_coordinate_labels')],
              'script':file_record(__file__,'explicit_spatial_subsetting'),
              'command':' '.join([sys.executable,*sys.argv]),'python':sys.version,'pandas':pd.__version__}
    target=out/'manifest.json';target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest':file_record(target,'frozen_zoom_manifest'),'windows':windows},indent=2))


if __name__=='__main__':main()
