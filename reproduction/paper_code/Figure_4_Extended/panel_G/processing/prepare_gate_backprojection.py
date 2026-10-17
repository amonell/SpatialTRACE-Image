#!/usr/bin/env python3
"""Freeze existing IF gate calls and verified DAPI/E-cadherin microscopy."""
import argparse
import json
import sys
from paper_paths import Path, lock_record, rendering_script
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from Figure_4.release import load_release,release_file
from Figure_4.common import require_file,file_record
from frozen_preprocessing.model_design_comparisons_v1.common import context,record,verify,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-manifest',type=Path,required=True)
    p.add_argument('--channel-manifest',type=Path,required=True)
    p.add_argument('--channel-key',type=Path,required=True,help='Confirmed antibody identity and its evidence; no visual guessing')
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    release=load_release(a.release_manifest);channels=json.loads(require_file(a.channel_manifest).read_text())
    key=json.loads(require_file(a.channel_key).read_text())
    if key.get('status')!='CONFIRMED' or not key.get('evidence'):
        raise RuntimeError('An E-cadherin antibody/channel key must be confirmed before rendering')
    channel=int(key['E_cadherin_channel_index'])
    if channel not in (1,2,3):raise RuntimeError('Invalid epithelial channel')
    if key['source_image_path']!=channels['source']['path']:raise RuntimeError('Channel key belongs to a different source image')
    expected=key['E_cadherin_metadata_name'];windows={};out.mkdir(parents=True)
    for condition in ('DMSO','RARi'):
        window=channels['windows'][condition]
        if not window['DAPI_byte_exact_to_frozen_crop']:raise RuntimeError('Microscopy registration not verified')
        ecad=window['channels'][channel]
        if ecad['metadata_name']!=expected:raise RuntimeError('Confirmed channel name differs from microscopy metadata')
        prediction_rec=release['outputs'][f'whole_tissue/{condition.lower()}_predictions.parquet']
        rows=pd.read_parquet(release_file(prediction_rec['path'],release))
        x0,x1,y0,y1=window['bounds_xxyy_px']
        selected=rows.loc[(rows.centroid_x_fullres_px>=x0)&(rows.centroid_x_fullres_px<x1)&
                          (rows.centroid_y_fullres_px>=y0)&(rows.centroid_y_fullres_px<y1)].copy()
        if selected.empty:raise RuntimeError('Empty locked field')
        selected['is_P14']=selected['class'].eq('Transferred')
        # Use saved calls exactly; never reassign a cell or refit boundaries here.
        path=out/f'{condition}_cells.parquet';selected.to_parquet(path,index=False)
        windows[condition]={'bounds_xxyy_px':window['bounds_xxyy_px'],'DAPI':window['channels'][0]['array'],
            'E_cadherin':ecad['array'],'cells':record(path),'source_predictions':prediction_rec,
            'n':len(selected),'P14_n':int(selected.is_P14.sum()),
            'gate_counts':{str(k):int(v) for k,v in selected.coordinate_gate.value_counts().items()}}
        verify(windows[condition]['DAPI']);verify(windows[condition]['E_cadherin'])
    write_json(out/'manifest.json',{**context(),'status':'FROZEN_REGISTERED_GATE_BACKPROJECTION',
        'release':record(a.release_manifest),'channel_manifest':record(a.channel_manifest),'channel_key':record(a.channel_key),
        'gates':release['outputs']['imap/p14_gate_definitions.json'],'checkpoint':release['checkpoint'],
        'pixel_size_um':channels['pixel_size_um'],'windows':windows,
        'display':{'DAPI':channels['display']['0'],'E_cadherin':channels['display'][str(channel)]},
        'gate_assignments_changed':False,'boundaries_changed':False,'separate_condition_boundaries':False,
        'P14_rule':'Saved QuPath class Transferred','scope':'Qualitative anatomical correspondence, not compartment classification accuracy',
        'code':[record(Path(__file__))]})


if __name__=='__main__':main()
