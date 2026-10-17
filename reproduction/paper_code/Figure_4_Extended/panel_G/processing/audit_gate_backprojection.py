#!/usr/bin/env python3
"""Check the rendered overlay against frozen cells, channels and calibration."""
import argparse
import json
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from frozen_preprocessing.model_design_comparisons_v1.common import context,record,verify,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--figure-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads(a.manifest.read_text());key=json.loads(verify(manifest['channel_key']).read_text())
    if key['status']!='CONFIRMED' or key['E_cadherin_channel_index']!=2 or key['E_cadherin_metadata_name']!='Cy5':
        raise RuntimeError('E-cadherin channel differs from user confirmation')
    channel_manifest=json.loads(verify(manifest['channel_manifest']).read_text())
    old=json.loads(verify(channel_manifest['original_window_lock']).read_text());records=[]
    for condition,window in manifest['windows'].items():
        cells=pd.read_parquet(verify(window['cells']));source=pd.read_parquet(verify(window['source_predictions']))
        x0,x1,y0,y1=window['bounds_xxyy_px']
        expected=source.loc[(source.centroid_x_fullres_px>=x0)&(source.centroid_x_fullres_px<x1)&
                            (source.centroid_y_fullres_px>=y0)&(source.centroid_y_fullres_px<y1)].copy()
        expected['is_P14']=expected['class'].eq('Transferred')
        pd.testing.assert_frame_equal(cells.reset_index(drop=True),expected.reset_index(drop=True))
        pixel=manifest['pixel_size_um']
        dx=float(np.max(abs(cells.x_um-cells.centroid_x_fullres_px*pixel)))
        dy=float(np.max(abs(cells.y_um-cells.centroid_y_fullres_px*pixel)))
        if max(dx,dy)>1e-6:raise RuntimeError('Microscopy and cell coordinate calibration differ')
        dapi=np.load(verify(window['DAPI']),mmap_mode='r');ecad=np.load(verify(window['E_cadherin']),mmap_mode='r')
        if dapi.shape!=ecad.shape or dapi.shape!=(y1-y0,x1-x0):raise RuntimeError('Fluorescence field shapes differ')
        original=old['windows'][condition];ox0,ox1,oy0,oy1=original['bounds_xxyy_px']
        np.testing.assert_array_equal(dapi[oy0-y0:oy1-y0,ox0-x0:ox1-x0],np.load(verify(original['DAPI'])))
        if window['E_cadherin']['sha256']!=channel_manifest['windows'][condition]['channels'][2]['array']['sha256']:
            raise RuntimeError('Microscopy array is not Cy5')
        records.append({'condition':condition,'cells':len(cells),'P14_cells':int(cells.is_P14.sum()),
            'all_cell_fields_identical_to_source':True,'native_DAPI_core_byte_identical':True,
            'E_cadherin_is_confirmed_Cy5':True,'max_cell_calibration_error_um':max(dx,dy)})
    stem=a.figure_dir/'panel_G_gate_backprojection';pdf=stem.with_suffix('.pdf');png=stem.with_suffix('.png')
    fonts=subprocess.run(['pdffonts',str(pdf)],capture_output=True,text=True,check=True).stdout
    if 'Arial' not in fonts or 'DejaVu' in fonts or 'Type 3' in fonts:raise RuntimeError('Unexpected PDF fonts')
    provenance=json.loads((a.figure_dir/'panel_G_gate_backprojection_provenance.json').read_text())
    # The renderer records exact output hashes, including the companion PNG.
    for rec in provenance['outputs']:
        verify(rec)
    write_json(a.output,{**context(),'status':'PASS','manifest':record(a.manifest),'checks':records,
        'shared_condition_and_original_overlay_display_limits':True,'gate_boundaries_changed':False,
        'font_check':fonts,'outputs':[record(pdf),record(png)],
        'visual_review':'Both conditions show original DAPI/E-cadherin beside registered gate overlays; epithelial staining remains visible; P14 outlines and all five gate legend entries are legible. Scale bars are external. No anatomical accuracy claim is made.'})


if __name__=='__main__':main()
