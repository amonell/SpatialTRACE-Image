#!/usr/bin/env python3
"""Verify the seven-slot Figure 3A substitution and canonical publication."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from figure_organization import load_catalog
from figure_assembly.compositor import file_sha256
from Figure_3.supervised_example import supervised_example


def pixels(path):
    with Image.open(path) as image:return np.asarray(image.convert('RGBA'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    catalog=load_catalog()  # Requires all eight original artwork hashes.
    example=supervised_example()
    state=json.loads((args.run/'complete_figure_run.json').read_text())
    assert 'PASSED' in state['status'] and len(state['figures'])==1
    assert state['panel_publication']['published_panels']==5
    qa=json.loads((args.run/'QA/Figure_3/qa.json').read_text())
    assert not qa['errors'],qa['errors']
    provenance=json.loads((args.run/'Figure3_provenance.json').read_text())
    assert provenance['panel_recomposition']['status']=='PIXEL_IDENTICAL'
    paths=set()
    def verify(record):
        key=(record['path'],record['sha256'])
        if key not in paths:
            assert file_sha256(key[0])==key[1],key[0]
            paths.add(key)
    for record in provenance['outputs']+provenance['code']+provenance['preparation']['inputs']:
        verify(record)
    for record in example['manifest']['inputs']:
        verify(record)
    # Byte-level original shard inputs and unchanged source prediction values
    # are covered independently by test_supervised_example.py.
    layout=json.loads((ROOT/'Figure_3/assembly/layout.json').read_text())
    old=pixels(args.baseline/'Figure3.png');new=pixels(args.run/'Figure3.png')
    assert old.shape==new.shape
    changed=np.any(old!=new,axis=2)
    allowed=np.zeros(changed.shape,dtype=bool)
    sx=new.shape[1]/layout['page_size_pt'][0];sy=new.shape[0]/layout['page_size_pt'][1]
    counts={}
    for name in ('sft_local','sft_context','sft_fine','head_cv','head_epi','context_cv','context_epi'):
        x,y,w,h=layout['slots'][name]
        left=max(0,int(np.floor(x*sx))-2);right=min(new.shape[1],int(np.ceil((x+w)*sx))+2)
        top=max(0,int(np.floor(y*sy))-2);bottom=min(new.shape[0],int(np.ceil((y+h)*sy))+2)
        allowed[top:bottom,left:right]=True
        counts[name]=int(changed[top:bottom,left:right].sum())
        assert counts[name]>0,name
    outside=int(np.count_nonzero(changed & ~allowed))
    assert outside==0,f'Unexpected changes outside the seven approved slots: {outside}'
    # Audit every canonical final panel, including the seven untouched figures.
    panels=[]
    for entry in catalog['figures']:
        package=entry['package']
        manifest=json.loads((Path(catalog['output_root'])/package/'final_panel_manifest.json').read_text())
        assert [p['panel'] for p in manifest['panels']]==entry['panels']
        for panel in manifest['panels']:
            for kind in ('pdf','png'):
                verify({'path':panel[kind],'sha256':panel[kind+'_sha256']})
            if package=='Figure_3':
                assert Path(manifest['complete_page']).resolve()==(args.run/'Figure3.pdf').resolve()
                assert file_sha256(args.run/'panels'/package/Path(panel['png']).name)==panel['png_sha256']
            baseline=args.baseline/'panels'/package/Path(panel['png']).name
            different=not np.array_equal(pixels(panel['png']),pixels(baseline))
            assert different==(package=='Figure_3' and panel['panel']=='A'),(package,panel['panel'])
            panels.append({'package':package,'panel':panel['panel'],'changed':different})
    result={'status':'PASSED','created_at':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,*sys.argv],'original_artwork':'ALL_EIGHT_HASHES_UNCHANGED',
            'changed_slots':counts,'pixels_changed_outside_approved_slots':outside,
            'changed_panels':1,'unchanged_panels':len(panels)-1,'panels':panels,
            'geometry_anchors':len(qa['geometry']['anchors']),
            'unique_path_hash_pairs_verified':len(paths),
            'canonical_Figure_3_publication':'VERIFIED','regional_cells':len(example['table']),
            'scope':'Display, geometry, source and output integrity; not an independent accuracy estimate.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='panels'},indent=2))


if __name__=='__main__':main()
