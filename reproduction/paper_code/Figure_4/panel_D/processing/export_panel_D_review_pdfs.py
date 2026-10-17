#!/usr/bin/env python3
"""Copy a validated review's PDFs to a new user-facing folder."""
import argparse
from datetime import datetime, timezone
import json
import os
from paper_paths import Path, lock_record, rendering_script
import platform
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Figure_4.panel_D.processing.prepare_panel_D_field_candidates import record, verify, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-dir',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    source=a.review_dir.resolve();destination=a.destination.absolute()
    report=source/'home_export_provenance.json'
    if destination.exists() or destination.is_symlink() or report.exists():
        raise FileExistsError('Review export already exists; use a new destination/review')
    audit_path=source/'data_identity_audit.json';audit=json.loads(audit_path.read_text())
    if audit['status']!='PASS':raise RuntimeError('Review audit has not passed')
    provenance=json.loads(verify(audit['provenance']).read_text())
    pdfs=[source/provenance['combined_pdf']]
    pdfs += [source/'individual_PDFs'/f"Option_{c['option']:02d}.pdf" for c in provenance['candidates']]
    outputs={r['path']:r for r in provenance['outputs']}
    for path in pdfs:verify(outputs[str(path)])
    staging=Path(tempfile.mkdtemp(prefix='rari-review-copy-',dir=destination.parent))
    for path in pdfs:
        target=staging/path.name;shutil.copy2(path,target)
        if record(target)['sha256']!=outputs[str(path)]['sha256']:
            raise RuntimeError(f'Copy hash mismatch: {target}')
    if destination.exists():raise FileExistsError(destination)
    os.rename(staging,destination)
    save_json(report,{'status':'VERIFIED_PDF_COPIES','created_utc':datetime.now(timezone.utc).isoformat(),
        'command':[sys.executable,*sys.argv],'python':platform.python_version(),'script':record(__file__),
        'audit':record(audit_path),'source_provenance':audit['provenance'],
        'destination':str(destination),'pdf_count':len(pdfs),
        'copies':[{'source':str(path),**record(destination/path.name)} for path in pdfs],
        'source_files_preserved':True,'published_figures_changed':False})
    print(f'{len(pdfs)} verified PDF copies: {destination}')


if __name__=='__main__':main()
