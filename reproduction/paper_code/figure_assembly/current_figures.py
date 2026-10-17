"""Keep the user-facing figure folder limited to the eight current PDFs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from paper_paths import Path, lock_record, rendering_script
import shutil
import tempfile

from figure_assembly.compositor import file_sha256


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')


def refresh_current_figures(data_root, entries):
    """Publish hash-verified pages; keep QA/provenance and backups elsewhere.

    Sources are resolved through the canonical final-panel manifests, never
    timestamps or filename guesses. Publication preserves all old PDF bytes.
    """
    root=Path(data_root).resolve()
    parent=root/'Regenerated_Figures'
    destination=parent/'Current'
    archive=root/'Figure_Run_Archive'
    names=[Path(entry['reference']).name for entry in entries]
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate complete-page names')
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError('Current must be an ordinary directory')
    if destination.exists() and {p.name for p in destination.iterdir()}!=set(names):
        raise ValueError('Current contains unrecognized files; archive them explicitly before publishing')
    records=[]
    for entry,name in zip(entries,names,strict=True):
        index=root/entry['package']/'final_panel_manifest.json'
        manifest=json.loads(index.read_text())
        if manifest.get('owner')!='spatial_axes_manuscript.final_panel_layers':
            raise ValueError(f'Unowned canonical manifest: {index}')
        source=Path(manifest['complete_page'])
        if source.name!=name or file_sha256(source)!=manifest['complete_page_sha256']:
            raise ValueError(f'Current page source/hash mismatch: {source}')
        if destination.exists():
            current=destination/name
            if not current.is_file() or current.is_symlink():
                raise ValueError(f'Invalid current PDF: {current}')
        records.append({'package':entry['package'],'name':name,'source':str(source),
                        'sha256':manifest['complete_page_sha256'],'index':index,'manifest':manifest})
    archive.mkdir(parents=True,exist_ok=True)
    transaction=archive/'current_publications'/timestamp()
    transaction.mkdir(parents=True,exist_ok=False)
    stage=Path(tempfile.mkdtemp(prefix='current-stage-',dir=transaction))
    for record in records:
        target=stage/record['name'];shutil.copy2(record['source'],target)
        if file_sha256(target)!=record['sha256']:
            raise RuntimeError('Staged PDF hash mismatch')
        shutil.copy2(record['index'],transaction/(record['package']+'_manifest_before.json'))
    parent.mkdir(parents=True,exist_ok=True)
    previous=transaction/'previous_current'
    if destination.exists():os.replace(destination,previous)
    try:
        os.replace(stage,destination)
        for record in records:
            manifest=record['manifest']
            if Path(record['source'])!=destination/record['name']:
                manifest['complete_page_source']=record['source']
            manifest['complete_page']=str(destination/record['name'])
            manifest['current_export_published_at_utc']=transaction.name
            record['index'].write_text(json.dumps(manifest,indent=2)+'\n')
    except Exception:
        if destination.exists():os.replace(destination,stage)
        if previous.exists():os.replace(previous,destination)
        for record in records:
            shutil.copy2(transaction/(record['package']+'_manifest_before.json'),record['index'])
        raise
    result={'status':'PUBLISHED','directory':str(destination),
            'figures':[{'package':r['package'],'path':str(destination/r['name']),
                        'source':r['manifest'].get('complete_page_source',r['source']),
                        'sha256':r['sha256']} for r in records],
            'archive':str(transaction),'contents':'Current contains PDFs only; supporting records live outside Regenerated_Figures.'}
    (transaction/'publication.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
