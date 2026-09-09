"""Assemble Figure 3 using fresh science and allowlisted original manual paths."""
from __future__ import annotations
import json
from paper_paths import Path, lock_record, rendering_script
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from figure_assembly.static_art import extract_manual_art,inspect_path_events,inspect_text,contains,intersects,sha256
from figure_assembly.compositor import register_fonts
from reportlab.pdfbase.pdfmetrics import stringWidth
from Figure_3.figure_inputs import DEFAULT_PAIRED
from Figure_3.supervised_example import DEFAULT_SUPERVISED

SUPERSEDED_SFT_CONTOURS = {
    3963: [323.285, 115.324, 2.06, 1.842],
    3966: [324.0551, 71.171, 0.52, 0.464],
    3995: [324.1307, 181.7817, 8.542, 7.437],
}


def schematic_label(original, x, y):
    """Replace training-report text at its registered original baselines.

    Counts, schedules, and checkpoint selection are documented in Methods.
    None of the scientific layers or original manual vector paths change.
    """
    report = {
        'Pretraining Data Summary': 'Paired representation learning',
        'Unique pairs of spatially diverse context': 'Paired local and context views',
        'crops and local crops:': 'from Xenium and IF DAPI images',
        'Total local crops': 'Predict clean teacher features',
        'Total context crops': 'within and across scales',
        'Total augmented crops': None,
        ': 22,048': None,
        ': 176,384': None,
    }
    if 160 < x < 290 and 95 < y < 153 and original in report:
        return report[original]
    supervision = {
        'Summary': 'Coordinate supervision',
        'Xenium-only SFT': 'Graph-derived coordinate targets',
        '13 supervised epochs': 'Full-model fine-tuning',
        '81,656 multiscale Xenium training crops': 'Two coordinates per cell',
    }
    if 305 < x < 500 and 333 < y < 370 and original in supervision:
        return supervision[original]
    return {
        '2048 px': '666 µm',
        '512 px': '166 µm',
        '128 px': '42 µm',
        'Both ViT pretraining and SFT:': 'Local and context fields:',
        'Only SFT:': 'Fine branch (fine-tuning only):',
        'Single Nucleus': 'Target nucleus',
        'Epithelial Axis': 'Epithelial distance',
        'Multitask heads': 'Coordinate heads',
    }.get(original, original)


def metric_label_expectations(metrics):
    """Require all displayed metrics from the same frozen record as the renderer."""
    return [f'{label} = {metrics[axis][key]:.3f}'
            for axis in ('crypt_villus', 'epithelial_distance')
            for label,key in (('Pearson r','pearson_r'),('MAE','mae'),('R²','r2'))]


def obsolete_sft_contours(events):
    """Remove only the original Cont1 nuclear outlines from replaced inputs.

    Their neighboring rectangular strokes are original colored input frames
    and remain unchanged. Paths are registered to the hash-locked source PDF.
    """
    lookup={event['index']:event for event in events}
    for index,bounds in SUPERSEDED_SFT_CONTOURS.items():
        event=lookup.get(index)
        if (event is None or event['op']!='S' or event['path_operations']!=334 or
                max(abs(a-b) for a,b in zip(event['bbox_pt'],bounds))>.001):
            raise RuntimeError('Original supervised Cont1 outline registration changed')
    return set(SUPERSEDED_SFT_CONTOURS)


def manual_layer_indices(selected, required, ordered_groups):
    """Keep overlapping card fills and strokes in their original paint order."""
    lookup={event['index']:event for event in selected}
    ordered=[]
    for group in ordered_groups:
        indices=group['paint_indices']
        if indices!=sorted(indices) or any(index not in lookup for index in indices):
            raise RuntimeError(f"Missing or reordered manual paths: {group['id']}")
        events=[lookup[index] for index in indices]
        if [event['op'] for event in events]!=group['expected_ops']:
            raise RuntimeError(f"Changed manual paint sequence: {group['id']}")
        x=min(event['bbox_pt'][0] for event in events)
        y=min(event['bbox_pt'][1] for event in events)
        bounds=[x,y,max(event['bbox_pt'][0]+event['bbox_pt'][2] for event in events)-x,
                max(event['bbox_pt'][1]+event['bbox_pt'][3] for event in events)-y]
        if max(abs(a-b) for a,b in zip(bounds,group['bounds_pt']))>.001:
            raise RuntimeError(f"Changed manual group geometry: {group['id']}")
        ordered.extend(indices)
    if len(ordered)!=len(set(ordered)) or set(ordered)&set(required):
        raise RuntimeError('Manual path groups overlap')
    ordered_set=set(ordered)
    backgrounds=[event['index'] for event in selected
                 if event['index'] not in required and event['index'] not in ordered_set
                 and event['op'] in ('f','f*','B','B*','b','b*')
                 and event['bbox_pt'][2]*event['bbox_pt'][3]>500]
    foreground=[event['index'] for event in selected
                if event['index'] not in backgrounds and event['index'] not in ordered_set]
    if not set(required).issubset(foreground):
        raise RuntimeError('Required arrows or metric backings were dropped from the foreground')
    if len(backgrounds)+len(foreground)+len(ordered)!=len(selected):
        raise RuntimeError('Manual paint partition changed path coverage')
    return backgrounds,foreground,sorted(ordered)


def prepare(output_dir:Path,overwrite=False):
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    layout=json.loads((Path(__file__).parent/'layout.json').read_text());reference=Path(layout['reference']);slots=layout['slots']
    if sha256(reference)!=layout['reference_sha256']:raise RuntimeError('Figure 3 layout reference changed')
    for name in ['scientific_layers.pdf',*[f'scientific_panel_{p}.pdf' for p in 'ABCDE']]:
        if (output_dir/name).exists() and not overwrite:raise FileExistsError(output_dir/name)
    events=inspect_path_events(reference);selected=[]
    art_lock_path=Path(__file__).parent/'manual_path_lock.json';art_lock=json.loads(art_lock_path.read_text())
    if art_lock['reference_sha256']!=layout['reference_sha256']:raise RuntimeError('Manual path lock belongs to different artwork')
    required={r['index']:r for r in art_lock['required_paths']}
    event_lookup={e['index']:e for e in events}
    obsolete=obsolete_sft_contours(events)
    if obsolete & set(required):raise RuntimeError('A retired scientific contour overlaps required manual artwork')
    for index,record in required.items():
        if index not in event_lookup or max(abs(a-b) for a,b in zip(event_lookup[index]['bbox_pt'],record['bbox_pt']))>.001:raise RuntimeError(f'Required manual geometry changed: {record}')
    replaced=['student_local','student_context','teacher_local','teacher_context','head_cv','head_epi','context_cv','context_epi']
    # Every scientific prediction polygon, old student mask, image, and text
    # object is removed; original schematic frames/arrows remain independent.
    for event in events:
        if event['index'] in obsolete:continue
        x,y,w,h=event['bbox_pt'];op=event['op'];filled=op in ('f','f*','B','B*','b','b*')
        if event['index'] in required:
            selected.append(event);continue
        if y<380:
            if any(contains([slots[n][0]-.3,slots[n][1]-.3,slots[n][2]+.6,slots[n][3]+.6],event['bbox_pt']) for n in replaced):continue
            if filled and any(contains([r[0]-.3,r[1]-.3,r[2]+.6,r[3]+.6],event['bbox_pt']) and w*h>.85*r[2]*r[3] for r in slots.values()):continue
            selected.append(event)
        elif x<234 and y<531:
            if not filled:selected.append(event)
        elif 250<x<595 and 400<y<532:
            in_countbar=(383<x<416) or x>562
            if op=='S' and not in_countbar and event['path_operations']<=8:selected.append(event)
        elif y>548:
            in_map=any(contains([r[0]-3,r[1]-3,r[2]+6,r[3]+6],event['bbox_pt']) for n,r in slots.items() if n.startswith(('cv_','epi_')) and 'colorbar' not in n)
            if not filled and not in_map:selected.append(event)
    backgrounds,foreground,ordered=manual_layer_indices(selected,required,art_lock['ordered_path_groups'])
    audits=[]
    for name,indices in (('manual_background',backgrounds),('manual_foreground',foreground),('manual_transformer_stacks',ordered)):
        audits.append(extract_manual_art(reference,output_dir/(name+'.pdf'),source_sha256=layout['reference_sha256'],keep_paint_indices=indices,overwrite=overwrite))
    extracted=inspect_path_events(output_dir/'manual_transformer_stacks.pdf')
    if len(extracted)!=len(ordered) or any(
            event['op']!=event_lookup[index]['op'] or
            max(abs(a-b) for a,b in zip(event['bbox_pt'],event_lookup[index]['bbox_pt']))>.001
            for event,index in zip(extracted,ordered)):
        raise RuntimeError('Transformer vector extraction changed original geometry or paint order')
    command=[sys.executable,'-B',str(Path(__file__).parent/'render.py'),'--output-dir',str(output_dir),
             '--paired-manifest',str(DEFAULT_PAIRED/'manifest.json'),'--paired-views',str(DEFAULT_PAIRED/'paired_views.npz'),
             '--supervised-manifest',str(DEFAULT_SUPERVISED)]
    if overwrite:command.append('--overwrite')
    subprocess.run(command,check=True)
    data=json.loads((output_dir/'scientific_layers_provenance.json').read_text());metrics=data['metrics']
    elements=[{'id':'manual-background','kind':'static','path':'manual_background.pdf','rect_pt':[0,0,*layout['page_size_pt']],'z':0},
              {'id':'fresh-scientific-layers','kind':'pdf','path':'scientific_layers.pdf','rect_pt':[0,0,*layout['page_size_pt']],'z':1},
              {'id':'manual-foreground','kind':'static','path':'manual_foreground.pdf','rect_pt':[0,0,*layout['page_size_pt']],'z':2},
              {'id':'ordered-transformer-stacks','panel':'A','kind':'static','path':'manual_transformer_stacks.pdf','rect_pt':[0,0,*layout['page_size_pt']],'z':2}]
    summary_dir=ROOT/'Figure3/train_model/outputs'
    pretrain_path=summary_dir/'paired_representation_pretraining/pretraining/pretrain_summary.json'
    train_path=summary_dir/'production_representation_model_v2/training_summary.json'
    selection_path=summary_dir/'production_representation_model_v2/production_selection.json'
    input_lock=json.loads((Path(__file__).parent/'input_lock.json').read_text())['inputs']
    for path in (pretrain_path,train_path,selection_path):
        if sha256(path)!=lock_record(input_lock, path)['sha256']:raise RuntimeError(f'Frozen architecture summary changed: {path}')
    pretrain=json.loads(pretrain_path.read_text());training=json.loads(train_path.read_text());selection=json.loads(selection_path.read_text())
    replacements={'Xenium-aligned training data preparation':'Multiscale DAPI input preparation',
                  'Transformer Encoder x 6':'Shared transformer × 6',
                  '16 x 16 patch tokens':'16 × 16 patch tokens',
                  '256 tokens x 256D':'Scale embeddings',
                  'Whole tissue inference':'Regional inference',
                  'GAT-derived reference label':'Graph-derived reference',
                  'GAT-derived axis label':'Graph-derived reference',
                  'GAT-derived axis zoom':'Reference zoom',
                  'GAT-derived axis label zoom':'Reference zoom',
                  'Held out section':'Example section'}
    register_fonts();seen=[];preserved_headers={}
    for index,run in enumerate(inspect_text(reference)):
        original=run['text'].strip();x,y=run['x_pt'],run['y_pt']
        if any(text==original and abs(x-xx)<.03 and abs(y-yy)<.03 for text,xx,yy in seen):continue
        seen.append((original,x,y))
        # Count ticks follow the newly rendered count range. Coordinate ticks
        # retain their original anchors and active 0–1 or 0–0.7 normalization.
        if 400<y<532 and ((390<x<420) or x>570):continue
        text=schematic_label(original,x,y)
        if text is None:continue
        text=replacements.get(text,text)
        axis='crypt_villus' if x<420 else 'epithelial_distance'
        if 406<y<432 and 269<x<565:
            if original.startswith('Pearson r'):text=f"Pearson r = {metrics[axis]['pearson_r']:.3f}"
            elif original.startswith('MAE ='):text=f"MAE = {metrics[axis]['mae']:.3f}"
            elif original.startswith('R² ='):text=f"R² = {metrics[axis]['r2']:.3f}"
            elif original=='093':continue
        font='Arial-Bold' if run['paint_count']>1 or 'Bold' in run['source_font'] else 'Arial'
        size=run['font_size_pt']
        if text==original and run['advance_pt']>0:size=min(size,size*run['advance_pt']/max(.01,stringWidth(text,font,size)))
        if original=='256 tokens x 256D':
            size=5.1
            if stringWidth(text,font,size)>run['advance_pt']:
                raise RuntimeError('Scale-embedding label exceeds its original text footprint')
        if original=='Transformer Encoder x 6':size=min(size,size*run['advance_pt']/stringWidth(text,font,size))
        if original=='Epithelial Axis':
            # Keep the original label center, using the available box width
            # instead of shrinking a longer label to the old glyph width.
            size=5.0
            x+=(run['advance_pt']-stringWidth(text,font,size))/2
        if original.startswith('Pearson r') or original.startswith('MAE =') or original.startswith('R² ='):size=7.4
        color=run['color']
        if original in art_lock['header_palette']:
            color=[v/255 for v in art_lock['header_palette'][original]['rgb_8bit']];preserved_headers[original]=color
        elements.append({'id':f'manual-label-{index:03d}','kind':'text','text':text,'x_pt':x,'y_pt':y,'font_size_pt':size,'font':font,'rotation_degrees':run['rotation_degrees'],'color':color,'z':3})
    if set(preserved_headers)!=set(art_lock['header_palette']):raise RuntimeError('A required original colored header was lost')
    inputs=data['inputs']+[{'path':str(reference),'sha256':layout['reference_sha256'],'role':'manual_vector_and_geometry_reference'}]
    for path in (Path(__file__),Path(__file__).parent/'layout.json',Path(__file__).parent/'render.py',ROOT/'Figure_3/figure_inputs.py',art_lock_path,ROOT/'Figure_1/assembly/render_util.py',ROOT/'figure_assembly/static_art.py',pretrain_path,train_path,selection_path):
        inputs.append({'path':str(path),'sha256':sha256(path),'role':'assembly_source_or_frozen_summary'})
    path=ROOT/'Figure_3/supervised_example.py'
    inputs.append({'path':str(path),'sha256':sha256(path),'role':'shared_supervised_example_renderer'})
    return {'elements':elements,'inputs':inputs,'commands':[command],'warnings':[],
            'expected_text':metric_label_expectations(metrics),
            'static_art_audits':audits,'anchors':layout['anchors'],'rendered_anchors':data['rendered_anchors'],'scientific_panel_layers':data['scientific_panel_layers'],'scientific_axis_panels':data['scientific_axis_panels'],
            'required_manual_paths_verified':list(required.values()),'preserved_header_colors':preserved_headers,
            'ordered_manual_path_groups_verified':art_lock['ordered_path_groups'],
            'removed_obsolete_Cont1_supervised_contours':sorted(obsolete),
            'intentional_differences':layout['intentional_differences']}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    result=prepare(args.output_dir,args.overwrite);(args.output_dir/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
