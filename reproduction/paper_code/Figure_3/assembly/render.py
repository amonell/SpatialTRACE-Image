"""Render every scientific Figure 3 layer from hash-locked numeric inputs."""
from __future__ import annotations
import argparse
import json
from paper_paths import Path, lock_record, rendering_script
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Figure_1.assembly import render_util as U
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from Figure_3.figure_inputs import paired_views, locked_zoom, DEFAULT_ZOOM_RECORD, DEFAULT_PAIRED, PAIRED_GEOMETRY
from Figure_3.supervised_example import supervised_example, prediction_background, add_prediction_overlay, DEFAULT_SUPERVISED
from Figure_3.common import DEFAULT_DATA_RELEASE, data_release

SATA=Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript')
STORED=SATA/'Figure_3/stored_outputs'
CHECKPOINT_SHA='d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137'


def render(output_dir, paired_manifest=DEFAULT_PAIRED/'manifest.json', paired_view_path=DEFAULT_PAIRED/'paired_views.npz', supervised_manifest=DEFAULT_SUPERVISED):
    output_dir=Path(output_dir);layout=json.loads((Path(__file__).parent/'layout.json').read_text());slots=layout['slots']
    lock_path=Path(__file__).parent/'input_lock.json';input_lock=json.loads(lock_path.read_text())['inputs'];inputs=[U.record(lock_path,'reviewed_frozen_input_lock')]
    def read(path,role,expected=None):
        locked=lock_record(input_lock, path)['sha256']
        if expected is not None and expected!=locked:raise RuntimeError(f'Upstream manifest disagrees with reviewed input lock: {path}')
        inputs.append(U.record(path,role,locked));return Path(path)
    refmanifest=data_release(read(DEFAULT_DATA_RELEASE,'training_matched_coordinate_release'))
    def reference(name):
        rec=refmanifest['outputs'][name];return read(rec['path'],'current_graph_reference_'+name,rec['sha256'])
    frame=pd.read_parquet(reference('predictions_current_references.parquet'))
    metadata=pd.read_csv(reference('embedding_metadata_current_references.csv'))
    metrics=json.loads(reference('metrics.json').read_text())
    umaprec=refmanifest['outputs']['figure3f_umap.csv']
    umap=pd.read_csv(read(umaprec['path'],'frozen_UMAP_geometry',umaprec['sha256']))
    split=refmanifest['split'];read(split['path'],'locked_cell_split',split['sha256'])
    read(ROOT/'Figure3/train_model/outputs/production_representation_model_v2/best_crypt_villus_vit_model.pt','released_coordinate_checkpoint',CHECKPOINT_SHA)
    if len(frame)!=50000 or frame.row_id.nunique()!=50000 or len(umap)!=30000 or len(metadata)!=30000 or metadata.row_id.nunique()!=30000:raise RuntimeError('Locked Figure 3 cell counts changed')
    if set(metadata.source_id.astype(str))!={'sample_008'} or set(frame.source_id.astype(str))!={'sample_008'}:raise RuntimeError('Locked Figure 3 source changed')
    example=supervised_example(read(supervised_manifest,'approved_Xenium_supervised_example'))
    bp=example['manifest'];bounded=example['table'];supervised_inputs=example['images']
    for rec in bp['outputs']:read(rec['path'],'supervised_Xenium_'+Path(rec['path']).name,rec['sha256'])
    ap=STORED/'panel_A/panel_A_exact_historical_region_provenance.json'
    source_prov=json.loads(read(ap,'original_prepared_crop_provenance').read_text())
    crop_path=STORED/'panel_A/panel_A_exact_historical_region.npz'
    crop_rec=next(r for r in source_prov['outputs'] if Path(r['path']).name==crop_path.name)
    crops=np.load(read(crop_path,'exact_original_DAPI_crops',crop_rec['sha256']))
    cropmeta=json.loads(read(STORED/'panel_A/panel_A_exact_historical_region.json','exact_original_crop_metadata').read_text())
    if cropmeta['center_id']!='if:cont1:nucleus:352082':raise RuntimeError('Original pretraining field mismatch')
    inputs.append(U.record(DEFAULT_ZOOM_RECORD,'locked_artist_selected_zoom'))
    zoom=locked_zoom()
    fig=U.figure();anchors={}
    def image(name,array,gray=True):
        anchors[name]=U.image(fig,slots[name],array,gray=gray);return anchors[name]
    for name in ('overview','context','local','fine'):
        image(name,crops[name])
    for name,source in (('sft_local','local_image'),('sft_context','context_image'),('sft_fine','fine_image')):image(name,supervised_inputs[source])
    for name,k,flip in (('augmentation_original',0,False),('augmentation_rotation',1,False),('augmentation_flip',1,True)):
        arr=np.rot90(crops['context'],k);arr=np.fliplr(arr) if flip else arr;image(name,arr)
    # Actual dataset inputs: clean original teachers and identically augmented
    # local/context students; black student patches illustrate token masking.
    views, paired_record = paired_views(read(paired_manifest,'actual_dataset_paired_view_manifest'),read(paired_view_path,'actual_dataset_paired_views'))
    for name,array in views.items():image(name,array)
    for key in ('head_cv','head_epi','context_cv','context_epi'):
        ax=image(key,prediction_background(example,key))
        add_prediction_overlay(ax,example,key)
    anchors['head_colorbar']=U.colorbar_strip(fig,slots['head_colorbar'],'viridis')
    embedding=pd.concat([umap,metadata],axis=1)
    for name,column,cmap in (('umap_cv','target_axis','viridis'),('umap_epi','epithelial_distance_clipped_1p0','coolwarm')):
        vmax=.7 if name=='umap_epi' else 1
        anchors[name],_=U.scatter(fig,slots[name],embedding,'umap_1','umap_2',value=column,cmap=cmap,size=.11,vmax=vmax,alpha=.78)
        anchors[name+'_colorbar']=U.colorbar_strip(fig,slots[name+'_colorbar'],cmap,vertical=False,vmax=vmax)
    for name,xcol,ycol,cmap in (('scatter_cv','target_axis','predicted_axis_coordinate','viridis'),('scatter_epi','epithelial_distance_clipped_1p0','predicted_epithelial_distance_clipped_1p0','coolwarm')):
        ax=U.axis(fig,slots[name]);artist=ax.hexbin(frame[xcol],frame[ycol],gridsize=55,extent=(0,1,0,1),mincnt=1,cmap=cmap,norm=LogNorm(),linewidths=0,rasterized=True)
        ax.set_xlim(0,1);ax.set_ylim(0,1);anchors[name]=ax
        cax=U.axis(fig,slots[name+'_colorbar'],off=False);cb=fig.colorbar(artist,cax=cax)
        if cb.solids is not None:
            cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.solids.set_linewidth(.5)
        cb.ax.tick_params(labelsize=6.2,pad=1,length=2);cb.ax.minorticks_off();cb.set_label('Cell count',fontsize=7,labelpad=2)
        anchors[name+'_colorbar']=cax
    xx=frame.centroid_x_fullres_px;yy=frame.centroid_y_fullres_px;pad=max(xx.max()-xx.min(),yy.max()-yy.min())*.02
    full_limits=(xx.min()-pad,xx.max()+pad,yy.min()-pad,yy.max()+pad)
    for axis,ref,pred,cmap in (('cv','target_axis','predicted_axis_coordinate','viridis'),('epi','epithelial_distance_clipped_1p0','predicted_epithelial_distance_clipped_1p0','coolwarm')):
        vmax=.7 if axis=='epi' else 1
        for variant,column,limits in (('reference',ref,full_limits),('prediction',pred,full_limits),('reference_zoom',ref,zoom),('prediction_zoom',pred,zoom)):
            name=axis+'_'+variant;iszoom=variant.endswith('zoom');display=frame
            if iszoom:
                x0,x1,y0,y1=zoom;display=frame[xx.between(x0,x1)&yy.between(y0,y1)]
            ax,_=U.scatter(fig,slots[name],display,'centroid_x_fullres_px','centroid_y_fullres_px',value=column,cmap=cmap,size=1.25 if iszoom else .33,limits=limits,vmax=vmax)
            if not iszoom:
                x0,x1,y0,y1=zoom;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#111111',linewidth=.6))
            anchors[name]=ax;anchors[name+'_colorbar']=U.colorbar_strip(fig,slots[name+'_colorbar'],cmap,vmax=vmax)
    panel_axes={name:('B' if name.startswith('umap_') else 'C' if name.startswith('scatter_') else 'D' if name.startswith('cv_') else 'E' if name.startswith('epi_') else 'A') for name in anchors}
    layers=U.save_panel_layers(fig,output_dir,anchors,panel_axes)
    actual=U.save(fig,output_dir/'scientific_layers.pdf',anchors)
    result={'inputs':inputs,'rendered_anchors':actual,'scientific_panel_layers':layers,'scientific_axis_panels':panel_axes,'metrics':metrics,'rows':{'main_maps':len(frame),'UMAP':len(umap),'bounded_context':len(bounded)},'mask_seed':[7,8],'mask_tokens':102,'mask_visualization':paired_record['mask_visualization'],'paired_geometry':PAIRED_GEOMETRY,'paired_source':{key:paired_record[key] for key in ('center_id','local_prepared_index','context_prepared_index','student_variant','teacher_variant','shared_student_intensity_parameters')},'zoom_limits_fullres_px':zoom,'coordinate_colorbars':{'rendering':'256 opaque vector bands','crypt_villus_limits':[0,1],'epithelial_UMAP_and_spatial_limits':[0,.7],'bounded_context_head_limits':[0,1]}}
    result['supervised_example']={key:bp[key] for key in ('stable_source_id','illustrated_row_id','illustrated_prepared_index','evaluation_scope','selection_rule','display')}
    (output_dir/'scientific_layers_provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--paired-manifest',type=Path,required=True);parser.add_argument('--paired-views',type=Path,required=True);parser.add_argument('--supervised-manifest',type=Path,required=True);parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    for name in ('scientific_layers.pdf','scientific_layers_provenance.json',*[f'scientific_panel_{p}.pdf' for p in 'ABCDE']):
        if (args.output_dir/name).exists() and not args.overwrite:raise FileExistsError(args.output_dir/name)
    render(args.output_dir,args.paired_manifest,args.paired_views,args.supervised_manifest)
