"""Regenerate Figure 1 scientific layers at the locked September 5 page geometry."""
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
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection

SATA=Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")
GRAPH=SATA/"Figure_2/stored_outputs/shared_graph_components"
STORED=SATA/"Figure_1/stored_outputs/figure1_inputs"


def render(output_dir):
    output_dir=Path(output_dir);layout=json.loads((Path(__file__).parent/"layout.json").read_text())
    lock_path=Path(__file__).parent/"input_lock.json"
    input_lock=json.loads(lock_path.read_text())["inputs"]
    inputs=[U.record(lock_path,"reviewed_frozen_input_lock")]
    def read(path,role):
        inputs.append(U.record(path,role,lock_record(input_lock, path)["sha256"]));return path
    region_lock_path=Path(__file__).parent/"source_region_lock.json"
    regions=json.loads(region_lock_path.read_text());inputs.append(U.record(region_lock_path,"registered_artist_region_lock"))
    for rec in regions["source_assets"]:inputs.append(U.record(rec["path"],"artist_region_registration_evidence",rec["sha256"]))
    section=pd.read_parquet(read(GRAPH/"section_cells.parquet","graph_section_cells"))
    zoom=pd.read_parquet(read(GRAPH/"day90_zoom_cells.parquet","graph_zoom_cells"))
    labels=pd.read_parquet(read(GRAPH/"sparse_labels.parquet","sparse_coordinates"))
    sys.path.insert(0,str(ROOT/"Figure_1"))
    from common import verify_peyer_component,load_source_regions,verify_training_input_parity
    regions=load_source_regions(region_lock_path)
    peyer_path=SATA/"Figure_1/stored_outputs/peyer_repair_v1/peyer_region_cells.parquet"
    peyer_manifest=SATA/"Figure_1/stored_outputs/peyer_repair_v1/manifest.json"
    verify_peyer_component(peyer_path,peyer_manifest)
    peyer=pd.read_parquet(read(peyer_path,"corrected_Peyer_labels_and_predictions"));read(peyer_manifest,"corrected_Peyer_window_manifest")
    registry=json.loads(read(ROOT/"frozen_preprocessing/frozen_model_registry.json","model_registry").read_text())
    infer_dir=STORED/"panel_D_training_input_parity_v1"
    infer_prov=json.loads(read(infer_dir/"provenance.json","bounded_image_prediction_provenance").read_text())
    contract_path=read(ROOT/'Figure_1/panel_D/processing/figure1d_inference_contract.json','training_input_contract')
    read(verify_training_input_parity(infer_prov,contract_path),'golden_training_input_parity')
    cp=registry["models"]["xenium_image_coordinate_model"]["checkpoint"]
    if infer_prov["checkpoint_sha256"]!=cp["sha256"]:raise RuntimeError("Figure 1d prediction model mismatch")
    rec=next(x for x in infer_prov["outputs"] if Path(x["path"]).name=="predictions.csv")
    inputs.append(U.record(infer_dir/"predictions.csv","bounded_image_predictions",rec["sha256"]))
    predictions=pd.read_csv(infer_dir/"predictions.csv")
    training=pd.read_parquet(read(STORED/"day90_training_graph_cells.parquet","paired_training_graph_reference"))
    lock=json.loads(read(ROOT/"Figure_1/panel_D/processing/figure1d_region_lock.json","fixed_training_and_inference_regions").read_text())
    from Figure_4.release import load_release,release_file
    release_path=SATA/"Figure_4/stored_outputs/scientific_repair_v1/evaluation/release_manifest.json"
    release=load_release(release_path);read(release_path,"repaired_IF_release")
    if_path=release_file(SATA/"Figure_4/stored_outputs/scientific_repair_v1/evaluation/figure1e/rari_zoom_predictions.parquet",release)
    if_cells=pd.read_parquet(read(if_path,"current_IF_predictions"))
    data={name:np.load(read(STORED/path,role)) for name,path,role in (
        ("peyer_dapi","day90_peyer_dapi.npy","Peyer_DAPI"),("training_dapi","day90_training_dapi.npy","training_DAPI"),
        ("unseen_dapi","day90_inference_dapi.npy","unseen_DAPI"),("if_image","rari_if_multichannel_rgb.npy","IF_RGB"))}
    fig=U.figure();anchors={};slots=layout["image_slots"]
    def rect(name):return slots[name]["rect_pt"]
    def points(name,frame,x="x",y="y",**kw):
        ax,art=U.scatter(fig,rect(name),frame,x,y,**kw);anchors[name]=ax;return ax,art
    points("whole_tissue",section[section.batch=="day90_SI_r2"],color="#b8b8b8",size=.12)
    points("villus_zoom",zoom,color="#b8b8b8",size=.65,limits=regions["regions"]["villus_zoom"]["limits_um_xyxy"])
    ax,_=points("sparse_axis",zoom,color="#c8c8c8",size=.55,limits=regions["regions"]["sparse_axis"]["limits_um_xyxy"])
    visible_labels=regions["regions"]["sparse_axis"]["visible_villus_ids"]
    if visible_labels!=["day90_SI_r2__villus_004"]:raise RuntimeError("Original single labeled villus changed")
    selected=labels[(labels.batch.astype(str)=="day90_SI_r2")&labels.training_villus_id.astype(str).isin(visible_labels)]
    ax.scatter(selected.x,selected.y,c=selected.crypt_villus_label,cmap="viridis",vmin=0,vmax=1,s=1.1,linewidths=0,rasterized=True)
    v=labels[(labels.batch.astype(str)=="day90_SI_r2")&labels.training_villus_id.astype(str).eq("day90_SI_r2__villus_004")].reset_index(drop=True)
    tree=cKDTree(v[["x","y"]]);_,nn=tree.query(v[["x","y"]],k=min(4,len(v)))
    segments=[[v.loc[a,["x","y"]].to_numpy(float),v.loc[int(b),["x","y"]].to_numpy(float)] for a,row in enumerate(nn[:,1:]) for b in row if a<b]
    ax,_=points("sparse_graph",v,value="crypt_villus_label",size=3.0)
    ax.add_collection(LineCollection(segments,colors="#414141",linewidths=.13,alpha=.4,zorder=0))
    train_limits=lock["regions"]["paired_training_example"]["limits_um_xyxy"]
    points("dense_axis",training,value="crypt_villus_prediction",size=.65,limits=train_limits)
    points("training_axis",training,value="crypt_villus_prediction",size=.65,limits=train_limits)
    points("image_prediction",predictions,x="centroid_x_fullres_px",y="centroid_y_fullres_px",value="predicted_axis_coordinate",size=.5,limits=lock["regions"]["unseen_image_example"]["pixel_window_xyxy"])
    points("if_prediction",if_cells,x="centroid_x_fullres_px",y="centroid_y_fullres_px",value="predicted_axis_coordinate",size=.8,limits=(17000,23000,48000,52500))
    points("peyer_probability",peyer,x="x_spatial",y="y_spatial",value="gat_peyer_probability",cmap="magma",size=.65)
    for name in ("training_dapi","unseen_dapi","if_image","peyer_dapi"):
        anchors[name]=U.image(fig,rect(name),data[name],gray=name!="if_image",normalize=name!="if_image")
    ax,_=points("peyer_labels",peyer,x="x_spatial",y="y_spatial",color="#c8c8c8",size=.6)
    positive=peyer.peyer_manual_label.fillna(0).astype(float)>0
    ax.scatter(peyer.loc[positive,"x_spatial"],peyer.loc[positive,"y_spatial"],color="#7E57C2",s=.8,linewidths=0,rasterized=True)
    numeric=layout["numeric_slots"]
    for name,cmap in (("axis_colorbar","viridis"),("dense_colorbar","viridis"),("peyer_colorbar","magma")):
        anchors[name]=U.colorbar_strip(fig,numeric[name],cmap)
    groups={'A':('whole_tissue','villus_zoom','peyer_dapi'),'B':('sparse_axis','peyer_labels','axis_colorbar'),'C':('sparse_graph','dense_axis','peyer_probability','dense_colorbar','peyer_colorbar'),'D':('training_axis','training_dapi','unseen_dapi','image_prediction'),'E':('if_image','if_prediction')}
    panel_axes={name:panel for panel,names in groups.items() for name in names}
    layers=U.save_panel_layers(fig,output_dir,anchors,panel_axes)
    actual=U.save(fig,output_dir/"scientific_layers.pdf",anchors)
    result={"inputs":inputs,"rendered_anchors":actual,"scientific_panel_layers":layers,"scientific_axis_panels":panel_axes,"rows":{"Peyer":len(peyer),"IF":len(if_cells),"bounded_Xenium":len(predictions)},"source_region_registration":regions,"coordinate_colorbars":{"rendering":"256 opaque vector bands","normalization":[0,1]},"output":str(output_dir/"scientific_layers.pdf")}
    (output_dir/"scientific_layers_provenance.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--overwrite",action="store_true");args=parser.parse_args()
    for name in ("scientific_layers.pdf","scientific_layers_provenance.json",*[f'scientific_panel_{p}.pdf' for p in 'ABCDE']):
        if (args.output_dir/name).exists() and not args.overwrite:raise FileExistsError(args.output_dir/name)
    render(args.output_dir)
