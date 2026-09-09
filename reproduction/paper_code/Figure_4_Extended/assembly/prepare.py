"""Assemble freshly rendered Extended Figure4 layers at the reviewed anchors."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from Figure_4.assembly.common import (PageInputs, HOME, SATA, REPAIR, PAGE, PIXEL_UM,
                                      colorbar, imap, top_text)
from Figure_4_Extended.peyer import contrast, scale_bar
from Figure_4_Extended.probability_display import (DISPLAY_CONFIG, probability_display,
                                                 probability_range, label_probability_colorbar)
from Figure_4.frozen_annotations import training_example_counts, joined_training_history
from figure_assembly.static_art import extract_manual_art, inspect_path_events

LAYOUT=Path(__file__).with_name("layout.json")


def workflow(page,layout):
    path=page.out/"panel_A_original_vector_paths.pdf"
    audit=extract_manual_art(layout["reference"],path,source_sha256=layout["reference_sha256"],
         keep_paint_indices=layout["manual_path_indices"],overwrite=page.overwrite)
    page.elements.append({"id":"A_original_manual_paths","panel":"A","kind":"pdf","path":path.name,
                          "rect_pt":[0,0,*PAGE],"fit":"exact"})
    page.outputs.append({"path":str(path),"sha256":audit["output_sha256"]})
    page.details["manual_vector_extraction"]=audit
    regenerated_paths=inspect_path_events(path)
    if len(regenerated_paths)!=len(layout["manual_path_indices"]):
        raise RuntimeError("The workflow extraction retained unexpected path objects")
    for original_index,event in zip(layout["manual_path_indices"],regenerated_paths,strict=True):
        page.anchors.append({"id":f"A_original_path_{original_index}","rect_pt":event["bbox_pt"]})
    page.read(HOME/"figure_assembly/static_art.py",role="structural_static_extraction_helper")
    summary=json.loads(page.stage("stage1_heads/training_summary.json").read_text())
    counts=training_example_counts(pd.read_csv(page.stage("supervised_manifest.csv")),summary)
    page.details["workflow_training_counts"]=counts
    for text,x,y,size,bold,align in [
        ("IF-only fine-tuning",51.4824,36.0414,9.6,True,"left"),
        ("Multiscale ViT + CNN",73.3263,62.121,9.6,False,"center"),
        ("Xenium-fine-tuned checkpoint",73.3263,73.641,9.2,False,"center"),
        ("IF training data",195.7804,53.807,9.6,False,"center"),
        (f"{counts['cells']:,} labeled cells",195.7804,65.327,9.3,False,"center"),
        (f"{counts['villi']} villi",195.7804,76.847,9.6,False,"center"),
        ("Multiscale ViT + CNN",305.3701,59.0951,9.6,False,"center"),
        ("IF-fine-tuned checkpoint",305.3701,72.1223,9.3,False,"center")]:
        page.text(text,x,y,size,bold,align,panel="A")


def transfer(page):
    table=pd.read_csv(page.frozen("transfer_comparison/final_lineage_transfer_metrics.tsv"),sep="\t")
    if not table.n.eq(4843).all():raise RuntimeError("Unexpected IF transfer test membership")
    figure=page.figure();rect=[423.714,34.577,141.102,61.916]
    axis=page.axis(figure,"B_transfer_tiles",rect)
    values=[]
    keys=table.model_key.drop_duplicates().tolist()
    for predicate in (lambda k:"zero" in k,lambda k:"zero" not in k):
        selected=table.loc[table.model_key.map(predicate)]
        row=[]
        for coordinate in ("Crypt-villus","Epithelial distance"):
            match=selected.loc[selected.coordinate.eq(coordinate)]
            if len(match)!=1:raise RuntimeError(f"Ambiguous transfer table {keys} {coordinate}")
            row.append(float(match.spearman_rho.iloc[0]))
        values.append(row)
    for i,row in enumerate(values):
        for j,value in enumerate(row):
            rgba=plt.get_cmap("YlGnBu")(.25+.70*value)
            axis.add_patch(Rectangle((j,i),1,1,facecolor=rgba[:3],edgecolor="white",lw=.5))
            axis.text(j+.5,i+.5,f"{value:.2f}",ha="center",va="center",fontsize=9,fontweight="bold",color="white" if value>.6 else "#333333")
    axis.set(xlim=(0,2),ylim=(2,0),xticks=[.5,1.5],yticks=[.5,1.5])
    axis.set_xticklabels(["Crypt-villus","Epithelial\ndistance"],fontsize=7.5)
    axis.set_yticklabels(["Zero-shot","IF fine-tuned"],fontsize=7.65)
    axis.tick_params(length=0,pad=4)
    for spine in axis.spines.values():spine.set_visible(False)
    top_text(figure,494.49,28.79,"IF transfer · Spearman ρ",ha="center",fontsize=8.1,fontweight="bold")
    page.details["transfer"]={"cells":4843,"spearman_only":True,"values":values,"tiles":"opaque RGB vector fills"}
    page.save(figure,"panel_B_transfer")


def history(page):
    frames=[pd.read_csv(page.stage(stage+"/history.csv")) for stage in ("stage1_heads","stage2_full")]
    summaries=[json.loads(page.stage(stage+"/training_summary.json").read_text()) for stage in ("stage1_heads","stage2_full")]
    data,history_info=joined_training_history(frames,summaries)
    selected1,selected2=history_info["selected_combined_epochs"]
    specs=[("C_objective",42.847,"train_loss","validation_loss","Smooth L1 objective"),
           ("C_crypt_villus",228.368,"train_axis_mae","validation_axis_mae","Crypt-villus MAE"),
           ("C_epithelial",413.889,"train_epithelial_mae","validation_epithelial_mae","Epithelial-distance MAE")]
    figure=page.figure()
    for ident,x,tr,val,title in specs:
        axis=page.axis(figure,ident,[x,138.073,138.449 if x==42.847 else 138.448,98.832])
        axis.plot(data.completed_epoch,data[tr],color="#555555",marker="o",ms=2.5,lw=1,label="Training")
        axis.plot(data.completed_epoch,data[val],color="#2878a8",marker="o",ms=2.5,lw=1,label="Validation")
        axis.axvline(history_info["stage_boundary"],color="#777777",lw=.7,ls=":")
        axis.scatter([selected1],[data.loc[data.completed_epoch.eq(selected1),val].iloc[0]],marker="*",s=35,facecolor="white",edgecolor="#777777",lw=.6,zorder=4,label="Stage-1 selection")
        axis.scatter([selected2],[data.loc[data.completed_epoch.eq(selected2),val].iloc[0]],s=21,color="#c75b12",zorder=4,label="Selected checkpoint")
        axis.set_title(title,fontsize=8.5,pad=6);axis.set_xlabel("Completed epoch",fontsize=7.5,labelpad=4)
        axis.spines[["top","right"]].set_visible(False)
        if ident=="C_objective":axis.legend(frameon=False,fontsize=6,loc="best",handlelength=2,labelspacing=.35)
        if ident=="C_crypt_villus":
            axis.text(.45,.99,"heads only",transform=axis.transAxes,ha="right",va="top",fontsize=6,color="#555555")
            axis.text(.47,.99,"full model",transform=axis.transAxes,ha="left",va="top",fontsize=6,color="#555555")
    top_text(figure,280.04,117.35,"IF supervised fine-tuning history",ha="center",fontsize=9)
    page.details["IF_history"]=history_info
    page.save(figure,"panel_C_history")


def controls(page):
    page.frozen("imap/p14_gate_definitions.json")
    figure=page.figure()
    for condition,rect,title_y,barrect in [
        ("DMSO",[44.887,322.236,173.601,111.996],303.017,[223.21,328.96,4.92,98.52]),
        ("RARi",[287.556,322.236,173.601,111.996],303.1273,[465.85,328.96,4.92,98.52])]:
        artist=imap(page,figure,condition,"non-P14",rect,title_y=title_y,
                    title_x=126.07029 if condition=="DMSO" else 374.92162,
                    xlabel=True,
                    xlabel_anchor=(80.6858 if condition=="DMSO" else 331.6822,457.705))
        cax=page.axis(figure,f"D_{condition}_density_bar",barrect)
        colorbar(figure,cax,artist,ticks=[0,5,10,15],label="Probability density")
    page.save(figure,"panel_E_non_P14_IMAPs")


def peyer_windows(page):
    page.read(HOME/"Figure_4_Extended/peyer.py",role="native_Peyer_contrast_and_scale_helper")
    page.read(DISPLAY_CONFIG, role="shared_Peyer_probability_display")
    page.read(HOME/"Figure_4_Extended/probability_display.py", role="shared_Peyer_probability_display_helper")
    display, probability_norm = probability_display()
    from frozen_preprocessing.peyer_representation_v1.release import load_release, RELEASE_POINTER
    active_path, active = load_release()
    page.read(RELEASE_POINTER, role="active_classifier_release_pointer")
    page.read(active_path, role="representation_classifier_release")
    rec = active["window_release"]
    manifest=json.loads(page.read(rec["path"],rec["sha256"],"representation_classifier_window_release").read_text())
    if manifest["status"]!="FROZEN_REPRESENTATION_CLASSIFIER_WINDOWS" or manifest["threshold"]!=.5:
        raise RuntimeError("Unexpected corrected classifier release")
    for key in ("checkpoint","label_lock","window_lock"):
        rec=manifest[key];page.read(rec["path"],rec["sha256"],key)
    lock=json.loads(Path(manifest["window_lock"]["path"]).read_text())
    rec=manifest["inference"]["if"]["predictions"]
    rows=pd.read_csv(page.read(rec["path"],rec["sha256"],"corrected_Peyer_IF_predictions"))
    imagepath=page.read(HOME/"Figure4/imap_if/outputs/image_cache/figure4_if_series0_channel0_uint16.npy",
                       "0451386a72b456d622e3f2f595d13d6e7cd5bdea830743b034894be8afb5f6ec","native_IF_DAPI_cache")
    image=np.load(imagepath,mmap_mode="r")
    figure=page.figure();artist=None
    rects={"DMSO":([9.2493,491.7792,139.9806,139.9806],[157.6289,491.5424,139.98066,140.45396]),
           "RARi":([301.7755,490.2391,139.9806,139.9806],[450.15504,490.00233,139.98062,140.45393])}
    for window in lock["IF_windows"]:
        condition=window["condition"];x0,x1,y0,y1=window["bounds_px"]
        selected=rows.loc[rows.condition.eq(condition)].copy()
        if len(selected)!=window["cells"]:raise RuntimeError("Fixed Peyer window membership changed")
        cx,cy=window["center_um"];half=window["width_um"]/2;bounds=(cx-half,cx+half,cy-half,cy+half)
        selected["x_um"]=selected.centroid_x_fullres_px*PIXEL_UM
        selected["y_um"]=selected.centroid_y_fullres_px*PIXEL_UM
        ia=page.axis(figure,f"E_{condition}_image",rects[condition][0])
        pa=page.axis(figure,f"E_{condition}_probability",rects[condition][1])
        ia.imshow(contrast(image[y0:y1,x0:x1]),cmap="gray",extent=(bounds[0],bounds[1],bounds[3],bounds[2]),interpolation="none",aspect="auto")
        order=np.argsort(selected.peyer_probability.to_numpy())
        pa.scatter(selected.x_um,selected.y_um,s=2,color="#c7c7c7",lw=0,rasterized=False)
        artist=pa.scatter(selected.x_um.to_numpy()[order],selected.y_um.to_numpy()[order],
            c=selected.peyer_probability.to_numpy()[order],cmap=display["colormap"],norm=probability_norm,
            s=5.5,lw=0,rasterized=False)
        for axis in [ia,pa]:
            axis.set(xlim=bounds[:2],ylim=(bounds[3],bounds[2]));axis.axis("off");scale_bar(axis,bounds)
        top_text(figure,rects[condition][0][0]+69.99,rects[condition][0][1]-4.5,f"{condition} IF image",ha="center",fontsize=7.5)
        top_text(figure,rects[condition][1][0]+69.99,rects[condition][1][1]-4.5,"Predicted Peyer’s patch probability",ha="center",fontsize=7.5)
        page.details["Peyer_"+condition]={"bounds_px":window["bounds_px"],"cells":len(selected),"native_DAPI_only_raster":True,
             "probability_range_audit":probability_range(selected.peyer_probability),
             "probability_markers_vector":True,"marker_area_pt2":5.5,"scale_bar_um":100}
    cax=page.axis(figure,"E_shared_probability_bar",[236.5045,636.9741,149.352,7.392])
    cb=colorbar(figure,cax,artist,horizontal=True,label="Peyer’s patch probability")
    label_probability_colorbar(cb)
    page.save(figure,"panel_F_Peyer_windows")


def prepare(output_dir:Path,overwrite=False):
    page=PageInputs(output_dir,overwrite);layout=json.loads(LAYOUT.read_text())
    page.read(HOME/"Figure_4/frozen_annotations.py",role="frozen_workflow_annotation_helper")
    page.read(layout["reference"],layout["reference_sha256"],"September5_layout_reference_only")
    workflow(page,layout);transfer(page);history(page);controls(page);peyer_windows(page)
    # The controls/windows are drawn at historical registered coordinates.
    # Translate their intact fresh layers into new E/F slots; the shared IMAP
    # label registration continues to describe its immutable source artwork.
    for element in page.elements:
        if element.get('panel') in {'E','F'} and element['kind']=='pdf':
            top,shift=(270,173) if element['panel']=='E' else (469,172)
            element.update(source_rect_pt=[0,top,PAGE[0],200],
                           rect_pt=[0,top+shift,PAGE[0],200])
    for anchor in page.anchors:
        if anchor['id'].startswith(('D_','E_')):
            old=anchor['id'][0]
            anchor['rect_pt'][1]+=173 if old=='D' else 172
            anchor['id']=('E' if old=='D' else 'F')+anchor['id'][1:]
    script=HOME/'Figure_4_Extended/panel_D/figure_script/make_panel_D_gate_backprojection.py'
    manifest=SATA/'Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json'
    page.read(manifest,'f3a67a9481d9969126e2df03d2e7e598158bcd286e92295d717779c7c8817d5b','fixed_gate_backprojection')
    page.read(script,role='gate_backprojection_renderer')
    command=[sys.executable,'-B',str(script),'--manifest',str(manifest),
             '--page-layout',str(LAYOUT),'--output-dir',str(page.out/'panel_D')]
    subprocess.run(command,check=True)
    provenance=page.out/'panel_D/panel_D_gate_backprojection_provenance.json'
    page.inputs.extend(json.loads(provenance.read_text())['inputs'])
    page.read(provenance,role='fresh_gate_backprojection_provenance')
    page.elements.append({'id':'panel_D_gate_backprojection','panel':'D','kind':'pdf',
        'path':'panel_D/panel_D_gate_backprojection.pdf','rect_pt':[0,0,*PAGE],'fit':'exact'})
    page.anchors.extend(json.loads((page.out/'panel_D/panel_D_gate_backprojection.geometry.json').read_text())['rendered_anchors'])
    page.details['gate_backprojection_command']=command
    for letter,x,y in [("a.",12.33,18.86),("b.",387.99,21.01),("c.",8.74,109.46),("d.",9.81,266),("e.",9.42,468),("f.",9.81,646.62)]:
        page.text(letter,x,y,size=10.2,panel=letter[0].upper())
    page.text('Anatomical localization of IMAP gates',297.638,267,size=8.5,align='center',panel='D')
    return page.finish(LAYOUT,Path(__file__))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--overwrite",action="store_true")
    a=p.parse_args();prepare(a.output_dir,a.overwrite)
