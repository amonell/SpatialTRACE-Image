"""Prepare a complete Figure 1 from fresh data layers and original manual paths."""
from __future__ import annotations
import json
from paper_paths import Path, lock_record, rendering_script
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from figure_assembly.static_art import extract_manual_art,inspect_path_events,inspect_text,intersects,sha256
from figure_assembly.compositor import register_fonts
from reportlab.pdfbase.pdfmetrics import stringWidth


def model_name_elements(layout):
    """Add names inside the existing model boxes without moving their artwork."""
    register_fonts()
    result = []
    for specification in layout["model_name_labels"]:
        element = dict(specification)
        x, y, width, height = element.pop("block_rect_pt")
        size = element["font_size_pt"]
        lines = element.pop("lines", [{"text": element["text"], "y_pt": element["y_pt"]}])
        element.pop("module_lines", None)
        if element["align"] != "center" or abs(element["x_pt"] - (x + width / 2)) > 1e-6:
            raise RuntimeError("Model name must remain centered in its original box")
        for index, line in enumerate(lines):
            text_width = stringWidth(line["text"], element["font"], size)
            if text_width > width - 8 or not y + size < line["y_pt"] < y + height - 3:
                raise RuntimeError("Model name does not fit inside its original box")
            result.append({**element, **line, "id": f"{element['id']}-line-{index}"})
    return result


def module_text_style(run, layout):
    """Use the model-name font for every architecture line in its module."""
    for specification in layout["model_name_labels"]:
        x, y, width, height = specification["block_rect_pt"]
        text = run["text"].strip()
        if x < run["x_pt"] < x + width and y < run["y_pt"] < y + height:
            if text in specification.get("module_lines", {}):
                return {"font_size_pt": specification["font_size_pt"],
                        "font": specification["font"], "align": "center",
                        "x_pt": x + width / 2, "y_pt": specification["module_lines"][text]}
    return {}


def prepare(output_dir:Path,overwrite=False):
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    layout=json.loads((Path(__file__).parent/"layout.json").read_text())
    reference=Path(layout["reference"])
    if sha256(reference)!=layout["reference_sha256"]:raise RuntimeError("Figure 1 layout reference changed")
    registration=json.loads((Path(__file__).parent/"source_region_lock.json").read_text())
    if registration["artwork"]["sha256"]!=layout["reference_sha256"]:raise RuntimeError("Source-region registration belongs to different artwork")
    if registration["original_image_streams"]["/Im5"]!=registration["original_image_streams"]["/Im6"]:raise RuntimeError("Original C and D training image identities differ")
    for name in ['scientific_layers.pdf',*[f'scientific_panel_{p}.pdf' for p in 'ABCDE']]:
        if (output_dir/name).exists() and not overwrite:raise FileExistsError(output_dir/name)
    events=inspect_path_events(reference)
    selected=[e for e in events if e["index"] not in (46,47,48,53,125,132)]
    # Large filled containers are below images; arrows, outlines, and conceptual
    # quantification paths remain above them. Both sets retain original paths.
    backgrounds=[e["index"] for e in selected if e["op"] in ("f","f*","B","B*","b","b*") and e["bbox_pt"][2]*e["bbox_pt"][3]>3000]
    foreground=[e["index"] for e in selected if e["index"] not in backgrounds]
    audits=[]
    for name,indices in (("manual_background",backgrounds),("manual_foreground",foreground)):
        audits.append(extract_manual_art(reference,output_dir/(name+".pdf"),source_sha256=layout["reference_sha256"],keep_paint_indices=indices,overwrite=overwrite))
    command=[sys.executable,"-B",str(Path(__file__).parent/"render.py"),"--output-dir",str(output_dir)]
    if overwrite:command.append("--overwrite")
    subprocess.run(command,check=True)
    data=json.loads((output_dir/"scientific_layers_provenance.json").read_text())
    elements=[{"id":"manual-background","kind":"static","path":"manual_background.pdf","rect_pt":[0,0,*layout["page_size_pt"]],"z":0},
              {"id":"fresh-scientific-layers","kind":"pdf","path":"scientific_layers.pdf","rect_pt":[0,0,*layout["page_size_pt"]],"z":1},
              {"id":"manual-foreground","kind":"static","path":"manual_foreground.pdf","rect_pt":[0,0,*layout["page_size_pt"]],"z":2}]
    register_fonts()
    for i,run in enumerate(inspect_text(reference)):
        text=run["text"].strip()
        if text=="inference" and 395 < run["y_pt"] < 410:
            continue  # Second line of the replaced Figure 1d input label.
        input_label=text=="Prep for model" and 390 < run["y_pt"] < 400
        if input_label:text="DAPI input"
        if text=="Normal":text="Non-Peyer"
        font="Arial-Bold" if run["paint_count"]>1 or "Bold" in run["source_font"] else "Arial"
        size=run["font_size_pt"]
        if run["text"]==text and run["advance_pt"]>0:
            size=min(size,size*run["advance_pt"]/max(.01,stringWidth(text,font,size)))
        if "Immunofluorescence quantification"==text:
            max_width=152 if run["x_pt"]<400 else 150
            size=min(size,size*max_width/stringWidth(text,font,size))
        elements.append({"id":f"manual-label-{i:03d}","kind":"text","text":text,
                         "x_pt":300.4384 if input_label else run["x_pt"],
                         "y_pt":401.6568 if input_label else run["y_pt"],
                         "align":"center" if input_label else "left",
                         "font_size_pt":size,"font":font,"rotation_degrees":run["rotation_degrees"],
                         "color":run["color"],"z":3, **module_text_style(run, layout)})
    elements.extend(model_name_elements(layout))
    inputs=data["inputs"]+[{"path":str(reference),"sha256":layout["reference_sha256"],"role":"manual_vector_and_layout_reference"}]
    for file in (Path(__file__),Path(__file__).parent/"layout.json",Path(__file__).parent/"render.py",Path(__file__).parent/"render_util.py",ROOT/'Figure_1/common.py',ROOT/"figure_assembly/static_art.py"):
        inputs.append({"path":str(file),"sha256":sha256(file),"role":"assembly_source"})
    return {"elements":elements,"inputs":inputs,"commands":[command],"warnings":[],
            "static_art_audits":audits,"rendered_anchors":data["rendered_anchors"],"scientific_panel_layers":data['scientific_panel_layers'],"scientific_axis_panels":data['scientific_axis_panels'],
            "anchors":[{"id":k,"rect_pt":v["rect_pt"]} for k,v in layout["image_slots"].items()]+[{"id":k,"rect_pt":v} for k,v in layout["numeric_slots"].items()],
            "intentional_differences":layout["intentional_differences"]}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args();result=prepare(args.output_dir,args.overwrite)
    (args.output_dir/"preparation.json").write_text(json.dumps(result,indent=2)+"\n")
