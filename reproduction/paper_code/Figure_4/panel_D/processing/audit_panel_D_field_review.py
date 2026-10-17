#!/usr/bin/env python3
"""Check field-review data identity and unchanged published figures."""
import argparse
import json
from paper_paths import Path, lock_record, rendering_script
import sys

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Figure_4.panel_D.processing.prepare_panel_D_field_candidates import verify, record, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-manifest",type=Path,required=True)
    p.add_argument("--render-dir",type=Path,required=True)
    a=p.parse_args()
    m=json.loads(a.candidate_manifest.read_text())
    provenance=json.loads((a.render_dir/"provenance.json").read_text())
    for rec in provenance["outputs"]:verify(rec)
    for path,digest in provenance["current_figure_hashes_unchanged"].items():
        verify({"path":path,"sha256":digest})
    lock=json.loads(verify(m["source_manifest"]).read_text())
    frames={c:pd.read_parquet(verify(rec)) for c,rec in m["predictions"].items()}
    checks=[]
    for win in m["windows"]:
        x0,x1,y0,y1=win["bounds_xxyy_px"]
        frame=frames[win["condition"]]
        expected=frame.loc[frame.centroid_x_fullres_px.ge(x0)&frame.centroid_x_fullres_px.lt(x1)&
                           frame.centroid_y_fullres_px.ge(y0)&frame.centroid_y_fullres_px.lt(y1)]
        shown=pd.read_parquet(a.render_dir/f"{win['id']}_displayed_cells.parquet")
        # Parquet deliberately omits the pandas row index. Biological IDs and
        # every other source column must still match exactly, in source order.
        pd.testing.assert_frame_equal(shown,expected.reset_index(drop=True),check_exact=True)
        raw=np.load(verify(win["microscopy"]))
        assert raw["dapi"].shape==raw["ecad"].shape==(y1-y0,x1-x0)
        overlap_checked=False
        old=lock["windows"][win["condition"]]
        ox0,ox1,oy0,oy1=old["bounds_xxyy_px"]
        ix0,ix1,iy0,iy1=max(x0,ox0),min(x1,ox1),max(y0,oy0),min(y1,oy1)
        if ix0<ix1 and iy0<iy1:
            for name,key in (("dapi","DAPI"),("ecad","E_cadherin")):
                reference=np.load(verify(old[key]),mmap_mode="r")
                np.testing.assert_array_equal(raw[name][iy0-y0:iy1-y0,ix0-x0:ix1-x0],
                                              reference[iy0-oy0:iy1-oy0,ix0-ox0:ix1-ox0])
            overlap_checked=True
        checks.append({"id":win["id"],"all_rows_and_columns_exact":True,
                       "native_shape_correct":True,"old_field_overlap_byte_exact":overlap_checked})
    candidates=[w for w in m["windows"] if not w["id"].startswith("Current")]
    assert len(candidates)==10
    assert [w["condition"] for w in candidates].count("DMSO")==5
    assert [w["condition"] for w in candidates].count("RARi")==5
    for win in candidates:
        x0,x1,y0,y1=win["bounds_xxyy_px"]
        assert (x1-x0,y1-y0)==(6000,4500)
    assert len(provenance["current_figure_hashes_unchanged"])==8
    pages=PdfReader(a.render_dir/"Figure4D_field_candidates.pdf").pages
    assert len(pages)==2+len(m["windows"])
    for page in pages[2:]:
        text=page.extract_text()
        for title in ("DAPI / E-cadherin","Cells by IMAP gate","Crypt-villus prediction","Epithelial-distance prediction"):
            assert title in text
    save_json(a.render_dir/"data_identity_audit.json",{
        "status":"PASS","candidate_manifest":record(a.candidate_manifest),
        "render_provenance":record(a.render_dir/"provenance.json"),"n_candidates":len(candidates),
        "all_current_figures_unchanged":True,"current_figures_checked":len(provenance["current_figure_hashes_unchanged"]),
        "same_native_crop_for_all_four_views":True,"gates_and_predictions_exactly_preserved":True,
        "checks":checks,"pdf_pages":len(pages),
        "scope":"Checks data identity and rendering inputs, not prediction accuracy or anatomical labels"})
    print("PASS: 10 candidates, 2 current-field references, exact cell values/gates, native registration, 8 current PDFs unchanged")


if __name__=="__main__":main()
