"""Shared figure-only helpers for the user-requested graph comparisons."""
import json
import shutil
import sys
from datetime import datetime,timezone
from paper_paths import Path, lock_record, rendering_script
import pandas as pd

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from Figure_4.common import configure_figure_fonts,require_file,protect_outputs,save_pair,file_record,write_provenance

NAMES={'full':'SpatialTRACE-Graph','cell_only':'Cell only','uniform':'Uniform neighbors','randomized':'Randomized connections'}
COLORS={'full':'#202020','cell_only':'#3d79a6','uniform':'#bb7438','randomized':'#836797'}
AXES=[('crypt_villus','Crypt–villus coordinate'),('epithelial_distance','Epithelial-distance coordinate')]


def load_table(release_path, name):
    release=json.loads(require_file(release_path).read_text())
    if release['status']!='FROZEN_COMPLETE_COMPARISON':raise RuntimeError('Graph comparison is incomplete')
    rec=release['outputs'][name];path=require_file(rec['path'],rec['sha256'])
    return release,pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,sep='\t')


def export(fig, a, name, panel, release, script):
    stem=a.output_dir.resolve()/name;provenance=stem.with_name(name+'_provenance.json')
    outputs=[stem.with_suffix('.pdf'),stem.with_suffix('.png')]
    overwritten=protect_outputs(outputs+[provenance],a.overwrite)
    if overwritten:
        archive=a.output_dir.parent/'polished_panel_archive'/datetime.now(timezone.utc).strftime(f'%Y%m%dT%H%M%SZ_panel_{panel}')
        archive.mkdir(parents=True)
        for path in outputs+[provenance]:
            if path.exists():shutil.copy2(path,archive/path.name)
    save_pair(fig,stem,a.dpi)
    write_provenance(provenance,script=script,inputs=[file_record(a.release,'frozen_graph_comparison')]+[
        file_record(rec['path'],key) for key,rec in release['outputs'].items()],outputs=outputs,
        overwritten=overwritten,extra={'panel':panel,'aggregation_unit':'held-out section',
        'feature_lineage':release['feature_lineage'],'model_fitting_or_selection_in_renderer':False,
        'spatial_example':release.get('spatial_example')})


def parser(description):
    import argparse
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--overwrite',action='store_true')
    return p
