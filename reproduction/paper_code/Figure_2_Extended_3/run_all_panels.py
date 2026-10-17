#!/usr/bin/env python3
"""Render the three new graph-design components; never train models."""
import argparse
from paper_paths import Path, lock_record, rendering_script
import shlex
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--panels',nargs='+',choices=list('ABC'),default=list('ABC'))
    p.add_argument('--overwrite',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    names={'A':'graph_components','B':'matched_annotation_budgets','C':'heldout_spatial_errors'}
    for panel in a.panels:
        cmd=[sys.executable,str(Path(__file__).parent/f'panel_{panel}/figure_script/make_panel_{panel}_{names[panel]}.py'),
             '--release',str(a.release),'--output-dir',str(a.output_root/f'panel_{panel}')]
        if a.overwrite:cmd.append('--overwrite')
        print(shlex.join(cmd),flush=True)
        if not a.dry_run:subprocess.run(cmd,check=True)


if __name__=='__main__':main()
