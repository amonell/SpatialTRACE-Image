#!/usr/bin/env python3
"""Render the frozen nine-mouse non-P14 maps with one shared colorbar."""
from __future__ import annotations
import argparse
from paper_paths import Path, lock_record, rendering_script
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Figure_4.assembly.common import HOME, PageInputs, sha256
from Figure_4.replicate_revision_20261014.final_panels import load_inputs, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-pointer', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dpi', type=int, choices=[600], default=600)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for name in ('panel_F_non_P14_IMAPs.pdf', 'panel_F_non_P14_IMAPs.png',
                 'assembly_inputs_provenance.json'):
        if (args.output_dir/name).exists() and not args.overwrite:
            raise FileExistsError(args.output_dir/name)
    page = PageInputs(args.output_dir, args.overwrite)
    data = load_inputs(page, args.cohort_pointer)
    controls(page, data, panel='F')
    for item in list(page.outputs):
        pdf = Path(item['path'])
        subprocess.run(['pdftoppm', '-singlefile', '-r', str(args.dpi), '-png',
                        str(pdf), str(pdf.with_suffix(''))], check=True)
        png = pdf.with_suffix('.png')
        page.outputs.append({'path': str(png), 'sha256': sha256(png)})
    page.finish(HOME/'Figure_4_Extended/assembly/layout.json', Path(__file__),
                command=[sys.executable, *sys.argv])


if __name__ == '__main__':
    main()
