#!/usr/bin/env python3
"""Render the frozen nine-mouse P14 IMAP pair at its final lower-left F anchor."""
import argparse
from paper_paths import Path, lock_record, rendering_script
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Figure_4.assembly.common import HOME, PageInputs, sha256
from Figure_4.replicate_revision_20261014.final_panels import load_inputs, p14_imaps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cohort-pointer', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for name in ('panel_F_P14_IMAPs.pdf', 'panel_F_P14_IMAPs.png', 'assembly_inputs_provenance.json'):
        if (args.output_dir/name).exists() and not args.overwrite:
            raise FileExistsError(args.output_dir/name)
    page = PageInputs(args.output_dir, args.overwrite)
    data = load_inputs(page, args.cohort_pointer)
    p14_imaps(page, data)
    for output in list(page.outputs):
        pdf = Path(output['path'])
        subprocess.run(['pdftoppm', '-singlefile', '-r', '600', '-png', str(pdf), str(pdf.with_suffix(''))], check=True)
        png = pdf.with_suffix('.png')
        page.outputs.append({'path': str(png), 'sha256': sha256(png)})
    page.finish(HOME/'Figure_4/assembly/layout.json', Path(__file__), command=[sys.executable, *sys.argv])
