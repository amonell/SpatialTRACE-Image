"""Render fresh paper figures from a verified frozen-input bundle."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--figures', nargs='+', default=['Figure_1', 'Figure_2', 'Figure_2_Extended', 'Figure_2_Extended_2', 'Figure_3', 'Figure_3_Extended', 'Figure_4', 'Figure_4_Extended'])
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--keep-going', action='store_true', help='Attempt remaining figures if one fails; return a failing exit status')
    parser.add_argument('--arial-dir', type=Path, help='Directory containing the four licensed Arial TTF files (see README)')
    parser.add_argument('--allow-code-changes', action='store_true', help='Render explicitly edited figure code; frozen data hashes remain mandatory')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.arial_dir:
        os.environ['TISSUEMAPPER_ARIAL_DIR'] = str(args.arial_dir.resolve())
    bundle = args.bundle.resolve()
    source = Path(__file__).resolve().parent/'paper_code'
    if not source.is_dir():
        source = bundle/'paper_code'
    output = args.output_dir.resolve()
    if args.worker:
        sys.path[:0] = [str(Path(__file__).parent), str(source)]
        os.environ['TISSUEMAPPER_PAPER_BUNDLE'] = str(bundle)
        os.environ['TISSUEMAPPER_PAPER_OUTPUT'] = str(output)
        os.environ['TISSUEMAPPER_PAPER_CODE'] = str(source)
        sys.dont_write_bytecode = True
        import paper_paths
        from figure_assembly.compositor import compose_page
        from figure_assembly.qa import check_page, render_png
        from figure_assembly.panels import build_panel_layers, verify_recomposition
        spec = importlib.util.spec_from_file_location('paper_prepare', source/args.worker/'assembly/prepare.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        components = output/'components'
        components.mkdir()
        result = module.prepare(components)
        layout = json.loads((source/args.worker/'assembly/layout.json').read_text())
        layout['elements'] = result['elements']
        catalog = json.loads((source/'final_figure_order.json').read_text())
        entry = next(f for f in catalog['figures'] if f['package'] == args.worker)
        pdf = output/Path(entry['reference']).name
        direct = components/'direct_composition.pdf'
        composition = compose_page(layout, components, direct)
        panels = build_panel_layers(layout, result, components)
        layered = compose_page({**layout, 'elements': panels['elements']}, components, pdf)
        recomposition = verify_recomposition(direct, pdf, output/'qa/recomposition')
        render_png(pdf, pdf.with_suffix('.png'), int(layout.get('png_dpi', 600 if args.worker in {'Figure_4', 'Figure_4_Extended'} else 300)))
        reference = paper_paths.Path(catalog['output_root'])/entry['reference']
        qa = check_page(pdf, reference, layout, composition, result, output/'qa/page')
        (output/'reproduction.json').write_text(json.dumps(dict(preparation=result, composition=composition,
            panel_composition=layered, final_panels=panels, panel_recomposition=recomposition, qa=qa,
            timestamp_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
            python=platform.python_version(),
            versions={name: importlib.metadata.version(name) for name in
                      ['numpy', 'matplotlib', 'pandas', 'pypdf', 'reportlab', 'pymupdf']},
            graph_checkpoint_verification='Historical model identities only; Graph binaries are intentionally not distributed. Frozen display-input hashes are verified.',
            output_sha256=digest(pdf)), default=str, indent=2))
        if qa['errors']:
            raise RuntimeError(f'Figure QA failed: {qa["errors"]}')
        return
    manifest = json.loads((bundle/'data_manifest.json').read_text())
    for item in manifest['files']:
        path = (bundle/item['path']).resolve()
        if not path.is_relative_to(bundle) or digest(path) != item['sha256']:
            raise ValueError(f'Bundle input mismatch: {item["path"]}')
    print(f'Verified {len(manifest["files"])} frozen inputs', flush=True)
    code_manifest = Path(__file__).resolve().parent/'code_manifest.json'
    if code_manifest.exists():
        for item in json.loads(code_manifest.read_text())['files']:
            path = (source/item['path']).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Invalid figure-code manifest path')
            if digest(path) != item['sha256']:
                if not args.allow_code_changes:
                    raise ValueError(f'Figure code changed: {item["path"]}; use --allow-code-changes only for intentional edits')
                print(f'Using explicitly modified code: {item["path"]}', flush=True)
    if args.verify_only:
        return
    for command in ['pdftoppm', 'pdffonts', 'pdftotext', 'pdfimages']:
        if shutil.which(command) is None:
            raise RuntimeError(f'Install Poppler before rendering; missing command: {command}')
    packages = {f['package'] for f in json.loads((source/'final_figure_order.json').read_text())['figures']}
    if not set(args.figures) <= packages:
        raise ValueError(f'Unknown figure; choose from {sorted(packages)}')
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TISSUEMAPPER_PAPER_BUNDLE=str(bundle), TISSUEMAPPER_PAPER_OUTPUT=str(output), TISSUEMAPPER_PAPER_CODE=str(source), MPLBACKEND='Agg',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), str(source)]))
    failures = []
    for package in args.figures:
        target = output/package
        target.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--bundle', str(bundle),
                   '--output-dir', str(target), '--worker', package]
        print('Rendering '+package, flush=True)
        with (target/'render.log').open('w') as log:
            status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            failures.append(package)
            print(f'{package} failed; inspect {target/"render.log"}', flush=True)
            if not args.keep_going:
                break
    if failures:
        raise RuntimeError('Figure rendering failed: ' + ', '.join(failures))
    print(output, flush=True)


if __name__ == '__main__':
    main()
