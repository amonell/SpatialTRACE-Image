#!/usr/bin/env python3
"""Revise only Figure 4 E–G and the corresponding Extended 4E cross-reference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import shutil
import sys
from xml.dom import minidom
from zipfile import ZipFile

E = (' P14 cells per 10,000 non-P14 cells across each section. ')
F = (' P14 enrichment in each coordinate gate, expressed as the log2 ratio of the fraction of P14 cells '
     'to the fraction of non-P14 cells in that gate. A pseudocount of 0.5 was added to gate-specific P14 '
     'counts before calculating enrichment. In e and f, each point represents one mouse; horizontal '
     'lines and whiskers show the mean ± SEM. Two-sided P values were obtained from linear models '
     'including treatment and experiment, fitted to log2 abundance in e and log2 enrichment in f. '
     'Gate comparisons were adjusted by the Benjamini–Hochberg procedure across the five gates (q). '
     'Significant comparisons are labeled: *P or q < 0.05; **P < 0.01. ')
G = (' Immune Allocation Maps (IMAPs) of P14 cells from DMSO (33,114 cells) and RARi (10,310 cells) mice. '
     'Colors and contours show the condition mean of individually normalized mouse probability densities, '
     'with equal weight for each mouse. Gate labels show mean within-mouse percentages of P14 cells. '
     'Both conditions use the same density color scale. Gate names describe provisional anatomical '
     'interpretations of the coordinate ranges. Musc, muscularis; IE, intraepithelial; LP, lamina propria. '
     'Panels e–g include four DMSO and five RARi mice from two experiments, with one small-intestinal '
     'section per mouse, after annotated tissue exclusions.')
EXT_E = (' Non-P14 IMAPs for DMSO (748,234 cells from four mice) and RARi (776,519 cells from five mice). '
         'Gates and per-mouse density normalization match Figure 4g. Colors and contours show the '
         'equal-weight mean of mouse probability densities; gate labels show mean within-mouse '
         'percentages of non-P14 cells. The two non-P14 maps share a density color scale. Musc, muscularis. ')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(node):
    return ''.join(t.firstChild.data for t in node.getElementsByTagName('w:t') if t.firstChild)


def set_text(run, value):
    nodes = run.getElementsByTagName('w:t')
    assert len(nodes) == 1
    node = nodes[0]
    for child in list(node.childNodes):
        node.removeChild(child)
    node.appendChild(node.ownerDocument.createTextNode(value))
    node.setAttribute('xml:space', 'preserve')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--publish-reviewed', action='store_true')
    args = parser.parse_args()
    output = args.output_dir/'2026_10_15_Figure_Legends.docx'
    manifest_path = args.output_dir/'legend_revision.json'
    if args.publish_reviewed:
        report = json.loads(manifest_path.read_text())
        assert digest(output) == report['output']['sha256']
        assert digest(args.source) == report['source']['sha256']
        source_dir = args.source.parent
        assert list(source_dir.glob('*.docx')) == [args.source]
        destination = source_dir/output.name
        archive = source_dir/'Old_Legends/figure4_efg_2026_10_15'/args.source.name
        if archive.exists() or destination.exists():
            raise FileExistsError('A publication or archive already exists')
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source, archive)
        assert digest(archive) == report['source']['sha256']
        # Copy the reviewed document before moving the superseded current one.
        shutil.copy2(output, destination)
        assert digest(destination) == report['output']['sha256']
        # Original is already backed up byte-for-byte; moving preserves it too.
        retained = args.output_dir/'previous_current'/args.source.name
        retained.parent.mkdir(parents=True, exist_ok=False)
        shutil.move(str(args.source), str(retained))
        report.update(status='REVIEWED_AND_PUBLISHED', current_document=str(destination),
                      previous_document_archive=str(archive), previous_current_copy=str(retained))
        manifest_path.write_text(json.dumps(report, indent=2)+'\n')
        print(destination)
        return
    if output.exists() or manifest_path.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.source) as original:
        infos = original.infolist()
        entries = {i.filename: original.read(i.filename) for i in infos}
    document = minidom.parseString(entries['word/document.xml'])
    body = document.getElementsByTagName('w:body')[0]
    paragraphs = [p for p in body.childNodes if p.nodeName == 'w:p']
    before = [p.toxml() for p in paragraphs]
    captions = {}
    for prefix in ('Figure 4 |', 'Extended Data Figure 4 |'):
        heading = next(i for i, p in enumerate(paragraphs) if text(p).startswith(prefix))
        captions[prefix] = heading+1
    main_index, extended_index = captions.values()
    main = paragraphs[main_index]
    runs = list(main.getElementsByTagName('w:r'))
    assert len(runs) == 16 and text(runs[12]) == 'e,' and text(runs[14]) == 'f,'
    prefix_xml = [r.toxml() for r in runs[:12]]
    set_text(runs[13], E)
    set_text(runs[15], F)
    letter, prose = runs[12].cloneNode(True), runs[13].cloneNode(True)
    set_text(letter, 'g,')
    set_text(prose, G)
    main.appendChild(letter)
    main.appendChild(prose)
    assert [r.toxml() for r in list(main.getElementsByTagName('w:r'))[:12]] == prefix_xml
    extended = paragraphs[extended_index]
    ext_runs = list(extended.getElementsByTagName('w:r'))
    assert len(ext_runs) == 12 and text(ext_runs[8]) == 'e,'
    ext_before = [r.toxml() for r in ext_runs]
    set_text(ext_runs[9], EXT_E)
    assert all(r.toxml() == ext_before[i] for i, r in enumerate(ext_runs) if i != 9)
    changed = [i for i, p in enumerate(paragraphs) if p.toxml() != before[i]]
    assert changed == [main_index, extended_index]
    revised_xml = document.toxml(encoding='UTF-8')
    with ZipFile(output, 'w') as revised:
        for info in infos:
            revised.writestr(info, revised_xml if info.filename == 'word/document.xml' else entries[info.filename])
    with ZipFile(output) as revised:
        assert set(revised.namelist()) == set(entries)
        assert all(revised.read(name) == data for name, data in entries.items() if name != 'word/document.xml')
    review = '\n\n'.join(text(paragraphs[i-1])+'\n'+text(paragraphs[i]) for i in changed)
    (args.output_dir/'revised_captions.txt').write_text(review+'\n')
    report = {'status': 'STAGED_FOR_VISUAL_REVIEW', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'source': {'path': str(args.source), 'sha256': digest(args.source)},
              'output': {'path': str(output), 'sha256': digest(output)},
              'script': {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
              'changed_paragraph_indices': changed, 'preserved': 'All other paragraphs, Figure4 A–D, Extended4 A–D/F, all other DOCX parts, existing citation content and run formatting',
              'changes': 'Main E/F/G caption and corresponding Extended4 E cohort/cross-reference; no Methods or manuscript changes'}
    manifest_path.write_text(json.dumps(report, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
