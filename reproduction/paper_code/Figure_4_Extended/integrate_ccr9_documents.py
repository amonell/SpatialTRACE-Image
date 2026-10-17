#!/usr/bin/env python3
"""Apply the owner's scoped CCR9 addition and Extended Figure 4 cross-reference shift."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import re
import shutil
from zipfile import ZipFile

from lxml import etree

ROOT = Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript')
AUDIT = ROOT / 'audits/extended4_iel_ccr9_2026-10-16'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'
XML_SPACE = '{http://www.w3.org/XML/1998/namespace}space'
SHIFT = dict(zip('abcdef', 'bcdefg'))
REFERENCE = re.compile(r'(Extended Data (?:Fig\.|Figure)\s*4)([a-f](?:\s*[–−-]\s*[a-f])?)')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(paragraph):
    return ''.join(paragraph.xpath('.//w:t/text()', namespaces=NS))


def replace_span(paragraph, start, end, replacement):
    nodes = paragraph.xpath('.//w:t', namespaces=NS)
    cursor = 0
    selected = []
    for node in nodes:
        size = len(node.text or '')
        if cursor < end and cursor + size > start:
            selected.append((node, max(0, start - cursor), min(size, end - cursor)))
        cursor += size
    if not selected:
        raise RuntimeError('Word text span did not resolve')
    for i, (node, left, right) in enumerate(selected):
        node.text = (node.text or '')[:left] + (replacement if i == 0 else '') + (node.text or '')[right:]
        node.set(XML_SPACE, 'preserve')


def append_run(paragraph, value, source_run=None, prepend=False):
    run = etree.Element(W + 'r')
    if source_run is not None and source_run.find(W + 'rPr') is not None:
        run.append(deepcopy(source_run.find(W + 'rPr')))
    node = etree.SubElement(run, W + 't')
    node.set(XML_SPACE, 'preserve')
    node.text = value
    if prepend:
        paragraph.insert(1 if paragraph.find(W + 'pPr') is not None else 0, run)
    else:
        paragraph.append(run)
    return run


def new_paragraph(template, value):
    paragraph = etree.Element(W + 'p')
    if template.find(W + 'pPr') is not None:
        paragraph.append(deepcopy(template.find(W + 'pPr')))
    append_run(paragraph, value, template.find(W + 'r'))
    return paragraph


def prepare():
    staged = AUDIT / 'documents_staged'
    staged.mkdir(exist_ok=False)
    specs = [('Manuscript', '2026_09_07_TissueFrame_Manuscript.docx', '2026_10_16_TissueFrame_Manuscript.docx', 'Old_versions'),
             ('Figure_Legends', '2026_10_15_Figure_Legends.docx', '2026_10_16_Figure_Legends.docx', 'Old_Legends'),
             ('Methods', '2026_09_07_Methods.docx', '2026_10_16_Methods.docx', 'Old_versions')]
    changes = []
    for folder, old_name, new_name, archive in specs:
        source = ROOT / folder / old_name
        if list((ROOT / folder).glob('*.docx')) != [source]:
            raise RuntimeError(f'Current {folder} changed; inspect before applying')
        if not (AUDIT / old_name).exists():
            shutil.copy2(source, AUDIT / old_name)
        with ZipFile(source) as package:
            original = {item.filename: package.read(item.filename) for item in package.infolist()}
            infos = package.infolist()
        root = etree.fromstring(original['word/document.xml'])
        paragraphs = root.xpath('//w:body/w:p', namespaces=NS)
        updates = []
        for i, paragraph in enumerate(paragraphs):
            before = text(paragraph)
            for match in reversed(list(REFERENCE.finditer(before))):
                suffix = ''.join(SHIFT.get(c, c) for c in match[2])
                replace_span(paragraph, match.start(2), match.end(2), suffix)
            if text(paragraph) != before:
                updates.append({'paragraph': i, 'before': before, 'after': text(paragraph), 'kind': 'shift_existing_extended_panel_reference'})
        if folder == 'Manuscript':
            paragraph = paragraphs[36]
            before = text(paragraph)
            if not before.endswith('(Fig. 4a).'):
                raise RuntimeError('Figure 4A introduction differs from the reviewed paragraph')
            append_run(paragraph, ' In a flow-cytometry validation experiment, RARi-treated mice had a lower percentage of CCR9+ cells among live CD8+ CD45.1+ intraepithelial lymphocyte (IEL) donors (Extended Data Fig. 4a).', paragraph.find(W + 'r'))
            updates.append({'paragraph': 36, 'before': before, 'after': text(paragraph), 'kind': 'add_CCR9_readout_beside_main_Figure_4A'})
        elif folder == 'Figure_Legends':
            paragraph = paragraphs[15]
            before = text(paragraph)
            replace_span(paragraph, 0, len(before), 'Extended Data Figure 4 | RARi response, immunofluorescence adaptation, and spatial controls.')
            updates.append({'paragraph': 15, 'before': before, 'after': text(paragraph), 'kind': 'scope_legend_title'})
            paragraph = paragraphs[16]
            before = text(paragraph)
            for match in reversed(list(re.finditer(r'(?<!\w)([a-f]),', before))):
                replace_span(paragraph, match.start(1), match.end(1), SHIFT[match[1]])
            runs = paragraph.findall(W + 'r')
            body = ' Representative CCR9-versus-CD45.1 flow plots and CCR9+ percentages among live CD8+ CD45.1+ IEL donors. Each point represents one mouse (three DMSO and four RARi); bars and whiskers show mean ± SD. Two-sided unpaired Welch t test, P = 0.0137. DMSO mouse 8 was excluded for failed cell recovery. The CCR9 endpoint was selected after data inspection and is exploratory. '
            append_run(paragraph, body, runs[1], prepend=True)
            append_run(paragraph, 'a,', runs[0], prepend=True)
            updates.append({'paragraph': 16, 'before': before, 'after': text(paragraph), 'kind': 'add_A_legend_and_reletter_existing_A_to_F_as_B_to_G'})
        else:
            heading = new_paragraph(paragraphs[5], 'CCR9 flow-cytometry analysis')
            body = new_paragraph(paragraphs[4], 'For the flow-cytometry validation in Extended Data Figure 4a, compensated FCS events were gated for lymphocytes, FSC singlets, live CD8+ cells, and CD45.1+ donors. CCR9+ frequency was calculated from all events within the donor gate. The same edited compensation matrix and native gate hierarchy were used for both conditions. DMSO mouse 8 was excluded for failed cell recovery, leaving three DMSO and four RARi mice. Independent mice were compared using a two-sided unpaired Welch t test; bars show mean ± SD. The endpoint was selected after data inspection and is exploratory; the exact two-sided Mann–Whitney sensitivity test gave P = 0.0571. Representative samples were chosen by proximity to the within-group median CCR9+ percentage, with ties resolved by higher donor event count and then lower mouse number.')
            parent = paragraphs[5].getparent()
            index = parent.index(paragraphs[5])
            parent.insert(index, heading)
            parent.insert(index + 1, body)
            updates.append({'insert_before_original_paragraph': 5, 'after': text(heading) + '\n' + text(body), 'kind': 'add_frozen_CCR9_analysis_methods'})
        target = staged / new_name
        payload = etree.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
        with ZipFile(target, 'w') as package:
            for item in infos:
                package.writestr(item, payload if item.filename == 'word/document.xml' else original[item.filename])
        with ZipFile(target) as package:
            assert package.testzip() is None
            assert all(package.read(key) == value for key, value in original.items() if key != 'word/document.xml')
        changes.append({'folder': folder, 'source': str(source), 'source_sha256': sha(source), 'staged': str(target),
                        'staged_sha256': sha(target), 'destination': str(ROOT / folder / new_name),
                        'archive': str(ROOT / folder / archive / '2026_10_16_IEL_CCR9_integration' / old_name),
                        'changes': updates, 'other_docx_package_parts_unchanged': True})
    manifest = {'status': 'STAGED_FOR_VISUAL_QA', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Owner-requested CCR9 panel addition with the Figure 4A introduction and shifted Extended Figure 4 references; no other manuscript findings revised.',
                'documents': changes, 'script': str(Path(__file__).resolve()), 'script_sha256': sha(__file__)}
    (AUDIT / 'document_changes.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'staged': [x['staged'] for x in changes], 'changed_paragraphs': sum(len(x['changes']) for x in changes)}))


def publish():
    path = AUDIT / 'document_changes.json'
    manifest = json.loads(path.read_text())
    if manifest['status'] != 'STAGED_FOR_VISUAL_QA':
        raise RuntimeError('Documents are not staged')
    for item in manifest['documents']:
        if sha(item['source']) != item['source_sha256'] or sha(item['staged']) != item['staged_sha256']:
            raise RuntimeError('Current or staged document changed before publication')
        if Path(item['destination']).exists() or Path(item['archive']).exists():
            raise FileExistsError('Document destination or archive already exists')
    for item in manifest['documents']:
        archive = Path(item['archive'])
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(item['source'], archive)
        shutil.copy2(item['staged'], item['destination'])
        assert sha(archive) == item['source_sha256'] and sha(item['destination']) == item['staged_sha256']
        assert len(list((ROOT / item['folder']).glob('*.docx'))) == 1
    manifest.update(status='PUBLISHED', published_at_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Published three clean documents; previous versions preserved in their existing archives.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    publish() if args.publish else prepare()
