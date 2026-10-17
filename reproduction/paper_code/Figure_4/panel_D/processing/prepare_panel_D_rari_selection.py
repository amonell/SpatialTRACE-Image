#!/usr/bin/env python3
"""Freeze an approved RARi review field while retaining the original DMSO field."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Figure_4.panel_D.processing.prepare_panel_D_field_candidates import record, verify, save_json


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    choice = json.loads(args.selection.read_text())
    check(choice['status'] == 'AUTHOR_APPROVED', 'Unapproved field selection')
    check(set(choice['selected']) == {'DMSO', 'RARi'}, 'Expected two conditions')
    inputs = [record(args.selection)]
    for key in ('previous_fields', 'review_manifest', 'reviewed_pdf', 'reviewed_cells', 'review_audit'):
        verify(choice[key]); inputs.append(choice[key])
    previous = json.loads(Path(choice['previous_fields']['path']).read_text())
    review = json.loads(Path(choice['review_manifest']['path']).read_text())
    audit = json.loads(Path(choice['review_audit']['path']).read_text())
    check(audit['status'] == 'PASS', 'Review data audit failed')
    check(audit['candidate_manifest']['sha256'] == choice['review_manifest']['sha256'], 'Wrong reviewed manifest')
    verify(review['source_manifest']); inputs.append(review['source_manifest'])
    original = json.loads(Path(review['source_manifest']['path']).read_text())
    for key in ('release', 'channel_manifest', 'channel_key', 'gates', 'checkpoint', 'pixel_size_um', 'display', 'P14_rule'):
        check(previous[key] == original[key], 'Frozen lineage differs: ' + key)
        if isinstance(previous[key], dict) and 'sha256' in previous[key]:
            verify(previous[key]); inputs.append(previous[key])
    check(review['display'] == previous['display'] and review['pixel_size_um'] == previous['pixel_size_um'],
          'Review channel display or calibration differs')
    dmso = deepcopy(previous['windows']['DMSO'])
    check(dmso['selected_review_id'] == choice['selected']['DMSO'], 'DMSO choice changed')
    check(dmso['bounds_xxyy_px'] == choice['bounds_xxyy_px']['DMSO'], 'DMSO bounds changed')
    check(dmso['n'] == choice['expected_cells']['DMSO'], 'DMSO count changed')
    for key in ('cells', 'DAPI', 'E_cadherin', 'source_predictions'):
        verify(dmso[key]); inputs.append(dmso[key])
    selected = [w for w in review['windows'] if w['id'] == choice['selected']['RARi']]
    check(len(selected) == 1, 'Reviewed field ID is not unique')
    selected = selected[0]
    check(selected['condition'] == 'RARi' and selected['bounds_xxyy_px'] == choice['bounds_xxyy_px']['RARi'],
          'Reviewed field differs from approval')
    source = review['predictions']['RARi']
    check(source == previous['windows']['RARi']['source_predictions'], 'Prediction source changed')
    full = pd.read_parquet(verify(source)); inputs.append(source)
    x0, x1, y0, y1 = selected['bounds_xxyy_px']
    check((x1-x0, y1-y0) == (6000, 4500), 'Unexpected field size')
    cells = full.loc[full.centroid_x_fullres_px.ge(x0) & full.centroid_x_fullres_px.lt(x1) &
                     full.centroid_y_fullres_px.ge(y0) & full.centroid_y_fullres_px.lt(y1)].copy()
    reviewed = pd.read_parquet(choice['reviewed_cells']['path'])
    pd.testing.assert_frame_equal(cells.reset_index(drop=True), reviewed.reset_index(drop=True), check_exact=True)
    check(len(cells) == choice['expected_cells']['RARi'] and not cells.duplicated(['source_id', 'object_id']).any(),
          'Unexpected selected cells')
    cells['is_P14'] = cells['class'].eq('Transferred')
    inputs.append(selected['microscopy'])
    with np.load(verify(selected['microscopy'])) as raw:
        for name in ('dapi', 'ecad'):
            check(raw[name].shape == (4500, 6000) and raw[name].dtype == np.uint16, 'Invalid native crop')
        args.output_dir.mkdir(parents=True)
        target = args.output_dir/'RARi_cells.parquet'
        cells.to_parquet(target, index=False)
        rari = {'bounds_xxyy_px': selected['bounds_xxyy_px'], 'selected_review_id': selected['id'],
                'source_predictions': source, 'cells': record(target), 'n': len(cells),
                'P14_n': int(cells.is_P14.sum()),
                'gate_counts': {str(k): int(v) for k, v in cells.coordinate_gate.value_counts().items()}}
        outputs = [rari['cells']]
        for name, output_name in (('dapi', 'DAPI'), ('ecad', 'E_cadherin')):
            target = args.output_dir/f'RARi_{output_name}.npy'
            np.save(target, raw[name])
            rari[output_name] = record(target); outputs.append(rari[output_name])
    result = {key: previous[key] for key in ('release', 'channel_manifest', 'channel_key', 'gates',
                                            'checkpoint', 'pixel_size_um', 'display', 'P14_rule')}
    result.update(status='FROZEN_REGISTERED_GATE_BACKPROJECTION',
        created_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        python=platform.python_version(), packages={n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'pyarrow')},
        git_state='Project is not a Git repository', random_seed=None,
        code=[record(__file__)], inputs=inputs, outputs=outputs, selection=choice,
        selection_record=record(args.selection), windows={'DMSO': dmso, 'RARi': rari},
        gate_assignments_changed=False, boundaries_changed=False, separate_condition_boundaries=False,
        same_bounds_in_main_and_extended=True, new_field_selection=True, model_inference_run=False,
        retained_DMSO_byte_exact=True, reviewed_RARi_cell_values_exact=True,
        scope=choice['selection_scope'], warnings=previous['warnings'])
    save_json(args.output_dir/'manifest.json', result)
    print(json.dumps(record(args.output_dir/'manifest.json'), indent=2))


if __name__ == '__main__':
    main()
