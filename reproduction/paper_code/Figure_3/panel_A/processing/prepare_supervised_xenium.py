#!/usr/bin/env python3
"""Freeze Figure 3A's approved Xenium example from verified inputs/predictions.

This stage subsets saved predictions and extracts one DAPI field. It performs
no model inference, fitting, checkpoint selection or prediction-based selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Figure_3.common import file_record, require_file, write_provenance

CHECKPOINT_SHA = 'd0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137'
PROTOCOL = 'direct_pyramid_crop_normalize_resize_uint8_v1'


def choose_center(frame, parity, window):
    """Geometry-only choice, independently reproduced before reading scores."""
    ids = {r['row_id']: r['prepared_index'] for r in parity['tensor_checks']}
    eligible = frame.loc[frame.row_id.isin(ids),
        ['row_id', 'centroid_x_fullres_px', 'centroid_y_fullres_px']].copy()
    cx, cy = (window[0] + window[1]) / 2, (window[2] + window[3]) / 2
    eligible['distance_squared'] = ((eligible.centroid_x_fullres_px - cx)**2 +
                                    (eligible.centroid_y_fullres_px - cy)**2)
    center = eligible.sort_values(['distance_squared', 'row_id']).iloc[0]
    return center, ids[center.row_id]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-lock', type=Path, required=True)
    parser.add_argument('--pipeline-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    declared = [out / n for n in ('images.npz', 'predictions.csv', 'manifest.json')]
    if any(p.exists() for p in declared):
        # Preserve immutable snapshots; re-extraction belongs in a new directory.
        raise FileExistsError('Frozen snapshot exists; use a fresh --output-dir (even with --overwrite).')
    source_lock = args.source_lock.resolve()
    lock = json.loads(source_lock.read_text())
    inputs = [file_record(source_lock, 'approved_supervised_source_lock')]
    def checked(record, role):
        path = require_file(record['path'], record['sha256'])
        inputs.append(file_record(path, role))
        return path
    source_prov = json.loads(checked(lock['prediction_provenance'], 'source_prediction_provenance').read_text())
    parity = json.loads(checked(lock['training_parity'], 'golden_training_parity').read_text())
    contract = json.loads(checked(lock['crop_contract'], 'verified_crop_contract').read_text())
    region = json.loads(checked(lock['region_lock'], 'previously_locked_region').read_text())
    if source_prov['checkpoint_sha256'] != CHECKPOINT_SHA or not source_prov['golden_tensors_all_exact']:
        raise RuntimeError('Verified released prediction lineage changed')
    rec = next(r for r in source_prov['outputs'] if Path(r['path']).name == 'predictions.csv')
    frame = pd.read_csv(checked(rec, 'frozen_predictions'), float_precision='round_trip')
    window = region['regions']['unseen_image_example']['pixel_window_xyxy']
    center, prepared_index = choose_center(frame, parity, window)
    if center.row_id != lock['illustrated_row_id'] or prepared_index != lock['illustrated_prepared_index']:
        raise RuntimeError('Geometry-only center no longer matches the approved source lock')
    cx, cy = float(center.centroid_x_fullres_px), float(center.centroid_y_fullres_px)
    metadata = json.loads(checked(contract['prepared_metadata'], 'prepared_metadata').read_text())
    prepared = pd.read_csv(checked(contract['prepared_manifest'], 'prepared_manifest'))
    training = pd.read_csv(checked(contract['supervised_manifest'], 'released_supervised_manifest'))
    prepared_row = prepared.loc[prepared.prepared_index.eq(prepared_index)].iloc[0]
    training_row = training.loc[training.prepared_index.eq(prepared_index)].iloc[0]
    if prepared_row.source_id != lock['stable_source_id'] or training_row['split'] != 'test':
        raise RuntimeError('Source or released SFT assignment changed')
    np.testing.assert_allclose([cx, cy], [prepared_row.center_x_fullres_px, prepared_row.center_y_fullres_px], atol=1e-6, rtol=0)
    target_row = frame.loc[frame.row_id.eq(center.row_id)].iloc[0]
    if int(training_row.figure2_global_index) != int(target_row.figure2_global_index):
        raise RuntimeError('Prepared and prediction cell identity disagree')
    print('Locked geometry-only center', center.row_id, prepared_index, cx, cy, flush=True)
    images, tensor_checks = {}, []
    for branch, size in [('local', 256), ('context', 256), ('fine', 128)]:
        path = Path(metadata[branch + '_shard_paths'][prepared_index // metadata['rows_per_shard']])
        record = next(r for r in contract['golden_shards'] if r['path'] == str(path))
        checked(record, 'original_supervised_' + branch + '_shard')
        array = np.load(path, mmap_mode='r')[prepared_index % metadata['rows_per_shard'], 0].copy()
        golden = next(r for r in parity['tensor_checks'] if r['row_id'] == center.row_id and r['branch'] == branch)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if array.dtype != np.uint8 or array.shape != (size, size) or digest != golden['array_sha256']:
            raise RuntimeError('Saved inference input differs from golden supervised tensor')
        images[branch + '_image'] = array
        tensor_checks.append({'branch': branch, 'prepared_index': prepared_index, 'array_sha256': digest, 'exact': True})
    # Raw microscopy conversion is separate from inference. All predictions above
    # were already frozen after verification against these exact training bytes.
    checked(contract['raw_dapi'], 'raw_DAPI')
    checked(contract['prepared_pipeline_source'], 'original_DAPI_backend')
    sys.path.insert(0, str(args.pipeline_root / 'src'))
    from if_processing_pipeline import spatial_if_multiscale_dapi as crops
    section = region['source_section']
    source = crops.DapiSourceSpec(source_id=lock['stable_source_id'], source_group='xenium',
        section_id=section['batch'], sample_family=section['batch'], condition='day90',
        image_kind='ome_tiff', image_path=Path(section['dapi_path']), sidecar_dir=None,
        nuclei_mask_path=None, label_path=None, metadata_path=None,
        pixel_size_um=section['pixel_size_um'], holdout=True)
    cfg = crops.IfXeniumMultiscaleVitConfig(vit_input_size_px=256, fine_if_crop_fullres_px=128,
        fine_model_input_size_px=128, prepared_supervised_augmentation_variant_count=1)
    backend = crops.DapiCropManager([source], config=cfg).backend(source.source_id)
    width = lock['native_context_width_px']
    top, bottom, left, right = crops._crop_bounds_from_center(cy, cx, width, width)
    if left < window[0] or right > window[1] or top < window[2] or bottom > window[3]:
        raise RuntimeError('Illustrated field exceeds the verified prediction corpus')
    print('Extracting fixed native DAPI field', [left, right, top, bottom], flush=True)
    native = backend.load_native_crop(center_x_fullres_px=cx, center_y_fullres_px=cy, crop_fullres_px=width)
    images['context_dapi'] = np.rint(np.clip(native, 0, 1) * 255).astype(np.uint8)
    selected = frame.loc[frame.centroid_x_fullres_px.ge(left) & frame.centroid_x_fullres_px.lt(right) &
                         frame.centroid_y_fullres_px.ge(top) & frame.centroid_y_fullres_px.lt(bottom),
        ['row_id', 'source_id', 'figure2_global_index', 'centroid_x_fullres_px', 'centroid_y_fullres_px',
         'predicted_axis_coordinate', 'predicted_epithelial_distance_clipped_1p0']].copy()
    if selected.row_id.duplicated().any() or selected.empty:
        raise RuntimeError('Invalid bounded prediction membership')
    fine_width = contract['native_crop_sizes_px']['fine']
    ftop, _, fleft, _ = crops._crop_bounds_from_center(cy, cx, fine_width, fine_width)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'images.npz', **images)
    selected.to_csv(out / 'predictions.csv', index=False)
    write_provenance(out / 'manifest.json', script=Path(__file__), inputs=inputs,
        outputs=declared[:2], overwritten=False, extra={
            'operation': 'Subset existing verified predictions and extract one fixed microscopy field; no inference or fitting',
            'source_id': lock['source_id'], 'stable_source_id': lock['stable_source_id'],
            'illustrated_row_id': center.row_id, 'illustrated_prepared_index': int(prepared_index),
            'illustrated_global_index': int(target_row.figure2_global_index),
            'center_fullres_xy_px': [cx, cy], 'context_bounds_xyxy': [left, right, top, bottom],
            'fine_marker_xy': [(cx - fleft + .5) * 128 / fine_width - .5,
                               (cy - ftop + .5) * 128 / fine_width - .5],
            'native_crop_sizes_px': contract['native_crop_sizes_px'], 'pixel_size_um': section['pixel_size_um'],
            'checkpoint_sha256': CHECKPOINT_SHA, 'input_protocol': PROTOCOL,
            'row_count': len(selected), 'selection_rule': lock['selection_rule'],
            'evaluation_scope': lock['evaluation_scope'], 'released_sft_assignment': 'test',
            'source_prediction_split_column': 'Dropped: generic raw-manifest split is not the released SFT assignment',
            'tensor_checks': tensor_checks, 'display': lock['display'],
            'random_seed': None, 'warnings': ['Centroid dots are display markers, not segmentation boundaries.',
                'Reuse of a previously examined qualitative region is not a new performance estimate.']})
    print('Frozen Xenium example', out, 'cells', len(selected), flush=True)


if __name__ == '__main__':
    main()
