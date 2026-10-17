#!/usr/bin/env python3
"""Explicit bounded inference for the two exact Cont1 Figure 3a example slots.

This is not a training or evaluation run. It uses every nucleus centered in the
prespecified 2048-pixel context field, the one released Xenium checkpoint and
the original multiscale VSI crop backend. Results are saved before page drawing.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from paper_paths import Path, lock_record, rendering_script
import shlex
import sys

import numpy as np
import pandas as pd
import torch

from sft_input_protocol import PROTOCOL,quantize_sft_input,dequantize_sft_input

CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript')
PIPELINE = Path('/home/amonell/Desktop/Goldrath_Collaborations/giovanni/2026_04_07_IF_Processing_Pipeline')
MODEL_PACKAGE = Path('/home/amonell/Desktop/crypt-villus-vit')
CHECKPOINT = CODE_ROOT / 'Figure3/train_model/outputs/production_representation_model_v2/best_crypt_villus_vit_model.pt'
CHECKPOINT_SHA = 'd0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137'
SFT_PARITY_SHA = '9cb795ad7a80ff26d8a0e09bf795060ba9f70a198ed20e079423a0502920d01a'
PRETRAIN_PARITY_SHA = '3f102be2f03fc4667152f10121b6acefa354b7c19134adff0bc5884a70c83c13'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path, role):
    return {'path': str(path), 'sha256': sha256(path), 'role': role}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sft-parity',type=Path,default=Path(__file__).with_name('sft_input_parity.json'))
    parser.add_argument('--pretraining-parity',type=Path,default=Path(__file__).with_name('pretraining_input_parity.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    for path,expected in ((args.sft_parity,SFT_PARITY_SHA),(args.pretraining_parity,PRETRAIN_PARITY_SHA)):
        if sha256(path)!=expected:raise ValueError(f'Frozen crop-parity evidence changed: {path}')
    sft_parity=json.loads(args.sft_parity.read_text())
    if not sft_parity['all_direct_exact'] or len(sft_parity['records'])!=9 or not all(r['direct_normalized_quantized']['exact'] for r in sft_parity['records']):
        raise ValueError('Released supervised crop protocol lacks golden-array parity')
    pretraining_parity=json.loads(args.pretraining_parity.read_text())
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Bounded inference exports are immutable; choose a new directory: {output}')
    source_provenance = DATA_ROOT / 'Figure_3/stored_outputs/panel_A/panel_A_exact_historical_region_provenance.json'
    region_metadata = DATA_ROOT / 'Figure_3/stored_outputs/panel_A/panel_A_exact_historical_region.json'
    provenance = json.loads(source_provenance.read_text())
    metadata = json.loads(region_metadata.read_text())
    bounds = metadata['crop_records']['context']['bounds_fullres_px']
    if bounds != {'left': 14685, 'right': 16733, 'top': 21510, 'bottom': 23558}:
        raise ValueError('The prespecified Figure 3a context field changed')
    wanted = {'if_nuclei_table', 'cont1_vsi_header', 'cont1_nuclei_segmentation',
              'cont1_tissue_metadata', 'microscopy_crop_implementation'}
    sources = {record['role']: record for record in provenance['inputs'] if record['role'] in wanted}
    for source in sources.values():
        if sha256(source['path']) != source['sha256']:
            raise ValueError(f"Frozen original microscopy input mismatch: {source['path']}")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Released Xenium checkpoint hash mismatch')
    sys.path.insert(0, str(PIPELINE / 'src'))
    sys.path.insert(0, str(MODEL_PACKAGE / 'src'))
    from if_processing_pipeline import spatial_if_multiscale_dapi as crops
    from crypt_villus_vit.model import load_model_checkpoint
    sidecar = Path(sources['cont1_vsi_header']['path']).parent / '_Image_Cont1_'
    tree = hashlib.sha256()
    sidecar_files = []
    for path in sorted(sidecar.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(sidecar).as_posix().encode()
        tree.update(len(relative).to_bytes(8, 'big')); tree.update(relative)
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b''):
                tree.update(block)
        sidecar_files.append(file_record(path, 'original_vsi_sidecar'))
    expected_tree = next(r['sha256_tree'] for r in provenance['inputs'] if r['role'] == 'cont1_vsi_sidecar_tree')
    if tree.hexdigest() != expected_tree:
        raise ValueError('Original VSI sidecar tree hash mismatch')
    table = pd.read_csv(sources['if_nuclei_table']['path'])
    selected = table.loc[(table.center_x_fullres_px >= bounds['left']) &
                         (table.center_x_fullres_px < bounds['right']) &
                         (table.center_y_fullres_px >= bounds['top']) &
                         (table.center_y_fullres_px < bounds['bottom'])].copy()
    selected = selected.sort_values('object_label_id').reset_index(drop=True)
    if len(selected) != 2755 or selected.object_label_id.nunique() != 2755:
        raise ValueError(f'Expected all 2,755 locked context nuclei, observed {len(selected)}')
    source = crops.DapiSourceSpec(
        source_id='if:cont1', source_group='if', section_id='cont1', sample_family='cont1',
        condition='Cont1', image_kind='vsi', image_path=Path(sources['cont1_vsi_header']['path']),
        sidecar_dir=sidecar, nuclei_mask_path=Path(sources['cont1_nuclei_segmentation']['path']),
        label_path=None, metadata_path=Path(sources['cont1_tissue_metadata']['path']),
        pixel_size_um=metadata['pixel_size_um'], holdout=False)
    backend = crops.DapiCropManager([source], config=crops.IfXeniumMultiscaleVitConfig()).backend('if:cont1')
    # Cache only the union of raw ROIs required by this fixed field. The existing
    # backend still performs each crop's exact bounds, removal mask,
    # normalization and resize. This avoids thousands of repeated VSI reads.
    read_roi = crops.load_bioformats_channel_roi
    cache, cache_records = {}, []
    scales = [('local_image',512,256), ('context_image',2048,256), ('fine_image',128,128)]
    for _, native, size in scales:
        level = backend.choose_level(crop_fullres_px=native, output_size_px=size)
        shape = backend.level_shapes_yx[level]
        fy = shape[0] / backend.full_shape_yx[0]
        fx = shape[1] / backend.full_shape_yx[1]
        height = crops._downsampled_crop_size(native, full_size=backend.full_shape_yx[0], downsampled_size=shape[0])
        width = crops._downsampled_crop_size(native, full_size=backend.full_shape_yx[1], downsampled_size=shape[1])
        boxes = [crops._crop_bounds_from_center(r.center_y_fullres_px*fy, r.center_x_fullres_px*fx, height, width)
                 for r in selected.itertuples()]
        top = max(0, min(b[0] for b in boxes)); bottom = min(shape[0], max(b[1] for b in boxes))
        left = max(0, min(b[2] for b in boxes)); right = min(shape[1], max(b[3] for b in boxes))
        if level in cache:
            raise ValueError('Cache assumes distinct exact model-input pyramid levels')
        stack, roi_metadata = read_roi(backend.dataset, series_index=2, resolution_level=level,
                                      channel_indices=[backend.dapi_index], y=slice(top,bottom), x=slice(left,right))
        cache[level] = (np.asarray(stack), top, left, roi_metadata)
        cache_records.append({'level': level, 'raw_bounds_tblr': [top,bottom,left,right], 'native_crop_px': native,
                              'model_input_px': size, 'raw_array_sha256': hashlib.sha256(np.asarray(stack).tobytes()).hexdigest()})
        print(f'Cached bounded raw VSI level {level}: {stack.shape}', flush=True)
    def cached_roi(dataset, *, series_index, resolution_level, channel_indices, y, x):
        stack, top, left, info = cache[int(resolution_level)]
        if not (top <= y.start < y.stop <= top+stack.shape[1] and left <= x.start < x.stop <= left+stack.shape[2]):
            raise ValueError('Requested microscopy outside the explicitly bounded raw cache')
        return stack[:,y.start-top:y.stop-top,x.start-left:x.stop-left].copy(), info
    # Check cache equality against the unmodified VSI backend on a fixed first
    # nucleus at each scale before replacing its read function.
    first = selected.iloc[0]
    checks = {}
    for key,native,size in scales:
        original = backend.load_model_input(center_y_fullres_px=first.center_y_fullres_px,
                     center_x_fullres_px=first.center_x_fullres_px,crop_fullres_px=native,output_size_px=size)
        crops.load_bioformats_channel_roi = cached_roi
        cached = backend.load_model_input(center_y_fullres_px=first.center_y_fullres_px,
                     center_x_fullres_px=first.center_x_fullres_px,crop_fullres_px=native,output_size_px=size)
        crops.load_bioformats_channel_roi = read_roi
        if not np.array_equal(original,cached):
            raise ValueError(f'Raw-cache crop differs from original backend: {key}')
        checks[key] = {'exact_array_equal': True, 'shape': list(original.shape)}
    crops.load_bioformats_channel_roi = cached_roi
    def model_input(row,native,size):
        raw=backend.load_model_input(center_y_fullres_px=row.center_y_fullres_px,
             center_x_fullres_px=row.center_x_fullres_px,crop_fullres_px=native,output_size_px=size)
        return dequantize_sft_input(quantize_sft_input(raw))
    illustrated=selected.loc[selected.object_label_id.eq(metadata['nucleus_label_id'])]
    if len(illustrated)!=1:raise ValueError('Original illustrated nucleus missing from locked membership')
    illustrated=illustrated.iloc[0]
    illustrated_inputs={key:quantize_sft_input(backend.load_model_input(
        center_y_fullres_px=illustrated.center_y_fullres_px,center_x_fullres_px=illustrated.center_x_fullres_px,
        crop_fullres_px=native,output_size_px=size))[0] for key,native,size in scales}
    frozen_crop_path=source_provenance.parent/'panel_A_exact_historical_region.npz'
    frozen_crop_record=next(r for r in provenance['outputs'] if r['path']==str(frozen_crop_path))
    if sha256(frozen_crop_path)!=frozen_crop_record['sha256']:raise ValueError('Original microscopy crop changed')
    with np.load(frozen_crop_path) as frozen:
        if not np.array_equal(illustrated_inputs['fine_image'],frozen['fine']):raise ValueError('Fine input no longer matches original exact center')
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model, payload = load_model_checkpoint(CHECKPOINT, device=device, strict=True)
    config = asdict(model.config)
    if config['encoder_architecture'] != 'shared_scale_aware' or not config['retain_scale_embeddings']:
        raise ValueError(f'Released shared-scale architecture not loaded: {config}')
    values = []
    torch.set_num_threads(4)
    with torch.inference_mode():
        for start in range(0,len(selected),args.batch_size):
            batch_rows=selected.iloc[start:start+args.batch_size]
            batch={key:torch.from_numpy(np.stack([model_input(r,native,size)
                    for r in batch_rows.itertuples()])).to(device) for key,native,size in scales}
            predicted=model(batch)
            values.append(np.column_stack([predicted['predicted_axis_coordinate'].cpu().numpy(),
                                           predicted['predicted_epithelial_distance_clipped_1p0'].cpu().numpy()]))
            print(f'Bounded inference {min(start+args.batch_size,len(selected))}/{len(selected)}',flush=True)
    predictions=np.concatenate(values)
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite frozen-model inference')
    output.mkdir(parents=True,exist_ok=False)
    exported=pd.DataFrame({'nucleus_id':selected.object_label_id,
        'centroid_x_fullres_px':selected.center_x_fullres_px, 'centroid_y_fullres_px':selected.center_y_fullres_px,
        'predicted_axis_coordinate':predictions[:,0], 'predicted_epithelial_distance':predictions[:,1]})
    exported.to_csv(output/'predictions.csv',index=False)
    labels=np.load(sources['cont1_nuclei_segmentation']['path'],mmap_mode='r')
    np.savez_compressed(output/'region_labels.npz', labels=labels[bounds['top']:bounds['bottom'],bounds['left']:bounds['right']])
    np.savez_compressed(output/'illustrated_sft_inputs.npz',**illustrated_inputs)
    (output/'sft_input_parity.json').write_text(args.sft_parity.read_text())
    (output/'pretraining_input_parity.json').write_text(args.pretraining_parity.read_text())
    inputs=[*sources.values(),*sidecar_files,file_record(CHECKPOINT,'released_xenium_checkpoint'),
            file_record(region_metadata,'locked_context_geometry'),file_record(source_provenance,'original_source_provenance'),
            file_record(MODEL_PACKAGE/'src/crypt_villus_vit/model.py','inference_architecture_source'),
            file_record(Path(__file__).with_name('sft_input_protocol.py'),'supervised_quantization_protocol'),
            file_record(args.sft_parity,'released_supervised_golden_crop_evidence'),
            file_record(args.pretraining_parity,'separate_pretraining_crop_evidence'),
            file_record(frozen_crop_path,'frozen_illustrated_fine_input')]
    summary={'status':'COMPLETE_BOUNDED_INFERENCE','operation':'One frozen checkpoint, original prespecified field; no fitting, tuning, selection or performance evaluation',
        'source_id':'if:cont1','bounds_fullres_px':bounds,'row_count':len(exported),'checkpoint_sha256':CHECKPOINT_SHA,
        'model_config':config,'raw_cache':cache_records,'cache_equivalence_checks':checks,
        'input_protocol':PROTOCOL,'input_protocol_description':'Choose raw pyramid for direct256 local/context or direct128 fine; apply removal mask and per-crop normalization, resize, round to uint8, dequantize by255. No512 intermediate.',
        'golden_supervised_arrays_exact':9,'pretraining_input_distinction':'Cont1 pretraining used512-prepared uint8 crops resized to256; this separate pretraining recipe is preserved in the teacher/student displays and is not the released coordinate SFT input protocol.',
        'illustrated_nucleus_id':int(metadata['nucleus_label_id']),'illustrated_input_shapes':{key:list(value.shape) for key,value in illustrated_inputs.items()},
        'command':shlex.join([sys.executable,*sys.argv]),'timestamp':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'torch':torch.__version__,'inputs':inputs,
        'outputs':[file_record(output/'predictions.csv','bounded_coordinate_predictions'),file_record(output/'region_labels.npz','original_segmentation_crop'),file_record(output/'illustrated_sft_inputs.npz','illustrated_supervised_model_inputs'),file_record(output/'sft_input_parity.json','released_supervised_golden_crop_evidence'),file_record(output/'pretraining_input_parity.json','separate_pretraining_crop_evidence')],
        'script':file_record(Path(__file__),'bounded_inference_code'),
        'limitation':'Illustrative zero-shot coordinate inference on the original IF pretraining example, not an independent performance estimate.'}
    (output/'provenance.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(output,flush=True)


if __name__=='__main__':
    main()
