#!/usr/bin/env python3
"""Read-only, fixed-cell comparison of raw crop protocols to production SFT shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from paper_paths import Path, lock_record, rendering_script
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--region-lock', type=Path, required=True)
    parser.add_argument('--pipeline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print('importing numeric libraries', flush=True)
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print('importing original prepared-crop backend', flush=True)
    sys.path.insert(0, str(args.pipeline_root / 'src'))
    from if_processing_pipeline import spatial_if_multiscale_dapi as crops
    contract = json.loads(args.contract.read_text())
    metadata = json.loads(Path(contract['prepared_metadata']['path']).read_text())
    manifest = pd.read_csv(metadata['manifest_path'])
    region = json.loads(args.region_lock.read_text())
    section = region['source_section']
    chosen = manifest.loc[manifest.prepared_index.isin([4489, 4492, 4502])]
    assert len(chosen) == 3
    assert set(chosen.source_id) == {'xenium:timecourse_replicates:day90_SI_r2'}
    print(chosen[['prepared_index', 'source_id', 'center_x_fullres_px', 'center_y_fullres_px']].to_dict('records'), flush=True)
    source = crops.DapiSourceSpec(source_id='xenium:timecourse_replicates:day90_SI_r2', source_group='xenium',
        section_id='day90_SI_r2', sample_family='day90_SI_r2', condition='day90',
        image_kind='ome_tiff', image_path=Path(section['dapi_path']), sidecar_dir=None,
        nuclei_mask_path=None, label_path=None, metadata_path=None,
        pixel_size_um=section['pixel_size_um'], holdout=True)
    config = crops.IfXeniumMultiscaleVitConfig(vit_input_size_px=256, fine_if_crop_fullres_px=128,
        fine_model_input_size_px=128, prepared_supervised_augmentation_variant_count=1)
    backend = crops.DapiCropManager([source], config=config).backend(source.source_id)
    print('backend ready', backend.full_shape_yx, backend.level_shapes_yx, flush=True)
    records = []
    for row in chosen.itertuples():
        idx = int(row.prepared_index)
        for branch in ('local', 'context', 'fine'):
            native = contract['native_crop_sizes_px'][branch]
            size = 128 if branch == 'fine' else 256
            shard = Path(metadata[f'{branch}_shard_paths'][idx // metadata['rows_per_shard']])
            target = np.asarray(np.load(shard, mmap_mode='r')[idx % metadata['rows_per_shard'], 0])
            kwargs = dict(center_x_fullres_px=float(row.center_x_fullres_px),
                center_y_fullres_px=float(row.center_y_fullres_px), crop_fullres_px=native)
            print('loading', idx, branch, native, size, flush=True)
            direct = backend.load_model_input(**kwargs, output_size_px=size)[0]
            direct_u8 = np.rint(np.clip(direct, 0, 1) * 255).astype(np.uint8)
            def compare(candidate):
                delta = candidate.astype(np.float64) - target
                return {'exact': bool(np.array_equal(candidate, target)),
                        'mae_uint8': float(np.abs(delta).mean()),
                        'max_abs_uint8': float(np.abs(delta).max()),
                        'array_sha256': digest(candidate)}
            record = {'prepared_index': idx, 'source_id': row.source_id, 'branch': branch,
                'native_crop_fullres_px': native, 'output_size_px': size,
                'center_x_fullres_px': row.center_x_fullres_px, 'center_y_fullres_px': row.center_y_fullres_px,
                'frozen_shard': str(shard), 'frozen_array_sha256': digest(target),
                'pyramid_level': backend.choose_level(crop_fullres_px=native, output_size_px=size),
                'direct_normalized_quantized': compare(direct_u8)}
            if branch != 'fine':
                parent = backend.load_model_input(**kwargs, output_size_px=512)[0]
                parent_u8 = np.rint(np.clip(parent, 0, 1) * 255).astype(np.uint8)
                resized = F.interpolate(torch.from_numpy(parent_u8.copy()).float()[None, None],
                    size=(256, 256), mode='bilinear', align_corners=False).round().clamp(0, 255).to(torch.uint8).numpy()[0, 0]
                record['via_512_quantized_then_256_quantized'] = compare(resized)
            records.append(record)
            print(json.dumps(record), flush=True)
    result = {'selection': 'First three fixed prepared test rows in the unchanged Figure1D window; no prediction-based selection',
              'records': records,
              'all_direct_exact': all(r['direct_normalized_quantized']['exact'] for r in records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE', args.output, 'all_direct_exact', result['all_direct_exact'], flush=True)


if __name__ == '__main__':
    main()
