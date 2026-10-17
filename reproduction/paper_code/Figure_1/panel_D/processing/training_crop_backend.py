"""Exact production-SFT crop pixels, with bounded inference and golden-row checks."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import sys

import numpy as np
import pandas as pd
import torch


def predict_training_protocol(rows, *, section, contract, checkpoint, pipeline_root,
                              output_dir, batch_size, device):
    from crypt_villus_vit.model import load_model_checkpoint
    from crypt_villus_vit.gates import assign_axis_epithelial_gates, summarize_gate_percentages
    sys.path.insert(0, str(Path(pipeline_root) / 'src'))
    from if_processing_pipeline import spatial_if_multiscale_dapi as crops
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from cell_identity import resolve_global_indices, resolve_prepared_overlap
    global_indices, identity = resolve_global_indices(rows, h5ad=Path(contract['h5ad_identity']['path']),
        batch=section['batch'], pixel_size_um=section['pixel_size_um'])
    golden = resolve_prepared_overlap(rows, global_indices=global_indices,
        training_manifest=contract['supervised_manifest']['path'], prepared_manifest=contract['prepared_manifest']['path'])
    print('Verified bounded identities', identity, 'golden overlap', len(golden), flush=True)
    model, payload = load_model_checkpoint(checkpoint, device=device, strict=True)
    model.eval()
    cfg = model.config
    if cfg.encoder_architecture != 'shared_scale_aware' or not cfg.retain_scale_embeddings:
        raise RuntimeError('Released model architecture changed')
    source = crops.DapiSourceSpec(source_id=contract['source_id'], source_group='xenium',
        section_id=section['batch'], sample_family=section['batch'], condition='day90',
        image_kind='ome_tiff', image_path=Path(section['dapi_path']), sidecar_dir=None,
        nuclei_mask_path=None, label_path=None, metadata_path=None,
        pixel_size_um=section['pixel_size_um'], holdout=True)
    prep_cfg = crops.IfXeniumMultiscaleVitConfig(vit_input_size_px=cfg.input_size_px,
        fine_if_crop_fullres_px=128, fine_model_input_size_px=cfg.fine_input_size_px,
        prepared_supervised_augmentation_variant_count=1)
    backend = crops.DapiCropManager([source], config=prep_cfg).backend(source.source_id)
    sizes = {'local': cfg.input_size_px, 'context': cfg.input_size_px, 'fine': cfg.fine_input_size_px}
    levels = {b: backend.choose_level(crop_fullres_px=contract['native_crop_sizes_px'][b],
              output_size_px=sizes[b]) for b in sizes}
    cached = {}
    for level in sorted(set(levels.values())):
        print('Loading original source-pyramid level', level, backend.level_shapes_yx[level], flush=True)
        cached[level] = crops._maybe_load_cached_level_image(backend, resolution_level=level, config=prep_cfg)
        print('Source-pyramid level ready', level, 'cached', cached[level] is not None, flush=True)
    metadata = json.loads(Path(contract['prepared_metadata']['path']).read_text())
    frozen = {}
    golden_records = []
    predictions = np.empty((len(rows), 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            batch = {b + '_image': [] for b in sizes}
            for i in range(start, stop):
                row = rows.iloc[i]
                for branch, size in sizes.items():
                    kwargs = dict(center_x_fullres_px=float(row.centroid_x_fullres_px),
                        center_y_fullres_px=float(row.centroid_y_fullres_px),
                        crop_fullres_px=contract['native_crop_sizes_px'][branch], output_size_px=size)
                    level = levels[branch]
                    if cached[level] is None:
                        image = backend.load_model_input(**kwargs)[0]
                    else:
                        image = crops._cached_level_model_input(cached[level], full_shape_yx=backend.full_shape_yx,
                            level_shape_yx=backend.level_shapes_yx[level], **kwargs)
                    image_u8 = crops._quantize_normalized_image_u8(image)
                    if i in golden:
                        index = golden[i]
                        path = metadata[branch + '_shard_paths'][index // metadata['rows_per_shard']]
                        if path not in {r['path'] for r in contract['golden_shards']}:
                            raise RuntimeError(f'Golden shard was not hash-locked: {path}')
                        if path not in frozen:
                            frozen[path] = np.load(path, mmap_mode='r')
                        target = frozen[path][index % metadata['rows_per_shard'], 0]
                        if not np.array_equal(image_u8, target):
                            raise RuntimeError(f'Golden tensor mismatch row{index} branch{branch}')
                        golden_records.append({'row_id': row.row_id, 'prepared_index': index, 'branch': branch,
                            'frozen_shard': path, 'exact': True,
                            'array_sha256': hashlib.sha256(image_u8.tobytes()).hexdigest()})
                    batch[branch + '_image'].append(image_u8)
            tensors = {key: torch.from_numpy(np.stack(value).astype(np.float32)[:, None] / 255).to(device)
                       for key, value in batch.items()}
            output = model(tensors)
            predictions[start:stop, 0] = output['predicted_axis_coordinate'].cpu().numpy()
            predictions[start:stop, 1] = output['predicted_epithelial_distance_clipped_1p0'].cpu().numpy()
            if start == 0 or stop % (batch_size * 5) == 0 or stop == len(rows):
                print('Prediction progress', stop, '/', len(rows), flush=True)
    if not np.isfinite(predictions).all() or len(golden_records) != 3 * len(golden):
        raise RuntimeError('Missing finite predictions or golden checks')
    development = pd.read_csv(contract['development_predictions']['path']).set_index('prepared_index')
    parity = []
    columns = ['predicted_axis_coordinate', 'predicted_epithelial_distance_clipped_1p0']
    for index, prepared_index in golden.items():
        expected = development.loc[prepared_index, columns].to_numpy(dtype=float)
        delta = np.abs(predictions[index] - expected)
        parity.append({'prepared_index': prepared_index, 'max_abs_difference': float(delta.max())})
    if max(r['max_abs_difference'] for r in parity) > 1e-5:
        raise RuntimeError('Frozen prepared evaluation and corrected raw inference predictions disagree')
    export = rows.reset_index(drop=True).copy()
    export['figure2_global_index'] = global_indices
    for j, column in enumerate(columns):
        export[column] = predictions[:, j]
    export['predicted_bin'] = np.clip(np.floor(predictions[:, 0] * 32), 0, 31).astype(int)
    export['predicted_offset'] = predictions[:, 0] * 32 - export.predicted_bin
    export = assign_axis_epithelial_gates(export)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_dir / 'predictions.csv', index=False)
    summarize_gate_percentages(export, group_columns=('source_id',)).to_csv(output_dir / 'gate_percentages.csv', index=False)
    summary = {'task_type': 'axis_regression', 'checkpoint_path': str(checkpoint),
        'checkpoint_format': payload.get('format'), 'row_count': len(rows), 'source_count': 1,
        'model_config': asdict(cfg), 'reference_pixel_size_um': contract['reference_pixel_size_um'],
        'resolved_crop_sizes_by_source': {contract['source_id']: contract['native_crop_sizes_px']},
        'preprocessing': contract['pixel_protocol'], 'pyramid_levels': levels,
        'device': device, 'batch_size': batch_size, 'golden_prepared_cell_count': len(golden),
        'golden_tensor_count': len(golden_records), 'golden_tensors_all_exact': True,
        'frozen_prediction_max_abs_difference': max(r['max_abs_difference'] for r in parity),
        'evaluation_scope': 'Same locked qualitative window; production SFT section-held-out,25 cells overlap the historical development-test evaluation; not a new untouched test.'}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output_dir / 'training_parity.json').write_text(json.dumps({'tensor_checks': golden_records,
        'prediction_checks': parity, 'source_section': section['batch'],
        'cell_identity_checks': identity,
        'prepared_manifest': contract['prepared_manifest'], 'frozen_shards': contract['golden_shards']}, indent=2) + '\n')
    return summary
