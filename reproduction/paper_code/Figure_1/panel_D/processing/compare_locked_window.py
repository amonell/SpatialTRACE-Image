#!/usr/bin/env python3
"""Compare old and corrected bounded outputs against identity-joined graph references."""
from __future__ import annotations
import argparse
import json
from paper_paths import Path, lock_record, rendering_script
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from common import file_record, sha256, write_provenance
from cell_identity import resolve_global_indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old-predictions', 'new-predictions', 'h5ad', 'registry', 'region-lock', 'dapi', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    old = pd.read_csv(args.old_predictions)
    new = pd.read_csv(args.new_predictions)
    for column in ('row_id', 'centroid_x_fullres_px', 'centroid_y_fullres_px'):
        if not old[column].equals(new[column]):
            raise RuntimeError(f'Old/new fixed rows changed: {column}')
    if len(new) != 4854 or new.row_id.duplicated().any():
        raise RuntimeError('Locked bounded cell identities changed')
    region = json.loads(args.region_lock.read_text())
    batch = region['source_section']['batch']
    px = region['source_section']['pixel_size_um']
    indices, identity = resolve_global_indices(new, h5ad=args.h5ad, batch=batch, pixel_size_um=px)
    h5ad_rows = identity['h5ad_rows']
    max_delta = identity['coordinate_max_abs_difference_um']
    registry = json.loads(args.registry.read_text())
    graph_inputs = registry['models']['graph_coordinate_all_label_visualization']['outputs']
    references = {}
    for branch, role in [('crypt_villus', 'crypt_villus_whole_dataset_predictions'),
                         ('epithelial_distance', 'epithelial_distance_whole_dataset_predictions')]:
        record = next(r for r in graph_inputs if r['role'] == role)
        if sha256(record['path']) != record['sha256']:
            raise RuntimeError('Frozen graph reference hash mismatch')
        values = np.load(record['path'], mmap_mode='r').reshape(-1)
        if len(values) != h5ad_rows:
            raise RuntimeError('Graph predictions and h5ad index length differ')
        references[branch] = np.asarray(values[indices])
    joined = new.copy()
    joined['figure2_global_index'] = indices
    joined['reference_crypt_villus'] = references['crypt_villus']
    joined['reference_epithelial_distance'] = references['epithelial_distance']
    metric_rows = []
    for branch, column in [('crypt_villus', 'predicted_axis_coordinate'),
                           ('epithelial_distance', 'predicted_epithelial_distance_clipped_1p0')]:
        target = references[branch]
        for label, frame in [('old_wrong_FOV_raw_float', old), ('corrected_training_input_parity', new)]:
            pred = frame[column].to_numpy()
            metric_rows.append({'axis': branch, 'inference': label, 'n': len(pred),
                'pearson_r': float(pearsonr(target, pred).statistic),
                'mae': float(mean_absolute_error(target, pred)), 'r2': float(r2_score(target, pred))})
        joined['old_' + column] = old[column].to_numpy()
    out.mkdir(parents=True)
    joined.to_parquet(out / 'identity_joined_comparison.parquet', index=False)
    summary = {'scope': 'Diagnostic agreement with dense transcriptomic/spatial graph references on the prespecified qualitative window; not manual-label accuracy and not a new untouched test.',
        'cell_count': len(new), 'section': batch, 'coordinate_max_abs_difference_um': max_delta,
        'join': identity['join'],
        'metrics': metric_rows}
    (out / 'diagnostic_metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    x0, x1, y0, y1 = region['regions']['unseen_image_example']['pixel_window_xyxy']
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.1), layout='constrained')
    image = np.load(args.dapi)
    lo, hi = np.percentile(image[image > 0], [.5, 99.8])
    axes[0].imshow(image, cmap='gray', vmin=lo, vmax=hi, extent=(x0, x1, y1, y0))
    axes[0].set_title('Unchanged DAPI window')
    for ax, values, title in zip(axes[1:], [references['crypt_villus'], old.predicted_axis_coordinate, new.predicted_axis_coordinate],
            ['Dense graph reference\n(original figure used this model)', 'Released ViT: previous crop pipeline', 'Same ViT: exact SFT crop pipeline']):
        ax.scatter(new.centroid_x_fullres_px, new.centroid_y_fullres_px, c=values,
                   cmap='viridis', vmin=0, vmax=1, s=1.2, linewidths=0, rasterized=True)
        ax.set_title(title)
    for ax in axes:
        ax.set(xlim=(x0, x1), ylim=(y1, y0), aspect='equal')
        ax.axis('off')
    fig.savefig(out / 'same_window_comparison.pdf', dpi=300)
    fig.savefig(out / 'same_window_comparison.png', dpi=300)
    plt.close(fig)
    inputs = [file_record(path, name) for name, path in vars(args).items() if name not in ('output_dir', 'h5ad')]
    inputs += graph_inputs
    stat = args.h5ad.stat()
    inputs.append({'path': str(args.h5ad), 'role': 'immutable_h5ad_index_identity', 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
        'hash_note': 'Large immutable H5AD; section identities and all plotted cell coordinates verified explicitly.'})
    write_provenance(out / 'provenance.json', script=HERE, inputs=inputs,
        outputs=[out / n for n in ('identity_joined_comparison.parquet', 'diagnostic_metrics.json', 'same_window_comparison.pdf', 'same_window_comparison.png')],
        extra=summary, overwritten=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
