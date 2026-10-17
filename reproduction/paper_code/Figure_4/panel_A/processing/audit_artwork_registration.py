#!/usr/bin/env python3
"""Recover imported microscopy regions by registering source snapshots to artwork."""
from __future__ import annotations

import argparse
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import SimilarityTransform, warp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artwork', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artwork = np.asarray(Image.open(args.artwork).convert('RGB'))
    records = {}
    for condition, filename, box in (
        ('DMSO', 'DMSO.png', (96, 100, 704, 687)),
        ('RARi', 'RAR.png', (714, 101, 1322, 687)),
    ):
        source_path = args.snapshots / filename
        source = np.asarray(Image.open(source_path).convert('RGB'))
        x0, y0, x1, y1 = box
        target = artwork[y0:y1, x0:x1]
        features = []
        for arr in (source, target):
            sift = SIFT()
            sift.detect_and_extract(rgb2gray(arr))
            features.append(sift)
        matches = match_descriptors(features[0].descriptors, features[1].descriptors, cross_check=True, max_ratio=0.8)
        source_xy = features[0].keypoints[matches[:, 0]][:, ::-1]
        target_xy = features[1].keypoints[matches[:, 1]][:, ::-1]
        model, inliers = ransac((target_xy, source_xy), SimilarityTransform, min_samples=3, residual_threshold=3, max_trials=2000, rng=0)
        corners = model(np.array([[0,0], [x1-x0,0], [x1-x0,y1-y0], [0,y1-y0]]))
        aligned = warp(source, inverse_map=model, output_shape=target.shape[:2], preserve_range=True)
        source_rgb = aligned.reshape(-1, 3)
        target_rgb = target.reshape(-1, 3).astype(float)
        coefficients = np.linalg.lstsq(np.c_[source_rgb, np.ones(len(source_rgb))], target_rgb, rcond=None)[0]
        residuals = model.residuals(target_xy[inliers], source_xy[inliers])
        records[condition] = {
            'source': str(source_path), 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'artwork_bbox_xyxy': box, 'target_to_source_matrix': model.params.tolist(),
            'target_corners_in_source_xy': corners.tolist(), 'scale': model.scale,
            'rotation_degrees': float(np.rad2deg(model.rotation)), 'matches': len(matches),
            'inliers': int(inliers.sum()), 'median_residual_source_px': float(np.median(residuals)),
            'diagnostic_rgb_affine_coefficients_only': coefficients.tolist(),
            'pearson_rgb_correlations': [float(np.corrcoef(source_rgb[:,c],target_rgb[:,c])[0,1]) for c in range(3)],
        }
        Image.fromarray(np.clip(aligned,0,255).astype('uint8')).save(args.output_dir / f'{condition}_registered_source.png')
        Image.fromarray(target).save(args.output_dir / f'{condition}_artwork_region.png')
        print(condition, json.dumps(records[condition]), flush=True)
    (args.output_dir / 'registration.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
