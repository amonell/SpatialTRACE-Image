"""One frozen Xenium example shared by Figure 3A component/page rendering."""
from __future__ import annotations

import hashlib
import json
from paper_paths import Path, lock_record, rendering_script

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from Figure_3.figure_inputs import locked_input, CHECKPOINT_SHA, SFT_INPUT_PROTOCOL, SATA

DEFAULT_SUPERVISED = SATA / 'Figure_3/stored_outputs/panel_A/supervised_xenium_v1/manifest.json'


def supervised_example(manifest_path: Path = DEFAULT_SUPERVISED):
    manifest = json.loads(locked_input(manifest_path).read_text())
    if (manifest['checkpoint_sha256'] != CHECKPOINT_SHA or
            manifest['input_protocol'] != SFT_INPUT_PROTOCOL or
            manifest['illustrated_row_id'] != 'sample_010_cell_0214763' or
            manifest['illustrated_prepared_index'] != 4540 or
            manifest['stable_source_id'] != 'xenium:timecourse_replicates:day90_SI_r2' or
            manifest['released_sft_assignment'] != 'test'):
        raise RuntimeError('Approved Xenium example lineage changed')
    paths = {}
    for record in manifest['outputs']:
        path = locked_input(record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError('Frozen Xenium manifest and output hashes disagree')
        paths[path.name] = path
    with np.load(paths['images.npz']) as stored:
        images = {key: stored[key].copy() for key in stored.files}
    shapes = {'local_image': (256, 256), 'context_image': (256, 256),
              'fine_image': (128, 128), 'context_dapi': (3132, 3132)}
    if set(images) != set(shapes) or any(a.shape != shapes[k] or a.dtype != np.uint8 for k, a in images.items()):
        raise RuntimeError('Frozen image branch shapes/bytes changed')
    for record in manifest['tensor_checks']:
        if hashlib.sha256(images[record['branch'] + '_image'].tobytes()).hexdigest() != record['array_sha256']:
            raise RuntimeError('Illustrated input no longer matches the saved inference tensor')
    table = pd.read_csv(paths['predictions.csv'], float_precision='round_trip')
    if len(table) != manifest['row_count'] or table.row_id.nunique() != len(table):
        raise RuntimeError('Frozen regional membership changed')
    selected = table.loc[table.row_id.eq(manifest['illustrated_row_id'])]
    if len(selected) != 1:
        raise RuntimeError('Illustrated cell missing or duplicated')
    xy = table[['centroid_x_fullres_px', 'centroid_y_fullres_px']].to_numpy(float)
    values = table[['predicted_axis_coordinate', 'predicted_epithelial_distance_clipped_1p0']].to_numpy(float)
    left, right, top, bottom = manifest['context_bounds_xyxy']
    if (not np.isfinite(xy).all() or not np.isfinite(values).all() or
            np.any(values < 0) or np.any(values > 1) or
            np.any(xy < [left, top]) or np.any(xy >= [right, bottom])):
        raise RuntimeError('Invalid coordinates or prediction values')
    return {'images': images, 'table': table, 'selected': selected.iloc[0], 'manifest': manifest}


def prediction_background(example, key):
    if key not in ('head_cv', 'head_epi', 'context_cv', 'context_epi'):
        raise ValueError(key)
    return example['images']['fine_image' if key.startswith('head_') else 'context_dapi']


def add_prediction_overlay(axis, example, key):
    """Draw saved per-cell values, using fixed-radius vector centroid markers."""
    background = prediction_background(example, key)
    manifest = example['manifest']
    column = 'predicted_axis_coordinate' if key.endswith('_cv') else 'predicted_epithelial_distance_clipped_1p0'
    display = manifest['display']
    radius = display['centroid_marker_radius_native_px']
    if key.startswith('head_'):
        xy = np.asarray([manifest['fine_marker_xy']])
        radius *= 128 / manifest['native_crop_sizes_px']['fine']
        values = np.asarray([example['selected'][column]])
    else:
        xy = example['table'][['centroid_x_fullres_px', 'centroid_y_fullres_px']].to_numpy(float)
        xy = xy - np.asarray(manifest['context_bounds_xyxy'])[[0, 2]]
        values = example['table'][column].to_numpy(float)
    colors = colormaps[display['colormap']](values)
    artist = PatchCollection([Circle(point, radius) for point in xy], facecolors=colors,
        edgecolors='none', alpha=display['centroid_marker_alpha'], rasterized=False)
    axis.add_collection(artist)
    axis.set_xlim(-.5, background.shape[1] - .5)
    axis.set_ylim(background.shape[0] - .5, -.5)
    return artist
