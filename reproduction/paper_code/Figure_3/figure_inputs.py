"""Shared, frozen Figure 3 choices for individual components and complete pages."""
from __future__ import annotations

import hashlib
import json
from paper_paths import Path, lock_record, rendering_script

import numpy as np
import pandas as pd
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parent
SATA = Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript')
DEFAULT_ZOOM_RECORD = ROOT / 'zoom_region_lock.json'
DEFAULT_BOUNDED = SATA / 'Figure_3/stored_outputs/complete_figure_bounded_inference_sft_u8_v2'
DEFAULT_PAIRED = SATA / 'Figure_3_Extended/stored_outputs/paired_view_repair_v1'
PAIRED_GEOMETRY = 'Clean variant-0 teachers; shared variant-1 (90-degree rotation) students with frozen shared intensity augmentation'
CHECKPOINT_SHA = 'd0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137'
SFT_INPUT_PROTOCOL = 'direct_pyramid_crop_normalize_resize_uint8_v1'


def locked_input(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    lock = json.loads((ROOT / 'assembly/input_lock.json').read_text())['inputs']
    expected = lock_record(lock, path).get('sha256')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    if expected is None or digest.hexdigest() != expected:
        raise RuntimeError(f'Unlisted or changed Figure 3 frozen input: {path}')
    return path


def locked_zoom(path: str | Path = DEFAULT_ZOOM_RECORD) -> tuple[float, ...]:
    """Read the accepted zoom; never perform candidate selection at render time."""
    if Path(path).resolve() != DEFAULT_ZOOM_RECORD.resolve():
        raise RuntimeError('Use the shared accepted Figure 3 zoom-region lock')
    record = json.loads(Path(path).read_text())
    if record['source_id'] != 'sample_008' or record['cells_per_full_map'] != 50000:
        raise RuntimeError('Zoom-region source membership changed')
    bounds = tuple(map(float, record['zoom_limits_fullres_px']))
    if len(bounds) != 4 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise RuntimeError('Invalid frozen zoom bounds')
    return bounds


def paired_views(manifest_path: str | Path, views_path: str | Path, seed: int = 7):
    """Render the actual frozen dataset pair, with illustrative student masks.

    Teachers retain prepared variant 0. Both student scales use variant 1 and
    the same saved intensity augmentation. The 102 black patches illustrate
    token masking; training replaces randomly masked tokens, not input pixels.
    """
    manifest = json.loads(locked_input(manifest_path).read_text())
    expected = {'center_id': 'if:cont1:nucleus:352082',
                'local_prepared_index': 2352, 'context_prepared_index': 5035,
                'student_variant': 1, 'teacher_variant': 0, 'mask_fraction': .4,
                'same_geometry_and_intensity_parameters_for_both_student_scales': True}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Frozen paired-view identity or augmentation protocol changed')
    view_path = locked_input(views_path)
    if view_path != Path(manifest['views']['path']) or hashlib.sha256(view_path.read_bytes()).hexdigest() != manifest['views']['sha256']:
        raise RuntimeError('Frozen paired-view manifest and arrays disagree')
    if seed != 7:
        raise RuntimeError('The reviewed illustrative student masks use seeds 7 and 8')
    result = {}
    with np.load(view_path) as arrays:
        for branch, branch_seed in (('local', seed), ('context', seed + 1)):
            for role in ('student', 'teacher'):
                source = arrays[f'{branch}_{role}_image']
                if source.shape != (256, 256) or not np.isfinite(source).all() or source.min() < 0 or source.max() > 1:
                    raise RuntimeError(f'Invalid frozen {branch}/{role} image')
                result[f'{role}_{branch}'] = source * 255
            for index in np.random.default_rng(branch_seed).choice(256, 102, replace=False):
                row, column = divmod(int(index), 16)
                result['student_' + branch][row*16:(row+1)*16, column*16:(column+1)*16] = 0
    return result, manifest


def bounded_inputs(provenance_path: str | Path = DEFAULT_BOUNDED / 'provenance.json'):
    provenance = json.loads(locked_input(provenance_path).read_text())
    if provenance['checkpoint_sha256'] != CHECKPOINT_SHA or provenance['row_count'] != 2755 or provenance.get('input_protocol') != SFT_INPUT_PROTOCOL:
        raise RuntimeError('Bounded Cont1 inference lineage changed')
    outputs = {record['role']: locked_input(record['path']) for record in provenance['outputs']}
    table = pd.read_csv(outputs['bounded_coordinate_predictions'])
    labels = np.load(outputs['original_segmentation_crop'])['labels']
    if len(table) != 2755 or table.nucleus_id.nunique() != 2755:
        raise RuntimeError('Bounded Cont1 nucleus membership changed')
    return table, labels, provenance


def bounded_sft_inputs(provenance: dict) -> dict[str, np.ndarray]:
    """Load the exact illustrated cell inputs used by frozen coordinate inference."""
    if provenance.get('input_protocol') != SFT_INPUT_PROTOCOL or provenance.get('illustrated_nucleus_id') != 352082:
        raise RuntimeError('Illustrated supervised input identity or protocol changed')
    records=[record for record in provenance['outputs'] if record['role']=='illustrated_supervised_model_inputs']
    if len(records)!=1:raise RuntimeError('Frozen supervised input arrays missing or ambiguous')
    path=locked_input(records[0]['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=records[0]['sha256']:
        raise RuntimeError('Illustrated supervised input manifest and arrays disagree')
    expected={'local_image':(256,256),'context_image':(256,256),'fine_image':(128,128)}
    with np.load(path) as arrays:
        if set(arrays.files)!=set(expected):raise RuntimeError('Supervised input branch coverage changed')
        result={key:arrays[key].copy() for key in expected}
    if any(array.dtype!=np.uint8 or array.shape!=expected[key] for key,array in result.items()):
        raise RuntimeError('Supervised input shape or stored-byte dtype changed')
    return result


def prediction_overlays(crops, table, labels, nucleus_id: int) -> dict[str, np.ndarray]:
    """Return actual saved predictions as RGBA overlays; no inference or fitting."""
    selected = table.loc[table.nucleus_id.eq(nucleus_id)]
    if len(selected) != 1:
        raise RuntimeError('Illustrated nucleus must occur exactly once in frozen predictions')
    selected = selected.iloc[0]
    target = crops['fine_target_mask'].astype(bool)
    result = {}
    for axis, column in (('cv', 'predicted_axis_coordinate'), ('epi', 'predicted_epithelial_distance')):
        color = colormaps['viridis'](float(selected[column]))
        head = np.zeros((*target.shape, 4))
        head[target] = color
        head[target, 3] = .72
        result['head_' + axis] = head
        lookup = np.zeros((int(labels.max()) + 1, 4), dtype=np.float32)
        lookup[table.nucleus_id.to_numpy(int)] = colormaps['viridis'](table[column].to_numpy(float))
        context = lookup[labels]
        context[..., 3] *= .7
        result['context_' + axis] = context
    return result
