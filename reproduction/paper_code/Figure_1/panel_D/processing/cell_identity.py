"""Reconstruct original manifest IDs and independently verify every spatial coordinate."""
from __future__ import annotations
from paper_paths import Path, lock_record, rendering_script
import h5py
import numpy as np
import pandas as pd


def resolve_global_indices(rows, *, h5ad: Path, batch: str, pixel_size_um: float):
    """Use the documented xenium_manifest.py row_id construction, never prepared obs_index."""
    with h5py.File(h5ad, 'r') as handle:
        categories = handle['obs/batch/categories'].asstr()[:]
        code = np.flatnonzero(categories == batch).item()
        section_indices = np.flatnonzero(handle['obs/batch/codes'][:] == code)
        coordinates = np.asarray(handle['obsm/X_spatial'][section_indices, :2])
        axis = np.asarray(handle['obs/crypt_villi_axis'][section_indices])
        epithelial = np.asarray(handle['obs/epithelial_distance'][section_indices])
        finite = np.isfinite(coordinates).all(1) & np.isfinite(axis) & np.isfinite(epithelial)
        # Exact original all_indices/row_ids construction in xenium_manifest.py.
        # Prepared obs_index refers to different source numbering.
        manifest_indices = section_indices[finite]
        ordinals = rows.row_id.str.rsplit('_cell_', n=1).str[-1].astype(int).to_numpy()
        indices = manifest_indices[ordinals]
        expected = np.asarray(handle['obsm/X_spatial'][indices, :2])
        n_rows = int(handle['obsm/X_spatial'].shape[0])
    actual = rows[['centroid_x_fullres_px', 'centroid_y_fullres_px']].to_numpy() * pixel_size_um
    delta = float(np.abs(actual - expected).max())
    if delta > 1e-6:
        raise RuntimeError(f'Original manifest ID reconstruction fails spatial validation: {delta} micrometers')
    return indices, {'h5ad_rows': n_rows, 'source_section_rows': len(section_indices),
        'finite_source_rows': len(manifest_indices), 'matched_rows': len(indices),
        'coordinate_max_abs_difference_um': delta,
        'join': 'Reconstruct original anonymous row IDs from finite h5ad section-row order using the frozen manifest generator; independently verify every centroid. Prepared obs_index is a different numbering and is not used.'}


def resolve_prepared_overlap(rows, *, global_indices, training_manifest, prepared_manifest):
    training = pd.read_csv(training_manifest)
    prepared = pd.read_csv(prepared_manifest).set_index('prepared_index')
    training = training.loc[training.figure2_global_index.isin(global_indices)]
    if training.figure2_global_index.duplicated().any():
        raise RuntimeError('Duplicate prepared-to-global identity mapping')
    by_global = training.set_index('figure2_global_index')
    golden = {}
    for i, global_index in enumerate(global_indices):
        if global_index not in by_global.index:
            continue
        match = by_global.loc[global_index]
        original = prepared.loc[int(match.prepared_index)]
        actual = rows.iloc[i][['centroid_x_fullres_px', 'centroid_y_fullres_px']].to_numpy(dtype=float)
        expected = original[['center_x_fullres_px', 'center_y_fullres_px']].to_numpy(dtype=float)
        if np.abs(actual - expected).max() > 1e-6:
            raise RuntimeError(f'Original prepared center does not verify the frozen global-index link: {global_index}')
        if match.split != 'test' or original.section_id != 'day90_SI_r2':
            raise RuntimeError('Unexpected production split or source section in overlap')
        golden[i] = int(match.prepared_index)
    if not golden:
        raise RuntimeError('No independently verified golden prepared cells')
    return golden
