"""Regression tests for opaque generic IDs versus prepared source-cell numbering."""
from paper_paths import Path, lock_record, rendering_script
import tempfile
import unittest
import h5py
import numpy as np
import pandas as pd

from cell_identity import resolve_global_indices, resolve_prepared_overlap


class CellIdentityTests(unittest.TestCase):
    def test_original_finite_order_and_coordinate_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.h5ad'
            with h5py.File(path, 'w') as f:
                f.create_dataset('obs/batch/categories', data=np.array(['day90_SI_r2'], dtype=h5py.string_dtype()))
                f.create_dataset('obs/batch/codes', data=np.zeros(4, dtype=int))
                f.create_dataset('obs/crypt_villi_axis', data=[.1, np.nan, .4, .5])
                f.create_dataset('obs/epithelial_distance', data=[.2, .3, .4, .5])
                f.create_dataset('obsm/X_spatial', data=[[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
            rows = pd.DataFrame({'row_id': ['sample_010_cell_0000001'],
                'centroid_x_fullres_px': [10.], 'centroid_y_fullres_px': [12.]})
            indices, evidence = resolve_global_indices(rows, h5ad=path, batch='day90_SI_r2', pixel_size_um=.5)
            self.assertEqual(indices.tolist(), [2])
            self.assertEqual(evidence['coordinate_max_abs_difference_um'], 0)
            rows.loc[0, 'centroid_x_fullres_px'] = 11
            with self.assertRaises(RuntimeError):
                resolve_global_indices(rows, h5ad=path, batch='day90_SI_r2', pixel_size_um=.5)

    def test_prepared_obs_index_is_never_generic_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({'figure2_global_index': [737409], 'prepared_index': [4489],
                          'split': ['test']}).to_csv(root / 'training.csv', index=False)
            pd.DataFrame({'prepared_index': [4489], 'obs_index': [209641],
                'section_id': ['day90_SI_r2'], 'center_x_fullres_px': [22919.5],
                'center_y_fullres_px': [36055.5]}).to_csv(root / 'prepared.csv', index=False)
            rows = pd.DataFrame({'row_id': ['sample_010_cell_0207503'],
                'centroid_x_fullres_px': [22919.5], 'centroid_y_fullres_px': [36055.5]})
            result = resolve_prepared_overlap(rows, global_indices=np.array([737409]),
                training_manifest=root / 'training.csv', prepared_manifest=root / 'prepared.csv')
            self.assertEqual(result, {0: 4489})


if __name__ == '__main__':
    unittest.main()
