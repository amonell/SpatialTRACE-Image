"""Guard the main-figure preprocessing correction and its rendering boundary."""
import json
from paper_paths import Path, lock_record, rendering_script
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from Figure_3.common import data_release, repaired_reference, DEFAULT_DATA_RELEASE, CODE_ROOT


class TrainingMatchedReleaseTests(unittest.TestCase):
    def test_release_preserves_scientific_choices(self):
        release=data_release()
        self.assertEqual(release['source_id'],'sample_008')
        self.assertEqual((release['cells'],release['umap_cells']),(50000,30000))
        for flag in ['cell_membership_changed','embedding_membership_changed','reference_values_changed','zoom_changed','model_refitted']:
            self.assertIs(release[flag],False)
        self.assertEqual(release['umap_parameters'],{'n_neighbors':30,'min_dist':.15,'metric':'cosine','seed':42})

    def test_stale_raw_release_is_refused(self):
        old=DEFAULT_DATA_RELEASE.parent.parent/'reference_repair_v1/manifest.json'
        with self.assertRaises(RuntimeError):data_release(old)
        with self.assertRaises(RuntimeError):
            repaired_reference(old.parent/'predictions_current_references.parquet',DEFAULT_DATA_RELEASE,'predictions_current_references.parquet')

    def test_golden_inputs_and_predictions_match(self):
        root=DEFAULT_DATA_RELEASE.parent
        parity=json.loads((root/'crop_parity.json').read_text())
        self.assertEqual(len(parity['checks']),1002)
        self.assertTrue(all(c['exact'] for c in parity['checks']))
        inference=json.loads((root/'inference_manifest.json').read_text())
        self.assertEqual(inference['golden_prediction_cells'],334)
        self.assertLess(inference['golden_prediction_max_abs_difference'],1e-5)

    def test_UMAP_rows_match_prediction_references(self):
        release=data_release();root=DEFAULT_DATA_RELEASE.parent
        umap=pd.read_csv(repaired_reference(root/'figure3f_umap.csv',DEFAULT_DATA_RELEASE,'figure3f_umap.csv'))
        metadata=pd.read_csv(root/'embedding_metadata_current_references.csv')
        frame=pd.read_parquet(root/'predictions_current_references.parquet').set_index('row_id')
        self.assertEqual(umap.shape,(30000,2));self.assertEqual(metadata.row_id.nunique(),30000)
        columns=['target_axis','epithelial_distance_clipped_1p0','predicted_axis_coordinate','predicted_epithelial_distance_clipped_1p0']
        np.testing.assert_allclose(metadata[columns],frame.loc[metadata.row_id,columns],rtol=0,atol=1e-7)
        self.assertEqual(np.load(root/'figure3f_embeddings.npz')['embeddings'].shape,(30000,256))

    def test_component_commands_use_current_frozen_inputs(self):
        result=subprocess.run([sys.executable,'-B',str(CODE_ROOT/'Figure_3/run_all_panels.py'),
            '--output-root',str(DEFAULT_DATA_RELEASE.parents[2]),'--panels','B','C','D','E','--dry-run'],check=True,capture_output=True,text=True)
        self.assertNotIn('reference_repair_v1',result.stdout)
        self.assertEqual(result.stdout.count('--reference-manifest '+str(DEFAULT_DATA_RELEASE)),4)
        self.assertIn(str(DEFAULT_DATA_RELEASE.parent/'figure3f_umap.csv'),result.stdout)

    def test_rendering_performs_no_inference_or_UMAP_fit(self):
        for path in [CODE_ROOT/'Figure_3/assembly/render.py',*[next((CODE_ROOT/f'Figure_3/panel_{letter}/figure_script').glob('*.py')) for letter in 'BCDE']]:
            text=path.read_text()
            for token in ['fit_transform(', 'predict_cells(', 'load_model_checkpoint(', '.backward(']:self.assertNotIn(token,text)

    def test_page_QA_uses_all_current_metric_labels(self):
        from Figure_3.assembly.prepare import metric_label_expectations
        metrics=json.loads((DEFAULT_DATA_RELEASE.parent/'metrics.json').read_text())
        labels=metric_label_expectations(metrics)
        self.assertEqual(len(labels),6)
        self.assertIn('Pearson r = 0.949',labels)
        self.assertIn('Pearson r = 0.841',labels)
        self.assertIn('MAE = 0.060',labels)
        self.assertIn('MAE = 0.102',labels)
        self.assertIn('R² = 0.892',labels)
        self.assertIn('R² = 0.695',labels)
        layout=json.loads((CODE_ROOT/'Figure_3/assembly/layout.json').read_text())
        self.assertNotIn('0.947',layout['expected_text'])
        self.assertNotIn('0.839',layout['expected_text'])


if __name__=='__main__':unittest.main()
