"""Regression checks for the user-approved supervised-side source replacement."""
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd

from Figure_3.supervised_example import supervised_example, DEFAULT_SUPERVISED, add_prediction_overlay, prediction_background
from Figure_3.panel_A.processing.prepare_supervised_xenium import choose_center

ROOT = Path(__file__).resolve().parents[1]


class XeniumExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = supervised_example()
        cls.lock = json.loads((ROOT/'Figure_3/panel_A/processing/supervised_xenium_source.json').read_text())

    def test_geometry_choice_does_not_use_predictions(self):
        provenance = json.loads(Path(self.lock['prediction_provenance']['path']).read_text())
        source = next(r for r in provenance['outputs'] if Path(r['path']).name == 'predictions.csv')
        # Retain only identities and spatial coordinates: no prediction/target access.
        frame = pd.read_csv(source['path'], usecols=['row_id','centroid_x_fullres_px','centroid_y_fullres_px'])
        parity = json.loads(Path(self.lock['training_parity']['path']).read_text())
        center, prepared = choose_center(frame.iloc[::-1], parity, [17548,23863,35764,39652])
        self.assertEqual(center.row_id, self.lock['illustrated_row_id'])
        self.assertEqual(prepared, 4540)

    def test_inputs_match_all_three_original_shard_tensors(self):
        contract = json.loads(Path(self.lock['crop_contract']['path']).read_text())
        metadata = json.loads(Path(contract['prepared_metadata']['path']).read_text())
        for branch in ('local','context','fine'):
            path = metadata[branch+'_shard_paths'][4540//metadata['rows_per_shard']]
            golden = np.load(path, mmap_mode='r')[4540%metadata['rows_per_shard'],0]
            np.testing.assert_array_equal(self.example['images'][branch+'_image'], golden)

    def test_prediction_values_and_membership_unchanged(self):
        provenance = json.loads(Path(self.lock['prediction_provenance']['path']).read_text())
        source = next(r for r in provenance['outputs'] if Path(r['path']).name == 'predictions.csv')
        frame = pd.read_csv(source['path'], float_precision='round_trip')
        left,right,top,bottom = self.example['manifest']['context_bounds_xyxy']
        frame = frame.loc[frame.centroid_x_fullres_px.ge(left)&frame.centroid_x_fullres_px.lt(right)&
                          frame.centroid_y_fullres_px.ge(top)&frame.centroid_y_fullres_px.lt(bottom)]
        table = self.example['table']
        self.assertEqual(len(table),2744)
        pd.testing.assert_frame_equal(table, frame[table.columns].reset_index(drop=True))

    def test_centroid_maps_preserve_frozen_values_and_geometry(self):
        for key in ('head_cv','head_epi','context_cv','context_epi'):
            figure,axis=plt.subplots()
            image=prediction_background(self.example,key)
            axis.imshow(image,cmap='gray',vmin=0,vmax=255)
            collection=add_prediction_overlay(axis,self.example,key)
            self.assertFalse(collection.get_rasterized())
            self.assertEqual(len(collection.get_paths()),1 if key.startswith('head_') else 2744)
            column='predicted_axis_coordinate' if key.endswith('_cv') else 'predicted_epithelial_distance_clipped_1p0'
            values=[self.example['selected'][column]] if key.startswith('head_') else self.example['table'][column]
            expected=colormaps['viridis'](values);expected[:,3]=.8
            np.testing.assert_array_equal(collection.get_facecolors(),expected)
            self.assertEqual(axis.get_xlim(),(-.5,image.shape[1]-.5))
            self.assertEqual(axis.get_ylim(),(image.shape[0]-.5,-.5))
            plt.close(figure)

    def test_unknown_source_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='xenium-source-lock-test-') as tmp:
            path=Path(tmp)/'manifest.json';path.write_bytes(DEFAULT_SUPERVISED.read_bytes())
            with self.assertRaises(RuntimeError):supervised_example(path)

    def test_both_renderers_share_same_frozen_helper(self):
        for path in (ROOT/'Figure_3/assembly/render.py',ROOT/'Figure_3/panel_A/figure_script/make_panel_A_image_components.py'):
            source=path.read_text()
            self.assertIn('supervised_example(',source)
            self.assertIn('add_prediction_overlay(',source)
            self.assertNotIn('bounded_inputs(',source)
            self.assertNotIn('bounded_sft_inputs(',source)


if __name__=='__main__':unittest.main()
