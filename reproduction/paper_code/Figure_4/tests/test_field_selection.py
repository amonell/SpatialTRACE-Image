"""The selected field pair must match between main D and extended E."""
import json
from paper_paths import Path, lock_record, rendering_script
import tempfile
import unittest
import pandas as pd

from Figure_4.field_selection import DEFAULT_POINTER, load_fields
from Figure_4.common import require_file

ROOT=Path(__file__).resolve().parents[2]


class SharedFieldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path,cls.release=load_fields()

    def test_exact_author_choice(self):
        self.assertEqual(self.release['selection']['selected'],{'DMSO':'Current_DMSO','RARi':'RARi32'})
        self.assertEqual(self.release['windows']['DMSO']['bounds_xxyy_px'],[45000,51000,35000,39500])
        self.assertEqual(self.release['windows']['RARi']['bounds_xxyy_px'],[31000,37000,47000,51500])
        self.assertTrue(self.release['same_bounds_in_main_and_extended'])
        self.assertFalse(self.release['model_inference_run'])

    def test_all_selected_values_and_gates_match_saved_predictions(self):
        for condition,window in self.release['windows'].items():
            rec=window['source_predictions']
            full=pd.read_parquet(require_file(rec['path'],rec['sha256']))
            rec=window['cells'];actual=pd.read_parquet(require_file(rec['path'],rec['sha256']))
            x0,x1,y0,y1=window['bounds_xxyy_px']
            expected=full.loc[full.centroid_x_fullres_px.ge(x0)&full.centroid_x_fullres_px.lt(x1)&
                              full.centroid_y_fullres_px.ge(y0)&full.centroid_y_fullres_px.lt(y1)].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual[full.columns],expected,check_exact=True)
            self.assertEqual(len(actual),window['n'])
            self.assertEqual(len(actual),{'DMSO':3776,'RARi':2603}[condition])
            self.assertTrue(actual.is_P14.equals(actual['class'].eq('Transferred')))

    def test_display_settings_unchanged(self):
        path=Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json')
        old=json.loads(path.read_text())
        for key in ('display','gates','checkpoint','pixel_size_um'):
            self.assertEqual(self.release[key],old[key])

    def test_original_dmso_assets_are_reused_exactly(self):
        path=Path(self.release['selection']['previous_fields']['path'])
        previous=json.loads(path.read_text())
        self.assertEqual(self.release['windows']['DMSO'],previous['windows']['DMSO'])

    def test_rari_matches_option32_review(self):
        rec=self.release['selection']['reviewed_cells']
        reviewed=pd.read_parquet(require_file(rec['path'],rec['sha256']))
        rec=self.release['windows']['RARi']['cells']
        current=pd.read_parquet(require_file(rec['path'],rec['sha256']))
        pd.testing.assert_frame_equal(current[reviewed.columns],reviewed,check_exact=True)

    def test_extended_images_fit_matching_native_aspect(self):
        layout=json.loads((ROOT/'Figure_4_Extended/assembly/layout.json').read_text())
        settings=layout['gate_projection_layout']
        anchors={a['id']:a['rect_pt'] for a in layout['anchors']}
        for name,rect in zip(('E_DMSO_image','E_DMSO_gates','E_RARi_image','E_RARi_gates'),settings['axes_rect_pt']):
            self.assertEqual(rect,anchors[name])
            self.assertAlmostEqual(rect[2]/rect[3],4/3)
        self.assertEqual(settings['cell_marker_area_pt2'],1.8)
        self.assertEqual(settings['P14_marker_area_pt2'],9)

    def test_hash_mismatch_stops_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            pointer=json.loads(DEFAULT_POINTER.read_text())
            pointer['manifest']['sha256']='0'*64
            path=Path(folder)/'pointer.json';path.write_text(json.dumps(pointer))
            with self.assertRaises(RuntimeError):load_fields(path)

    def test_unapproved_pointer_stops_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            pointer=json.loads(DEFAULT_POINTER.read_text());pointer['status']='REVIEW_ONLY'
            path=Path(folder)/'pointer.json';path.write_text(json.dumps(pointer))
            with self.assertRaises(RuntimeError):load_fields(path)


if __name__=='__main__':unittest.main()
