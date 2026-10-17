"""Regression checks for final-page/component scientific-input consistency."""
from paper_paths import Path, lock_record, rendering_script
import ast
import json
import tempfile
import unittest

import numpy as np

from Figure_3.figure_inputs import paired_views, locked_zoom, bounded_inputs, prediction_overlays, DEFAULT_ZOOM_RECORD, DEFAULT_PAIRED
from Figure_1.common import load_source_regions
from Figure_1.assembly import render_util as U

ROOT = Path(__file__).resolve().parents[1]


class ConsistencyTests(unittest.TestCase):
    def test_student_teacher_geometry(self):
        views,metadata=paired_views(DEFAULT_PAIRED/'manifest.json',DEFAULT_PAIRED/'paired_views.npz')
        self.assertEqual(metadata['center_id'],'if:cont1:nucleus:352082')
        self.assertEqual((metadata['local_prepared_index'],metadata['context_prepared_index']),(2352,5035))
        self.assertEqual((metadata['student_variant'],metadata['teacher_variant']),(1,0))
        with np.load(DEFAULT_PAIRED/'paired_views.npz') as frozen, np.load('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3/stored_outputs/panel_A/panel_A_exact_historical_region.npz') as source:
            for branch,seed in [('local',7),('context',8)]:
                teacher=views['teacher_'+branch];student=views['student_'+branch]
                np.testing.assert_array_equal(teacher,frozen[branch+'_teacher_image']*255)
                np.testing.assert_allclose(teacher,source[branch+'_model_input'],atol=2e-5,rtol=0)
                mask=np.zeros((256,256),bool)
                for index in np.random.default_rng(seed).choice(256,102,replace=False):
                    row,column=divmod(int(index),16);mask[row*16:(row+1)*16,column*16:(column+1)*16]=True
                self.assertEqual(np.count_nonzero(mask),102*16*16)
                self.assertTrue(np.all(student[mask]==0))
                np.testing.assert_array_equal(student[~mask],(frozen[branch+'_student_image']*255)[~mask])
                self.assertFalse(np.array_equal(student[~mask],np.rot90(teacher,1)[~mask]))

    def test_paired_views_stop_on_unlocked_inputs(self):
        with self.assertRaises(RuntimeError):
            paired_views(DEFAULT_PAIRED/'manifest.json',DEFAULT_PAIRED/'paired_views.npz',seed=8)
        with tempfile.TemporaryDirectory(prefix='figure-paired-lock-test-') as tmp:
            changed=Path(tmp)/'manifest.json'
            changed.write_text((DEFAULT_PAIRED/'manifest.json').read_text())
            with self.assertRaises(RuntimeError):
                paired_views(changed,DEFAULT_PAIRED/'paired_views.npz')

    def test_frozen_zoom_has_no_render_time_selection(self):
        self.assertEqual(locked_zoom(),(12577.508856109605,19377.81787081844,4194.409173331034,10888.386891712418))
        self.assertTrue(DEFAULT_ZOOM_RECORD.is_relative_to(ROOT))
        for panel in ['D','E']:
            script=next((ROOT/f'Figure_3/panel_{panel}/figure_script').glob('*.py'))
            tree=ast.parse(script.read_text())
            calls={node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
            self.assertIn('locked_zoom',calls);self.assertNotIn('zoom_limits',calls)

    def test_figure1_registered_regions(self):
        r=load_source_regions(ROOT/'Figure_1/assembly/source_region_lock.json')
        self.assertEqual(r['regions']['sparse_axis']['visible_villus_ids'],['day90_SI_r2__villus_004'])
        self.assertEqual(r['original_image_streams']['/Im5'],r['original_image_streams']['/Im6'])
        for panel in ['A','B']:
            text=next((ROOT/f'Figure_1/panel_{panel}/figure_script').glob('*.py')).read_text()
            self.assertIn('load_source_regions(',text)
        text=(ROOT/'Figure_1/panel_C/figure_script/make_panel_C_components.py').read_text()
        self.assertNotIn('zoom.crypt_villus_prediction',text)
        self.assertIn('paired_training_example',text)
        panel_map=json.loads((ROOT/'Figure_1/panel_order.json').read_text())
        active=next(p['components'] for p in panel_map['panels'] if p['letter']=='C')
        self.assertNotIn('panel_C_spatial_neighbors',active)
        self.assertNotIn('panel_C_focus_gene_expression',active)

    def test_metric_render_uses_frozen_values(self):
        text=(ROOT/'Figure_3/panel_C/figure_script/make_panel_C_coordinate_scatter.py').read_text()
        self.assertNotIn('stats.pearsonr',text)
        self.assertNotIn('Spearman ρ =',text)
        self.assertIn('results = saved',text)

    def test_bounded_overlays_are_saved_values(self):
        table,labels,_=bounded_inputs()
        with np.load('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3/stored_outputs/panel_A/panel_A_exact_historical_region.npz') as crops:
            out=prediction_overlays(crops,table,labels,352082)
            target=crops['fine_target_mask'].astype(bool)
            self.assertEqual(out['head_cv'].shape,(*target.shape,4))
            self.assertTrue(np.all(out['head_cv'][target,3]==.72))
            self.assertTrue(np.all(out['head_cv'][~target,3]==0))
            visible=np.isin(labels,table.nucleus_id)
            np.testing.assert_allclose(out['context_cv'][visible,3],.7)
            self.assertTrue(np.all(out['context_cv'][~visible,3]==0))

    def test_panel_layers_preserve_visibility(self):
        from pypdf import PdfReader
        fig=U.figure();a=U.axis(fig,[10,10,50,50]);b=U.axis(fig,[80,10,50,50]);a.plot([0,1]);b.plot([1,0])
        with tempfile.TemporaryDirectory(prefix='figure-panel-layer-test-') as tmp:
            layers=U.save_panel_layers(fig,Path(tmp),{'first':a,'second':b},{'first':'A','second':'B'})
            self.assertEqual(set(layers),{'A','B'})
            self.assertTrue(a.get_visible() and b.get_visible())
            for path in layers.values():
                page=PdfReader(Path(tmp)/path).pages[0]
                self.assertEqual(list(map(float,page.mediabox[2:])),[595.276,841.89])
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__=='__main__':unittest.main()
