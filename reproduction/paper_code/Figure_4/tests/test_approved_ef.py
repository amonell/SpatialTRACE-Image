"""The final E/G plots must preserve every value in the approved frozen analysis."""
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import unittest

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from Figure_4.assembly.common import PageInputs, configure
from Figure_4.replicate_revision_20261014 import approved_ab_panels as renderer


class PlotCapture:
    figure = PageInputs.figure
    axis = PageInputs.axis

    def save(self, figure, name):
        self.result = figure
        self.stem = name


class ApprovedPanelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pointer = json.loads(renderer.POINTER.read_text())
        rec = pointer['analysis_manifest']
        raw = Path(rec['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == rec['sha256']
        files = {Path(r['path']).name: r for r in json.loads(raw)['outputs']}
        def table(name):
            r = files[name]
            assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256']
            return pd.read_csv(r['path'], float_precision='round_trip')
        cls.data = {'metrics': table('per_mouse_metrics.csv'), 'gates': table('per_mouse_gate_counts.csv'),
                    'stats': table('statistics.csv'), 'colors': pointer['condition_palette']}

    def setUp(self):
        configure()

    def check_plot(self, name, table, value, expected_stars):
        page = PlotCapture()
        getattr(renderer, name)(page, self.data)
        try:
            axis = page.result.axes[0]
            self.assertFalse(page.result.texts, 'E/G titles must remain absent')
            self.assertFalse(axis.get_title())
            for legend in page.result.legends:
                self.assertEqual([t.get_text() for t in legend.get_texts()], ['DMSO', 'RARi'])
            dots = [c for c in axis.collections if isinstance(c, PathCollection)]
            actual = np.concatenate([c.get_offsets()[:, 1] for c in dots])
            np.testing.assert_array_equal(np.sort(actual), np.sort(table[value]))
            self.assertGreater(actual.min(), axis.get_ylim()[0])
            self.assertLess(actual.max(), axis.get_ylim()[1])
            for i, points in enumerate(dots):
                color = self.data['colors'][('DMSO', 'RARi')[i % 2]]
                np.testing.assert_allclose(points.get_facecolors()[0], to_rgba(color))
                np.testing.assert_array_equal(points.get_sizes(), [25.0])
            annotations = [t.get_text() for t in axis.texts]
            self.assertEqual(annotations, expected_stars)
            summaries = [c for c in axis.collections if isinstance(c, LineCollection)]
            self.assertEqual(len(summaries), 2*len(dots))
            for i, points in enumerate(dots):
                y = points.get_offsets()[:, 1]
                mean, sem = np.mean(y), np.std(y, ddof=1)/np.sqrt(len(y))
                horizontal, vertical = summaries[2*i:2*i+2]
                np.testing.assert_allclose(horizontal.get_segments()[0][:, 1], [mean, mean])
                np.testing.assert_allclose(vertical.get_segments()[0][:, 1], [mean-sem, mean+sem])
        finally:
            plt.close(page.result)

    def test_abundance_points_sem_and_significance(self):
        self.check_plot('abundance', self.data['metrics'], 'p14_per10k', ['**  P = 0.004'])

    def test_enrichment_points_sem_and_significance(self):
        self.check_plot('enrichment', self.data['gates'], 'log2_enrich', ['*  q = 0.046', '*  q = 0.016'])

    def test_active_panel_order(self):
        from figure_organization import panel_map
        panels = panel_map('Figure_4')
        self.assertEqual([p['letter'] for p in panels], list('ABCDEFG'))
        self.assertEqual(panels[4]['components'], ['panel_E_P14_abundance'])
        self.assertEqual(panels[5]['components'], ['panel_F_P14_IMAPs'])
        self.assertEqual(panels[6]['components'], ['panel_G_P14_enrichment'])


if __name__ == '__main__':
    unittest.main()
