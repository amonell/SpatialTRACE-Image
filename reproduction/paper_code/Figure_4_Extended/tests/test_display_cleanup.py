"""Regression checks for the approved Extended Figure 4 C/E display cleanup."""
import unittest
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
import numpy as np
from Figure_4.assembly.common import PageInputs, configure
from Figure_4.replicate_revision_20261014.final_panels import load_inputs, controls
from Figure_4_Extended.history_labels import draw_stage_labels


class PlotCapture:
    figure = PageInputs.figure
    axis = PageInputs.axis
    read = PageInputs.read

    def __init__(self):
        self.inputs, self.warnings, self.anchors = [], [], []
        self.details = {}

    def save(self, figure, name):
        self.result, self.stem = figure, name


class DisplayCleanupTests(unittest.TestCase):
    def setUp(self):
        configure()

    def test_stage_labels_follow_frozen_epoch_ranges(self):
        figure, axis = plt.subplots(figsize=(4, 2), dpi=100)
        try:
            axis.set_xlim(.5, 11.5)
            records = draw_stage_labels(axis, {'completed_epochs': [7, 4], 'stage_boundary': 7.5})
            self.assertEqual([r['center_epoch'] for r in records], [4, 9.5])
            self.assertEqual([(r['first_epoch'], r['last_epoch']) for r in records], [(1, 7), (8, 11)])
            figure.canvas.draw()
            divider = axis.transData.transform((7.5, 0))[0]
            boxes = [t.get_window_extent(figure.canvas.get_renderer()) for t in axis.texts]
            self.assertLess(boxes[0].x1, divider)
            self.assertGreater(boxes[1].x0, divider)
            # Display logic also follows a different frozen stage length.
            axis.clear()
            records = draw_stage_labels(axis, {'completed_epochs': [3, 2], 'stage_boundary': 3.5})
            self.assertEqual([r['center_epoch'] for r in records], [2, 4.5])
        finally:
            plt.close(figure)

    def test_one_shared_non_p14_bar_without_sample_sizes_or_footer(self):
        page = PlotCapture()
        data = load_inputs(page)
        grids = {condition: data['density'][f'{condition}__non_p14__mean_mouse'].copy()
                 for condition in ('DMSO', 'RARi')}
        controls(page, data)
        try:
            axes = page.result.axes
            self.assertEqual([a._assembly_id for a in axes], ['D_DMSO', 'D_RARi', 'D_shared_density_bar'])
            self.assertEqual(page.stem, 'panel_E_non_P14_IMAPs')
            for axis, condition in zip(axes[:2], ('DMSO', 'RARi'), strict=True):
                mesh = next(item for item in axis.collections if isinstance(item, QuadMesh))
                self.assertEqual((mesh.norm.vmin, mesh.norm.vmax), (0, 16))
                plotted = np.ma.asarray(mesh.get_array())
                grid = grids[condition].T
                np.testing.assert_array_equal(plotted.compressed(), grid[~np.ma.getmaskarray(plotted)])
                np.testing.assert_array_equal(grids[condition], data['density'][f'{condition}__non_p14__mean_mouse'])
            self.assertEqual(axes[2].get_ylabel(), 'Probability density')
            np.testing.assert_array_equal(axes[2].get_yticks(), [0, 4, 8, 12, 16])
            labels = [t.get_text() for t in page.result.texts]
            self.assertIn('DMSO non-P14', labels)
            self.assertIn('RARi non-P14', labels)
            self.assertFalse(any('n =' in t or 'mouse weights' in t or 'mean %' in t for t in labels))
            self.assertEqual(len(page.anchors), 10)  # all original gate-label boxes
        finally:
            plt.close(page.result)
            data['density'].close()


if __name__ == '__main__':
    unittest.main()
