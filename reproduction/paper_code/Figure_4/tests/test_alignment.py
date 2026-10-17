"""Display-only invariants for the author-requested Figure 4 alignment pass."""
import unittest
import numpy as np
from Figure_4.assembly import design


class AlignmentTests(unittest.TestCase):
    def test_top_images_and_headings_have_common_baselines(self):
        rectangles = [*design.MICROSCOPY_RECTS.values(), design.SPARSE_RECT]
        self.assertEqual({r[1] for r in rectangles}, {36.0})
        self.assertEqual({r[1]+r[3] for r in rectangles}, {177.0})
        self.assertEqual(design.PANEL_LETTERS['A'][1], design.PANEL_LETTERS['B'][1])

    def test_sparse_image_is_larger_without_stretching(self):
        self.assertAlmostEqual(design.SPARSE_RECT[2], 141*5552/4049)
        self.assertAlmostEqual(design.SPARSE_RECT[3]/design.SPARSE_RECT[2], 4049/5552)
        self.assertGreater(design.SPARSE_RECT[2]/153.13, 1.26)
        np.testing.assert_array_equal(np.array(design.SPARSE_RECT)[[1, 3]],
                                      np.array(design.SPARSE_BAR)[[1, 3]])

    def test_scatter_axes_and_colorbars_align(self):
        self.assertEqual({(r[1], r[3]) for r in design.SCATTER_RECTS + design.SCATTER_BARS},
                         {(234., 162.)})
        self.assertEqual({r[2] for r in design.SCATTER_RECTS}, {162.})

    def test_all_four_zoom_frames_and_bars_align(self):
        for rows in (design.WHOLE_RECTS, design.ZOOM_RECTS, design.MAP_BARS):
            self.assertEqual(len({tuple(r[1:]) for r in rows}), 1)
        for zoom, bar in zip(design.ZOOM_RECTS, design.MAP_BARS):
            self.assertAlmostEqual(zoom[0]+zoom[2]/2, bar[0]+bar[2]/2)
            self.assertAlmostEqual(zoom[2]/zoom[3], 4/3)

    def test_imaps_are_narrower_with_readable_labels_and_aligned_edges(self):
        self.assertEqual({(r[1], r[2], r[3]) for r in design.IMAP_RECTS}, {(666., 152., design.IMAP_HEIGHT)})
        left, right = design.IMAP_RECTS
        self.assertEqual(design.IMAP_LABEL_SIZE, 5.7)
        bar = design.IMAP_BAR
        self.assertAlmostEqual((left[0]+right[0]+right[2])/2, bar[0]+bar[2]/2)

    def test_statistics_share_a_right_column_and_lower_row_is_aligned(self):
        e, g = design.ABUNDANCE_RECT, design.ENRICHMENT_RECT
        self.assertAlmostEqual(e[0]+e[2]/2, g[0]+g[2]/2)
        self.assertLess(e[2], 156.)
        self.assertLess(e[3], 136.5)
        self.assertGreater(g[2], 156.)
        np.testing.assert_array_equal(np.array(design.IMAP_RECTS[0])[[1, 3]],
                                      np.array(design.ENRICHMENT_RECT)[[1, 3]])
        self.assertLess(design.ABUNDANCE_RECT[1]+design.ABUNDANCE_RECT[3], design.ENRICHMENT_RECT[1])
        self.assertLess(design.IMAP_RECTS[1][0]+design.IMAP_RECTS[1][2], design.ENRICHMENT_RECT[0])
        self.assertEqual(design.PANEL_LETTERS['F'][1], design.PANEL_LETTERS['G'][1])


if __name__ == '__main__':
    unittest.main()
