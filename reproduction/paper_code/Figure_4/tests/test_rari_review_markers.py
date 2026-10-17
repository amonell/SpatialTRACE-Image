"""Marker revisions must leave spatial values and display colors unchanged."""
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Figure_4.panel_D.figure_script.make_panel_D_rari_options import enlarge_scatter_dots, locator


class ReviewMarkerTests(unittest.TestCase):
    def test_changes_only_scatter_sizes(self):
        figure,axis=plt.subplots()
        try:
            image=axis.imshow(np.zeros((2,2)),vmin=0,vmax=1)
            line,=axis.plot([0,1],[0,1],linewidth=1)
            artists=[axis.scatter([0,1],[1,0],c=[.2,.8],s=size,vmin=0,vmax=1)
                     for size in (3,10,4.2,4.2)]
            figure.canvas.draw()
            original=[(a.get_sizes().copy(),a.get_offsets().copy(),a.get_facecolors().copy(),a.get_clim())
                      for a in artists]
            enlarge_scatter_dots(figure,2)
            figure.canvas.draw()
            for artist,(sizes,offsets,colors,clim) in zip(artists,original,strict=True):
                np.testing.assert_array_equal(artist.get_sizes(),sizes*4)
                np.testing.assert_array_equal(artist.get_offsets(),offsets)
                np.testing.assert_array_equal(artist.get_facecolors(),colors)
                self.assertEqual(artist.get_clim(),clim)
            np.testing.assert_array_equal(image.get_array(),np.zeros((2,2)))
            self.assertEqual(line.get_linewidth(),1)
        finally:
            plt.close(figure)

    def test_missing_layers_fail_before_export(self):
        figure,axis=plt.subplots()
        try:
            axis.scatter([0],[0])
            with self.assertRaises(RuntimeError):enlarge_scatter_dots(figure,2)
        finally:
            plt.close(figure)

    def test_locator_keeps_continuing_option_numbers(self):
        cells=pd.DataFrame({'centroid_x_fullres_px':[0,20000],
                            'centroid_y_fullres_px':[0,20000],
                            'predicted_axis_coordinate':[.1,.9]})
        windows=[{'bounds_xxyy_px':[i*500,i*500+6000,1000,5500]} for i in range(20)]
        figure=locator(cells,windows,[0,6000,0,4500],first_option=11)
        try:
            self.assertEqual([text.get_text() for text in figure.axes[0].texts],
                             [str(i) for i in range(11,31)]+['Current R1'])
            self.assertIn('20 new fields',figure.axes[0].get_title())
        finally:
            plt.close(figure)


if __name__=='__main__':unittest.main()
