"""Display-only regression checks against the actual registered artwork."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Figure_4.assembly.common import PAGE, configure
from Figure_4.imap_labels import registered_paths, draw_registered_gate_labels
from Figure_4.spatial import GATE_COLORS
from Figure_4_Extended.probability_display import probability_display, probability_range
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties


class DisplayRegistrationTests(unittest.TestCase):
    def test_all_twenty_label_boxes_and_baselines_match_original(self):
        configure()
        count = 0
        for population in ("P14", "non-P14"):
            config, paths = registered_paths(population)
            self.assertEqual(len(paths), 10)
            figure = plt.figure(figsize=(PAGE[0]/72, PAGE[1]/72))
            try:
                for condition, record in config["populations"][population].items():
                    x, y, w, h = record["axis_rect_pt"]
                    axis = figure.add_axes([x/PAGE[0], 1-(y+h)/PAGE[1], w/PAGE[0], h/PAGE[1]])
                    before = axis.get_position().bounds
                    report = draw_registered_gate_labels(axis, {gate: 42. for gate in GATE_COLORS},
                                                         population, condition, GATE_COLORS)
                    self.assertEqual(axis.get_position().bounds, before)
                    for box in report["boxes"]:
                        np.testing.assert_allclose(box["rect_pt"], box["expected_rect_pt"], atol=.001, rtol=0)
                        if box["intentional_difference"]:
                            original = box["reference_rect_pt"]
                            actual = box["rect_pt"]
                            if box.get("display_fit")=="wrap_preserving_bottom_right":
                                np.testing.assert_allclose([original[0]+original[2],original[1]+original[3]],
                                                           [actual[0]+actual[2],actual[1]+actual[3]])
                            else:
                                self.assertAlmostEqual(original[0]+original[2]/2, actual[0]+actual[2]/2)
                                np.testing.assert_allclose([original[1], original[3]], [actual[1], actual[3]])
                        count += 1
                    for line in report["text_baselines"]:
                        np.testing.assert_allclose([line["x_pt"], line["y_pt"]],
                                                   [line["expected_x_pt"], line["reference_y_pt"]], atol=.001, rtol=0)
                    self.assertEqual(sum("42.0%" in text.get_text() for text in axis.texts), 5)
                    self.assertTrue(all(text.get_fontfamily() == ["Arial"] for text in axis.texts))
            finally:
                plt.close(figure)
        self.assertEqual(count, 20)

    def test_standalone_uses_the_same_relative_positions(self):
        configure()
        config, _ = registered_paths("P14")
        record = config["populations"]["P14"]["DMSO"]
        figure = plt.figure(figsize=(8, 6))
        try:
            axis = figure.add_axes([.15, .2, .7, .55])
            report = draw_registered_gate_labels(axis, {gate: 12.3 for gate in GATE_COLORS},
                                                 "P14", "DMSO", GATE_COLORS)
            x, y, width, height = record["axis_rect_pt"]
            expected = record["labels"][0]["lines"][0]
            np.testing.assert_allclose(axis.texts[0].get_position(),
                                       [(expected["x_pt"]-x)/width, 1-(expected["y_pt"]-y)/height])
            self.assertAlmostEqual(report["font_size_pt"], 6.6*.7*8*72/width)
        finally:
            plt.close(figure)

    def test_musc_uses_original_box_and_baselines_without_changing_gate_keys(self):
        configure()
        for population in ('P14','non-P14'):
            config,_=registered_paths(population)
            for condition,record in config['populations'][population].items():
                figure=plt.figure(figsize=(PAGE[0]/72,PAGE[1]/72),dpi=300)
                try:
                    x,y,w,h=record['axis_rect_pt']
                    axis=figure.add_axes([x/PAGE[0],1-(y+h)/PAGE[1],w/PAGE[0],h/PAGE[1]])
                    report=draw_registered_gate_labels(axis,{g:12.3 for g in GATE_COLORS},population,condition,GATE_COLORS)
                    label=next(r for r in record['labels'] if r['gate']=='Muscularis')
                    self.assertNotIn('display_lines', label)
                    self.assertNotIn('display_box_rect_pt', label)
                    self.assertEqual([r['text'] for r in label['lines']], ['Musc','{percentage:.1f}%'])
                    patch=axis.patches[-1].get_window_extent(figure.canvas.get_renderer())
                    for text in axis.texts[-2:]:
                        bounds=text.get_window_extent(figure.canvas.get_renderer())
                        self.assertGreaterEqual(bounds.x0,patch.x0)
                        # Allow sub-pixel font-bbox rounding; registered boxes
                        # and visible glyph baselines remain checked exactly.
                        self.assertLessEqual(bounds.x1,patch.x1 + .1*figure.dpi/72)
                        # Font-wide descender bounds include empty space under
                        # digits. Check the actual visible vector glyphs against
                        # the preserved percentage baseline and box instead.
                        glyphs=TextPath((0,0),text.get_text(),size=text.get_fontsize(),
                                        prop=text.get_fontproperties()).get_extents()
                        baseline=text.get_transform().transform(text.get_position())[1]
                        self.assertGreaterEqual(baseline+glyphs.y0*figure.dpi/72,patch.y0)
                        self.assertLessEqual(baseline+glyphs.y1*figure.dpi/72,patch.y1)
                    self.assertEqual(set(r['gate'] for r in report['boxes']),set(GATE_COLORS))
                finally:plt.close(figure)

    def test_probability_colors_preserve_current_full_range_without_mutating_values(self):
        values = np.array([0., .1, .25, .5, .52237624])
        before = values.copy()
        config, norm = probability_display()
        np.testing.assert_allclose(norm(values), values)
        np.testing.assert_array_equal(values, before)
        self.assertEqual(config["ticklabels"][-1], "1")
        self.assertEqual(config["marker_area_pt2"], 5.5)
        self.assertEqual(probability_range(values)["above_display_maximum"], 0)
        self.assertFalse(probability_range(values)["predictions_rescaled"])

    def test_longer_current_percentage_fits_approved_centered_box(self):
        config, _ = registered_paths("non-P14")
        label = next(item for item in config["populations"]["non-P14"]["RARi"]["labels"]
                     if item["gate"] == "Crypt LP")
        x, y, width, height = label["display_box_rect_pt"]
        line = label["lines"][0]
        glyphs = TextPath((line["x_pt"]+label["text_x_shift_pt"], 0),
                          line["text"].format(percentage=10.5), size=6.6,
                          prop=FontProperties(family="Arial", weight="bold")).get_extents()
        self.assertGreater(glyphs.x0, x)
        self.assertLess(glyphs.x1, x+width)
        self.assertEqual(width, 50.267)

    def test_invalid_probability_is_not_silently_clipped_as_data(self):
        with self.assertRaises(RuntimeError):
            probability_range([.1, 1.2])


if __name__ == "__main__":
    unittest.main()
