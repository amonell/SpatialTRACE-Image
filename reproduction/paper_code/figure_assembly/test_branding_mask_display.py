"""Display-only regressions: names fit, classifier lines align, masks stay fixed."""
import json
from paper_paths import Path, lock_record, rendering_script
import unittest

import numpy as np

from figure_assembly.compositor import register_fonts
from figure_assembly.static_art import inspect_text, inspect_path_events
from Figure_1.assembly.prepare import model_name_elements
from Figure_2.assembly.prepare import manual_text_element
from Figure_3_Extended.panel_A.figure_script.make_panel_A_paired_pretraining_views import apply_mask, MASK_DISPLAY_GRAY
from reportlab.pdfbase.pdfmetrics import stringWidth

CODE = Path(__file__).resolve().parents[1]


class BrandingMaskTests(unittest.TestCase):
    def test_model_names_fit_original_boxes_and_keep_panel_ownership(self):
        layout = json.loads((CODE / 'Figure_1/assembly/layout.json').read_text())
        labels = model_name_elements(layout)
        self.assertEqual([x['text'] for x in labels], ['SpatialTRACE-', 'Graph', 'SpatialTRACE-', 'Image'])
        self.assertEqual([x['panel'] for x in labels], ['C', 'C', 'D', 'D'])
        paths = inspect_path_events(layout['reference'])
        for spec, index in zip(layout['model_name_labels'], [63, 68], strict=True):
            self.assertEqual(spec['block_rect_pt'], paths[index]['bbox_pt'])
            for line in spec['lines']:
                self.assertLessEqual(stringWidth(line['text'], spec['font'], spec['font_size_pt']), spec['block_rect_pt'][2] - 8)
            self.assertIn(spec['text'], layout['expected_text'])
        # Original architectural lines remain in the manual-label stream.
        source_text = {x['text'].strip() for x in inspect_text(layout['reference'])}
        self.assertTrue({'Graph', 'Attention', 'Network', 'Multiscale', 'Vision', 'transformer', '+', 'CNN'} <= source_text)

    def test_model_name_rejects_overflow(self):
        layout = json.loads((CODE / 'Figure_1/assembly/layout.json').read_text())
        layout['model_name_labels'][0]['font_size_pt'] = 14
        with self.assertRaisesRegex(RuntimeError, 'does not fit'):
            model_name_elements(layout)

    def test_image_module_text_group_is_vertically_centered(self):
        from fontTools.ttLib import TTFont
        layout = json.loads((CODE / 'Figure_1/assembly/layout.json').read_text())
        spec = next(s for s in layout['model_name_labels'] if s['panel'] == 'D')
        path = next(f['path'] for f in register_fonts() if f['font'] == spec['font'])
        with TTFont(path) as font:
            cmap, glyphs = font.getBestCmap(), font['glyf']
            scale = spec['font_size_pt'] / font['head'].unitsPerEm
            lines = spec['lines'] + [
                {'text': text, 'y_pt': y} for text, y in spec['module_lines'].items()]
            bounds = []
            for line in lines:
                chars = [glyphs[cmap[ord(c)]] for c in line['text'] if c != ' ']
                bounds.append((line['y_pt'] - max(g.yMax for g in chars) * scale,
                               line['y_pt'] - min(g.yMin for g in chars) * scale))
        _, top, _, height = spec['block_rect_pt']
        group_top = min(a for a, _ in bounds)
        group_bottom = max(b for _, b in bounds)
        self.assertAlmostEqual((group_top + group_bottom) / 2, top + height / 2, places=5)
        self.assertAlmostEqual(group_top - top, top + height - group_bottom, places=5)
        baselines = [line['y_pt'] for line in lines]
        np.testing.assert_allclose(np.diff(baselines),
            [11.52, 17.7255, 11.5195, 11.5205, 11.5195, 11.5205], atol=1e-7)

    def test_classifier_lines_share_center_and_original_spacing(self):
        register_fonts()
        layout = json.loads((CODE / 'Figure_2/assembly/layout.json').read_text())
        source = [x for x in inspect_text(layout['reference_pdf']) if x['text'] in {'Single classification', 'tasks'}]
        self.assertEqual(len(source), 2)
        rendered = [manual_text_element(x, i, layout['manual_text_replacements']) for i, x in enumerate(source)]
        self.assertEqual([x['text'] for x in rendered], ['Peyer’s patch', 'classifier'])
        self.assertEqual([x['align'] for x in rendered], ['center', 'center'])
        self.assertEqual([x['x_pt'] for x in rendered], [517.6989, 517.6989])
        self.assertAlmostEqual(rendered[1]['y_pt'] - rendered[0]['y_pt'], 9.6)
        for item in rendered:
            self.assertEqual(item['font_size_pt'], 8)
            self.assertLess(stringWidth(item['text'], item['font'], 8), 75.723 - 8)

    def test_other_figure2_manual_text_is_unchanged(self):
        layout = json.loads((CODE / 'Figure_2/assembly/layout.json').read_text())
        for i, item in enumerate(inspect_text(layout['reference_pdf'])):
            if item['text'] in {'Single classification', 'tasks', 'Single or multi', 'regression tasks'}:
                continue
            result = manual_text_element(item, i, layout['manual_text_replacements'])
            self.assertEqual(result['x_pt'], item['x_pt'])
            self.assertEqual(result['y_pt'], item['y_pt'])
            self.assertNotIn('align', result)
            expected = 'Coordinate GAT example' if item['text'] == 'Trainable GAT architecture' else item['text']
            self.assertEqual(result['text'], expected)

    def test_gray_masks_preserve_indices_and_unmasked_pixels(self):
        self.assertEqual(MASK_DISPLAY_GRAY, 128)
        original = np.arange(256 * 256, dtype=np.float32).reshape(256, 256) % 256
        before = original.copy()
        for seed in [7, 8]:
            displayed, indices = apply_mask(original, seed, .4)
            expected = sorted(map(int, np.random.default_rng(seed).choice(256, size=102, replace=False)))
            self.assertEqual(indices, expected)
            mask = np.zeros_like(original, dtype=bool)
            for index in indices:
                row, col = divmod(index, 16)
                mask[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = True
            np.testing.assert_array_equal(displayed[~mask], before[~mask])
            np.testing.assert_array_equal(displayed[mask], np.full(mask.sum(), 128))
            np.testing.assert_array_equal(original, before)

    def test_wrong_mask_input_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, '256 x 256'):
            apply_mask(np.zeros((128, 128)), 7, .4)


if __name__ == '__main__':
    unittest.main()
