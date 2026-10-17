"""Regression checks for original Figure 3A transformer-card occlusion."""
from __future__ import annotations

import copy
import json
from paper_paths import Path, lock_record, rendering_script
import tempfile
import unittest

from PIL import Image
from pypdf import PdfReader
from reportlab.pdfbase.pdfmetrics import stringWidth

from Figure_3.assembly.prepare import manual_layer_indices, obsolete_sft_contours
from figure_assembly.compositor import register_fonts
from figure_assembly.qa import font_check, private_content_check, render_png
from figure_assembly.static_art import extract_manual_art, inspect_path_events


class TransformerArtworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=Path(__file__).parent
        cls.layout=json.loads((directory/'layout.json').read_text())
        cls.lock=json.loads((directory/'manual_path_lock.json').read_text())
        cls.reference=Path(cls.layout['reference'])
        cls.groups=cls.lock['ordered_path_groups']
        cls.indices=[index for group in cls.groups for index in group['paint_indices']]
        cls.events=[event for event in inspect_path_events(cls.reference) if event['index'] in cls.indices]

    def test_original_fill_stroke_pairs_stay_in_one_layer(self):
        background,foreground,ordered=manual_layer_indices(self.events,{},self.groups)
        self.assertEqual(background,[])
        self.assertEqual(foreground,[])
        self.assertEqual(ordered,self.indices)
        self.assertEqual(len(ordered),28)

    def test_missing_reordered_or_changed_paths_fail_closed(self):
        with self.assertRaises(RuntimeError):
            manual_layer_indices(self.events[1:],{},self.groups)
        groups=copy.deepcopy(self.groups)
        groups[0]['paint_indices'][0:2]=reversed(groups[0]['paint_indices'][0:2])
        with self.assertRaises(RuntimeError):
            manual_layer_indices(self.events,{},groups)
        events=copy.deepcopy(self.events)
        events[0]['op']='S'
        with self.assertRaises(RuntimeError):
            manual_layer_indices(events,{},self.groups)
        events=copy.deepcopy(self.events)
        events[0]['bbox_pt'][0]-=1
        with self.assertRaises(RuntimeError):
            manual_layer_indices(events,{},self.groups)

    def test_extracted_original_front_cards_occlude_rear_edges(self):
        with tempfile.TemporaryDirectory(prefix='transformer-art-test-') as temporary:
            pdf=Path(temporary)/'ordered.pdf'
            extract_manual_art(self.reference,pdf,source_sha256=self.lock['reference_sha256'],
                               keep_paint_indices=self.indices)
            events=inspect_path_events(pdf)
            self.assertEqual([e['op'] for e in events],[e['op'] for e in self.events])
            self.assertEqual([e['bbox_pt'] for e in events],[e['bbox_pt'] for e in self.events])
            self.assertEqual(PdfReader(pdf).pages[0].extract_text(),'')
            self.assertEqual(font_check(pdf)['fonts'],[])
            self.assertEqual(private_content_check(pdf)['errors'],[])
            png=pdf.with_suffix('.png')
            render_png(pdf,png,288)
            with Image.open(png) as image:
                pixels=image.convert('RGB')
                # A rear card's right edge is hidden by the front cream fill.
                # Separating all fills from all strokes makes this pixel gold.
                for y in (75,122.4567):
                    hidden=pixels.getpixel((round(465.0064*4),round(y*4)))
                    front=pixels.getpixel((round(470*4),round(y*4)))
                    self.assertEqual(hidden,front)
                    self.assertGreater(min(hidden),200)

    def test_scale_embedding_label_fits_original_footprint(self):
        register_fonts()
        self.assertLessEqual(stringWidth('Scale embeddings','Arial',5.1),43.38938232)

    def test_replaced_supervised_inputs_drop_only_old_nuclear_outlines(self):
        events=inspect_path_events(self.reference)
        removed=obsolete_sft_contours(events)
        self.assertEqual(removed,{3963,3966,3995})
        # Adjacent rectangular input frames and every transformer path survive.
        self.assertFalse(removed & {3964,3967,3996})
        self.assertFalse(removed & set(self.indices))
        with self.assertRaises(RuntimeError):
            obsolete_sft_contours([e for e in events if e['index']!=3963])


if __name__=='__main__':
    unittest.main(verbosity=2)
