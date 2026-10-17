"""Protect the schematic-only publication edit from changing data or geometry."""
import unittest
from Figure_3.assembly.prepare import schematic_label


class SchematicProseTests(unittest.TestCase):
    def test_report_counts_removed_at_registered_anchors(self):
        for label in [': 22,048', ': 176,384', 'Total augmented crops']:
            self.assertIsNone(schematic_label(label, 200, 140))
            self.assertEqual(schematic_label(label, 200, 500), label)

    def test_procedure_replaces_training_summary(self):
        self.assertEqual(schematic_label('Pretraining Data Summary', 164, 103),
                         'Paired representation learning')
        self.assertEqual(schematic_label('13 supervised epochs', 317, 358),
                         'Full-model fine-tuning')

    def test_crop_widths_and_network_pixels_distinguished(self):
        for old, new in [('2048 px', '666 µm'), ('512 px', '166 µm'), ('128 px', '42 µm')]:
            self.assertEqual(schematic_label(old, 40, 180), new)
        self.assertEqual(schematic_label('256 px', 120, 180), '256 px')


if __name__ == '__main__':
    unittest.main()
