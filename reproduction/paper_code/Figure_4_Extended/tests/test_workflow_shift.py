import copy
import unittest

from Figure_4_Extended.publish_workflow_shift import check_translation


class WorkflowShiftTest(unittest.TestCase):
    def setUp(self):
        self.before = [
            {'id':'vectors','kind':'pdf','rect_pt':[0,130,370,100]},
            {'id':'heading','kind':'text','text':'IF-only fine-tuning','y_pt':166.0414},
            {'id':'letter','kind':'text','text':'b','y_pt':156},
        ]
        self.after = copy.deepcopy(self.before)
        self.after[0]['rect_pt'][1] += 12
        self.after[1]['y_pt'] += 12

    def test_uniform_shift_preserves_letter(self):
        check_translation(self.before, self.after, 12)

    def test_rejects_changed_text(self):
        self.after[1]['text'] = 'Different heading'
        with self.assertRaises(RuntimeError):
            check_translation(self.before, self.after, 12)

    def test_rejects_shifted_letter(self):
        self.after[2]['y_pt'] += 12
        with self.assertRaises(RuntimeError):
            check_translation(self.before, self.after, 12)

    def test_rejects_resizing(self):
        self.after[0]['rect_pt'][2] += 1
        with self.assertRaises(RuntimeError):
            check_translation(self.before, self.after, 12)


if __name__ == '__main__':
    unittest.main()
