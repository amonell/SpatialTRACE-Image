"""Fast update-safety checks; synthetic inputs never modify frozen science."""
import copy
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Figure_4.frozen_annotations import joined_training_history, training_example_counts
from Figure_4.spatial import (GATE_ORDER, frozen_gate_percentages, gate_background,
                              gate_percentages, gate_thresholds)


class FrozenAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.gates = {"schema_version": "figure4_coordinate_gates/v1",
                      "boundary_conventions": {gate: "schema-defined inequality" for gate in GATE_ORDER},
                      "thresholds": {"split_x": .28, "crypt_lp_cv_min": .25,
                                     "crypt_lp_cv_max": .4, "x_range": [0, 1], "y_range": [0, 1]}}

    def test_gate_geometry_uses_definition(self):
        altered = copy.deepcopy(self.gates)
        altered["thresholds"].update(split_x=.31, crypt_lp_cv_min=.21, crypt_lp_cv_max=.48)
        figure, axis = plt.subplots()
        try:
            gate_background(axis, altered)
            self.assertEqual(axis.patches[0].get_width(), .31)
            self.assertEqual(axis.patches[0].get_y(), .48)
            self.assertEqual(axis.patches[-1].get_height(), .21)
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [.31, .31])
        finally:
            plt.close(figure)

    def test_boundary_membership_is_exhaustive(self):
        frame = pd.DataFrame({"predicted_epithelial_distance_clipped_1p0": [.28, .29, .28, .29, .29],
                              "predicted_axis_coordinate": [.4, .4, .39, .25, .24]})
        self.assertEqual(gate_percentages(frame, self.gates), {gate: 20 for gate in GATE_ORDER})
        self.gates["thresholds"]["crypt_lp_cv_min"] = .5
        with self.assertRaises(RuntimeError):
            gate_thresholds(self.gates)

    def test_percentages_come_from_frozen_table(self):
        table = pd.DataFrame({"condition": ["DMSO"] * 5, "gate": GATE_ORDER,
                              "percent": [11., 29., 18., 22., 20.]})
        self.assertEqual(frozen_gate_percentages(table, "DMSO")["Top IE"], 11.)
        with self.assertRaises(RuntimeError):
            frozen_gate_percentages(pd.concat([table, table.iloc[:1]]), "DMSO")

    def test_history_offsets_follow_saved_stage_lengths(self):
        frames = [pd.DataFrame({"epoch": range(1, n + 1)}) for n in (3, 5)]
        summaries = [{"epochs": 3, "best_epoch": 2}, {"epochs": 5, "best_epoch": 4}]
        joined, info = joined_training_history(frames, summaries)
        self.assertEqual(joined.completed_epoch.tolist(), list(range(1, 9)))
        self.assertEqual(info["selected_combined_epochs"], [2, 7])
        self.assertEqual(info["stage_boundary"], 3.5)
        summaries[1]["best_epoch"] = 6
        with self.assertRaises(RuntimeError):
            joined_training_history(frames, summaries)

    def test_training_counts_use_only_saved_training_rows(self):
        frame = pd.DataFrame({"row_id": list("abcd"), "split": ["train"] * 3 + ["validation"],
                              "villus_shape_id": ["v1", "v1", "v2", "v3"]})
        self.assertEqual(training_example_counts(frame, {"train_row_count": 3}), {"cells": 3, "villi": 2})
        with self.assertRaises(RuntimeError):
            training_example_counts(frame, {"train_row_count": 4})


if __name__ == "__main__":
    unittest.main()
