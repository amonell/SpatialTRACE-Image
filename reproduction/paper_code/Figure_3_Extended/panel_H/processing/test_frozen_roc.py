"""Read-only regression checks for the frozen ROC display boundary."""
import ast
import json
from paper_paths import Path, lock_record, rendering_script
import sys
import unittest

import pandas as pd

from Figure_3_Extended.panel_H.figure_script.make_panel_H_peyer_classifier_strategy import load_frozen_roc
from Figure_3_Extended.run_all_panels import command_plan

SATA = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")
FROZEN = SATA / "frozen_preprocessing/peyer_representation_v1/roc"


class FrozenROCTests(unittest.TestCase):
    def test_renderer_has_no_metric_or_curve_generation(self):
        script = Path(__file__).resolve().parents[1] / "figure_script/make_panel_H_peyer_classifier_strategy.py"
        tree = ast.parse(script.read_text())
        forbidden = {"roc_curve", "roc_auc_score", "average_precision_score"}
        self.assertFalse(any(isinstance(node, ast.Name) and node.id in forbidden for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module == "sklearn.metrics" for node in ast.walk(tree)))

    def test_runner_uses_explicit_frozen_manifest_only(self):
        command = command_plan(Path("/tmp/roc-dry-run-unused"), python=sys.executable, overwrite=False)["H"]
        self.assertIn(str(FROZEN / "manifest.json"), command)
        self.assertNotIn("--predictions-root", command)
        self.assertIn("figure_script", command[1])
        self.assertFalse(any("prepare_frozen_roc_curves.py" in arg for arg in command))

    def test_tables_match_frozen_scientific_metrics(self):
        manifest_path = FROZEN / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("Frozen SATA display tables unavailable")
        summary = pd.read_csv(SATA / "frozen_preprocessing/peyer_representation_v1/vit/strategy_metrics.tsv", sep="\t")
        summary = summary.rename(columns={"auroc": "test_auroc", "average_precision": "test_average_precision"})
        _, _, _, verified, curves = load_frozen_roc(manifest_path, summary)
        self.assertEqual(len(curves), 4)
        self.assertTrue(all(row["n"] == 819 and row["positive_n"] == 91 for row in verified))
        self.assertIn("checkpoints_frozen_before_test", json.loads(manifest_path.read_text()))


if __name__ == "__main__":
    unittest.main()
