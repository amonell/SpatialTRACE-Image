#!/usr/bin/env python3
"""Explicit bounded conversion of frozen classifier predictions to ROC tables.

No model fitting, inference, selection, split construction, or permutation is
performed. This preparation is never invoked by a panel-rendering entry point.
"""
from __future__ import annotations

import argparse
import json
from paper_paths import Path, lock_record, rendering_script
import sys

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_3_Extended.common import file_record, protect_outputs, require_file, write_provenance

from frozen_preprocessing.peyer_representation_v1.release import load_release
ORDER = ("frozen_manual_sparse", "full_manual_sparse", "frozen_gat_soft_sparse", "full_gat_soft_sparse")


def prepare(release_path: Path, predictions_root: Path, output: Path) -> Path:
    _, active = load_release()
    release_path = require_file(release_path, active["comparison"]["sha256"])
    release = json.loads(release_path.read_text())
    if release["selected_variant"] != "full_manual_sparse" or release["selection_changed"] or release["threshold"] != .5:
        raise RuntimeError("The prespecified strategy or threshold changed")
    label_lock = require_file(release["lock"]["path"], release["lock"]["sha256"])
    summary_path = require_file(release["strategy_metrics"]["path"], release["strategy_metrics"]["sha256"])
    models = release["models"]
    model_paths = {name: require_file(models[name]["checkpoint"]["path"], models[name]["checkpoint"]["sha256"]) for name in ORDER}
    prediction_paths = {name: require_file(predictions_root / name / "xenium_test_predictions/predictions.csv", models[name]["test_predictions"]["sha256"]) for name in ORDER}
    table = pd.read_csv(summary_path, sep="\t").set_index("variant").reindex(ORDER).reset_index()
    table = table.rename(columns={"auroc": "test_auroc", "average_precision": "test_average_precision"})
    table["checkpoint"] = table.variant.map({name: str(path) for name, path in model_paths.items()})
    curve_path, metric_path, manifest_path = output / "roc_curves.tsv", output / "display_metrics.tsv", output / "manifest.json"
    protect_outputs([curve_path, metric_path, manifest_path], False)
    expected_rows, expected_labels = None, None
    curves, verified = [], []
    for variant in ORDER:
        predictions = pd.read_csv(prediction_paths[variant])
        required = {"row_id", "split", "peyer_label", "peyer_probability"}
        if not required.issubset(predictions.columns) or len(predictions) != 819 or predictions.row_id.nunique() != 819:
            raise RuntimeError(f"{variant}: expected 819 unique frozen test rows")
        if set(predictions.split.astype(str)) != {"test"}:
            raise RuntimeError("Only the frozen test split is permitted")
        predictions = predictions.sort_values("row_id").reset_index(drop=True)
        rows, labels = predictions.row_id.astype(str).tolist(), predictions.peyer_label.astype(int).tolist()
        if expected_rows is None:
            expected_rows, expected_labels = rows, labels
        elif rows != expected_rows or labels != expected_labels:
            raise RuntimeError("Variants do not use identical held-out cells and labels")
        target, score = predictions.peyer_label.astype(int), predictions.peyer_probability.astype(float)
        if int(target.sum()) != 91:
            raise RuntimeError("Expected 91 positive cells")
        fpr, tpr, thresholds = roc_curve(target, score)
        auroc, ap = float(roc_auc_score(target, score)), float(average_precision_score(target, score))
        saved = table.loc[table.variant.eq(variant)].iloc[0]
        if abs(auroc - float(saved.test_auroc)) > 1e-12 or abs(ap - float(saved.test_average_precision)) > 1e-12:
            raise RuntimeError("Prepared metrics differ from the frozen scientific release")
        curves.append(pd.DataFrame({"variant": variant, "fpr": fpr, "tpr": tpr, "threshold": thresholds}))
        verified.append({"variant": variant, "n": len(predictions), "positive_n": int(target.sum()),
                         "prevalence": float(target.mean()), "auroc": auroc, "average_precision": ap})
    pd.concat(curves, ignore_index=True).to_csv(curve_path, sep="\t", index=False)
    table.merge(pd.DataFrame(verified), on="variant", how="left", validate="one_to_one").to_csv(metric_path, sep="\t", index=False)
    write_provenance(manifest_path, script=Path(__file__),
        inputs=[file_record(release_path, "frozen_classifier_release"), file_record(label_lock, "frozen_label_lock"),
                file_record(summary_path, "frozen_scientific_metrics"),
                *[file_record(model_paths[name], name + "_checkpoint") for name in ORDER],
                *[file_record(prediction_paths[name], name + "_frozen_test_predictions") for name in ORDER]],
        outputs=[curve_path, metric_path], overwritten=False,
        extra={"status": "FROZEN_ROC_DISPLAY_TABLES_V1", "operation": "bounded deterministic display-table conversion",
               "variant_order": list(ORDER), "selected_variant": release["selected_variant"], "threshold": .5,
               "heldout_cells": 819, "positive_cells": 91, "verified_metrics": verified,
               "curve_table": file_record(curve_path, "frozen_ROC_coordinates"),
               "display_metrics": file_record(metric_path, "frozen_display_metrics"),
               "training_inference_selection_or_permutation_run": False})
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.release_manifest, args.predictions_root.expanduser().resolve(), args.output_dir.expanduser().resolve()))


if __name__ == "__main__":
    main()
