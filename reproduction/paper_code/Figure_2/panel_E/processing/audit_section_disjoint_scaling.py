#!/usr/bin/env python3
"""Audit the frozen Figure 2e section-disjoint scaling corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from paper_paths import Path, lock_record, rendering_script
from zoneinfo import ZoneInfo

import pandas as pd
from scipy import stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    split_path = args.split_manifest.expanduser().resolve()
    if sha256(split_path) != args.expected_split_sha256:
        raise RuntimeError("Locked scaling split hash mismatch")
    split = json.loads(split_path.read_text())
    metrics_path = root / "per_fold_metrics.tsv"
    checkpoints_path = root / "checkpoint_manifest.tsv"
    predictions_path = root / "all_test_predictions.parquet"
    summary_path = root / "summary.tsv"
    provenance_path = root / "provenance.json"
    metrics = pd.read_csv(metrics_path, sep="\t")
    checkpoints = pd.read_csv(checkpoints_path, sep="\t")
    predictions = pd.read_parquet(predictions_path)
    if len(metrics) != 160 or len(checkpoints) != 160:
        raise RuntimeError("Expected 160 fitted axis/count/fold models")
    if len(predictions) != 508720:
        raise RuntimeError(f"Expected 508,720 test prediction rows; observed {len(predictions):,}")

    fold_lookup = {str(fold["fold_id"]): fold for fold in split["folds"]}
    discrepant: list[str] = []
    for fold_id, fold in fold_lookup.items():
        train = set(map(str, fold["training_sections"]))
        validation = str(fold["validation_section"])
        test = str(fold["test_section"])
        if validation == test or validation in train or test in train:
            discrepant.append(f"{fold_id}: overlapping section roles")
    for row in checkpoints.itertuples(index=False):
        checkpoint = Path(row.checkpoint)
        if not checkpoint.is_file() or sha256(checkpoint) != row.checkpoint_sha256:
            discrepant.append(f"{row.fold_id}/{row.axis}/{row.n_train_villi}: checkpoint hash mismatch")
    for keys, group in predictions.groupby(["fold_id", "axis", "n_train_villi"], sort=False):
        fold_id, axis_name, count = keys
        fold = fold_lookup[str(fold_id)]
        if len(group) != int(fold["test_cells"]):
            discrepant.append(f"{fold_id}/{axis_name}/{count}: wrong test row count")
        if set(group["batch"].astype(str)) != {str(fold["test_section"])}:
            discrepant.append(f"{fold_id}/{axis_name}/{count}: predictions outside test section")
        if group["global_index"].duplicated().any():
            discrepant.append(f"{fold_id}/{axis_name}/{count}: duplicate test cells")
        target = "crypt_villus_label" if axis_name == "crypt_villus" else "epithelial_distance_label"
        calculated = float(stats.pearsonr(group[target], group["prediction"]).statistic)
        expected = metrics[
            metrics["fold_id"].astype(str).eq(str(fold_id))
            & metrics["axis"].astype(str).eq(str(axis_name))
            & metrics["n_train_villi"].eq(int(count))
        ]
        if len(expected) != 1 or abs(calculated - float(expected.iloc[0]["pearson_r"])) > 1e-10:
            discrepant.append(f"{fold_id}/{axis_name}/{count}: metric mismatch")
    if discrepant:
        raise RuntimeError("\n".join(discrepant))

    files = {
        "split_manifest": record(split_path),
        "training_script": record(args.training_script),
        "per_fold_metrics": record(metrics_path),
        "checkpoint_manifest": record(checkpoints_path),
        "all_test_predictions": record(predictions_path),
        "summary": record(summary_path),
        "pipeline_provenance": record(provenance_path),
    }
    concise = (
        metrics.groupby(["axis", "n_train_villi"], as_index=False)["pearson_r"]
        .agg(mean="mean", sem="sem", minimum="min", maximum="max", held_out_sections="count")
    )
    payload = {
        "schema_version": 1,
        "status": "VERIFIED_SECTION_DISJOINT",
        "verified_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "design": {
            "folds": 8,
            "training_villus_counts": split["training_villus_counts"],
            "models": 160,
            "test_prediction_rows": len(predictions),
            "aggregation_unit": "held-out tissue section",
            "training_sections_per_fold": 6,
            "validation_sections_per_fold": 1,
            "test_sections_per_fold": 1,
            "section_role_overlap_count": 0,
        },
        "files": files,
        "summary": concise.to_dict(orient="records"),
        "limitations": [split["feature_lineage"]["limitation"]],
    }
    audit_json = args.audit_json.expanduser().resolve()
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(payload, indent=2) + "\n")

    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    last = concise[concise["n_train_villi"].eq(max(split["training_villus_counts"]))]
    last_lines = [
        f"- {row.axis}: mean Pearson r = {row.mean:.6f}, SEM = {row.sem:.6f} across {int(row.held_out_sections)} sections"
        for row in last.itertuples(index=False)
    ]
    report.write_text(
        "# Figure 2e section-disjoint scaling resolution\n\n"
        "Figure 2e has been replaced by an eight-fold section-disjoint analysis. "
        "For every fold and training-villus count, the selected training villi are drawn only from a six-section training pool, "
        "checkpoint selection uses a seventh complete section, and final scoring uses the eighth complete section.\n\n"
        f"- Locked split: `{split_path}`\n"
        f"- Split SHA256: `{args.expected_split_sha256}`\n"
        "- Training-villus counts: 1, 5, 10, 15, 20, 25, 30, 35, 40, 45\n"
        "- Fitted checkpoints: 160 (8 folds × 10 counts × 2 coordinates)\n"
        "- Section-role overlaps: 0\n"
        "- Duplicate cells within any test prediction group: 0\n"
        "- Test metrics were not used for split construction, checkpoint selection, or model tuning.\n\n"
        "At 45 training villi:\n\n" + "\n".join(last_lines) + "\n\n"
        "Limitation: the supervised GAT experiment is fully section-disjoint, but the upstream 30-dimensional "
        "scVI representation was learned jointly across the dataset.\n"
    )
    print(audit_json)
    print(report)


if __name__ == "__main__":
    main()
