#!/usr/bin/env python3
"""Lock the section-disjoint Figure 2e annotation-scaling design before fitting."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from paper_paths import Path, lock_record, rendering_script
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EXPECTED_BASE_SPLIT_SHA256 = "746c72b9181d33883f43bc9db481c023d02abadfe67346df5b90d18f05c210f2"
EXPECTED_LABELS_SHA256 = "6eabc5f269efcb8c2a263b44016404aa8e34900fa49a1cca1a7b2f447d63be2c"
TRAIN_COUNTS = [1, *range(5, 46, 5)]


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_villus_order(labels: pd.DataFrame, sections: list[str], seed: int) -> list[str]:
    """Shuffle within section, then interleave sections to avoid small-k section imbalance."""
    rng = np.random.default_rng(seed)
    section_order = list(sections)
    rng.shuffle(section_order)
    queues: dict[str, list[str]] = {}
    for section in section_order:
        villi = sorted(labels.loc[labels["batch"].astype(str).eq(section), "training_villus_id"].astype(str).unique())
        rng.shuffle(villi)
        queues[section] = list(villi)
    ordered: list[str] = []
    while any(queues.values()):
        for section in section_order:
            if queues[section]:
                ordered.append(queues[section].pop())
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=530000)
    args = parser.parse_args()

    base_path = args.base_split.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path, expected in ((base_path, EXPECTED_BASE_SPLIT_SHA256), (labels_path, EXPECTED_LABELS_SHA256)):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Missing or hash-mismatched frozen input: {path}")
    if output.exists():
        raise FileExistsError(f"Refusing to alter locked split manifest: {output}")

    base = json.loads(base_path.read_text())
    labels = pd.read_csv(labels_path)
    folds: list[dict[str, object]] = []
    for fold in base["folds"]:
        fold_index = int(fold["fold_index"])
        training_sections = list(map(str, fold["training_sections"]))
        order_seed = int(args.seed + fold_index)
        order = balanced_villus_order(labels, training_sections, order_seed)
        if len(order) < max(TRAIN_COUNTS):
            raise RuntimeError(f"Fold {fold['fold_id']} has only {len(order)} training villi")
        runs = []
        for count_index, count in enumerate(TRAIN_COUNTS):
            selected = order[:count]
            selected_rows = labels[labels["training_villus_id"].astype(str).isin(selected)]
            if set(selected_rows["batch"].astype(str)) - set(training_sections):
                raise RuntimeError("A selected training villus escaped the fold's training sections")
            run_index = fold_index * len(TRAIN_COUNTS) + count_index
            runs.append(
                {
                    "n_train_villi": count,
                    "training_villi": selected,
                    "training_cells": int(len(selected_rows)),
                    "seeds": {
                        "crypt_villus": int(540000 + 2 * run_index),
                        "epithelial_distance": int(540001 + 2 * run_index),
                    },
                }
            )
        validation_rows = labels[labels["batch"].astype(str).eq(str(fold["validation_section"]))]
        test_rows = labels[labels["batch"].astype(str).eq(str(fold["test_section"]))]
        folds.append(
            {
                "fold_id": fold["fold_id"],
                "fold_index": fold_index,
                "training_sections": training_sections,
                "validation_section": fold["validation_section"],
                "test_section": fold["test_section"],
                "available_training_villi": len(order),
                "balanced_villus_order_seed": order_seed,
                "balanced_villus_order": order,
                "validation_cells": int(len(validation_rows)),
                "validation_villi": int(validation_rows["training_villus_id"].nunique()),
                "test_cells": int(len(test_rows)),
                "test_villi": int(test_rows["training_villus_id"].nunique()),
                "runs": runs,
            }
        )

    manifest = {
        "schema_version": 1,
        "split_id": "graph_model_section_disjoint_villus_scaling_v1",
        "status": "LOCKED_BEFORE_MODEL_FITTING_OR_TEST_METRICS",
        "locked_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "purpose": "Figure 2e annotation-scaling curve with complete section separation between training, validation, and test.",
        "base_section_split": {"path": str(base_path), "sha256": sha256(base_path)},
        "source_labels": {"path": str(labels_path), "sha256": sha256(labels_path)},
        "selection_rule": "Reuse the eight prespecified Figure 2d test/validation folds. Within each fold, shuffle villi within each training section and interleave sections using a fixed seed; take prefixes of this order without consulting validation or test performance.",
        "training_villus_counts": TRAIN_COUNTS,
        "aggregation_unit": "held-out tissue section",
        "fold_count_per_training_count": len(folds),
        "model_budget": base["model_budget"],
        "feature_lineage": base["feature_lineage"],
        "graph_rule": base["graph_rule"],
        "folds": folds,
        "blindness_statement": "All folds, training-villus subsets, seeds, and the common maximum of 45 training villi were fixed before any new model was fitted or any new test metric was computed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)
    print(sha256(output))


if __name__ == "__main__":
    main()
