"""Validate and expose saved scientific annotations without selecting or fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def training_example_counts(manifest: pd.DataFrame, summary: dict) -> dict[str, int]:
    training = manifest.loc[manifest.split.eq("train")]
    counts = {"cells": len(training), "villi": int(training.villus_shape_id.nunique())}
    if counts["cells"] != int(summary["train_row_count"]) or not all(counts.values()):
        raise RuntimeError("Training workflow counts disagree with the frozen training summary")
    if training.row_id.duplicated().any() or training.villus_shape_id.isna().any():
        raise RuntimeError("Ambiguous training identities in the frozen workflow input")
    return counts


def joined_training_history(frames: list[pd.DataFrame], summaries: list[dict]):
    """Use saved stage lengths and selections; never choose an epoch here."""
    if len(frames) != 2 or len(summaries) != 2:
        raise RuntimeError("Expected the released two-stage IF adaptation workflow")
    joined, offset, selections, lengths = [], 0, [], []
    for index, (frame, summary) in enumerate(zip(frames, summaries, strict=True)):
        frame = frame.copy()
        epochs = frame.epoch.to_numpy()
        if not np.array_equal(epochs, np.arange(1, len(frame) + 1)):
            raise RuntimeError("Saved history has missing, duplicate, or unordered epochs")
        if len(frame) != int(summary["epochs"]):
            raise RuntimeError("Saved history length disagrees with its training summary")
        selected = int(summary["best_epoch"])
        if selected not in epochs:
            raise RuntimeError("Saved selected epoch is absent from its stage history")
        frame["completed_epoch"] = frame.epoch.astype(int) + offset
        frame["stage"] = "Heads only" if index == 0 else "Full model"
        joined.append(frame)
        selections.append(offset + selected)
        lengths.append(len(frame))
        offset += len(frame)
    return pd.concat(joined, ignore_index=True), {
        "completed_epochs": lengths,
        "selected_stage_epochs": [int(s["best_epoch"]) for s in summaries],
        "selected_combined_epochs": selections,
        "stage_boundary": lengths[0] + .5,
    }
