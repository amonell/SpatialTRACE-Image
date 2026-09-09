from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GATE_SPLIT_X = 0.28
DEFAULT_GATE_SPLIT_Y = 0.30
DEFAULT_MUSCULARIS_SPLIT_X = 0.50

GATE_ORDER = (
    "Top IE",
    "Top LP",
    "Crypt IE",
    "Crypt LP",
    "Muscularis",
)

GATE_COLORS = {
    "Top IE": "#3A9AB2",
    "Top LP": "#A5C881",
    "Crypt IE": "#F26B5B",
    "Crypt LP": "#BDC881",
    "Muscularis": "#C6A87A",
}


def gate_membership(
    epithelial_distance: np.ndarray,
    axis_fraction: np.ndarray,
    *,
    split_x: float = DEFAULT_GATE_SPLIT_X,
    split_y: float = DEFAULT_GATE_SPLIT_Y,
    muscularis_split_x: float = DEFAULT_MUSCULARIS_SPLIT_X,
) -> dict[str, np.ndarray]:
    x = np.asarray(epithelial_distance, dtype=np.float64)
    y = np.asarray(axis_fraction, dtype=np.float64)
    left = x <= float(split_x)
    top = y >= float(split_y)
    bottom_right = (~left) & (~top)
    muscularis_cutoff = min(1.0, max(float(split_x), float(muscularis_split_x)))
    return {
        "Top IE": left & top,
        "Top LP": (~left) & top,
        "Crypt IE": left & (~top),
        "Crypt LP": bottom_right & (x <= muscularis_cutoff),
        "Muscularis": bottom_right & (x > muscularis_cutoff),
    }


def assign_axis_epithelial_gates(
    table: pd.DataFrame,
    *,
    axis_column: str = "predicted_axis_coordinate",
    epithelial_column: str = "predicted_epithelial_distance_clipped_1p0",
    output_column: str = "predicted_gate_name",
    split_x: float = DEFAULT_GATE_SPLIT_X,
    split_y: float = DEFAULT_GATE_SPLIT_Y,
    muscularis_split_x: float = DEFAULT_MUSCULARIS_SPLIT_X,
) -> pd.DataFrame:
    """Attach public Top/Crypt IE/LP/Muscularis gate labels to prediction rows."""

    missing = [column for column in (axis_column, epithelial_column) if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required prediction column(s): {missing}")
    membership = gate_membership(
        table[epithelial_column].to_numpy(dtype=np.float64),
        table[axis_column].to_numpy(dtype=np.float64),
        split_x=split_x,
        split_y=split_y,
        muscularis_split_x=muscularis_split_x,
    )
    gate_names = np.empty(len(table), dtype=object)
    gate_names[:] = "unassigned"
    for gate_name in GATE_ORDER:
        gate_names[membership[gate_name]] = gate_name
    out = table.copy()
    out[output_column] = gate_names
    return out


def summarize_gate_percentages(
    table: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("source_id",),
    gate_column: str = "predicted_gate_name",
) -> pd.DataFrame:
    if gate_column not in table.columns:
        raise KeyError(f"Missing gate column `{gate_column}`.")
    rows: list[dict[str, object]] = []
    grouped = table.groupby(list(group_columns), dropna=False, sort=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        total = int(len(group))
        counts = group[gate_column].astype(str).value_counts()
        for gate in GATE_ORDER:
            count = int(counts.get(gate, 0))
            row = {column: value for column, value in zip(group_columns, group_key)}
            row.update(
                {
                    "gate_name": gate,
                    "count": count,
                    "total_count": total,
                    "percent": 100.0 * float(count) / max(total, 1),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
