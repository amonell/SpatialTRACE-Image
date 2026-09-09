from __future__ import annotations

import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_PIXEL_X_CANDIDATES = (
    "centroid_x_fullres_px",
    "centroid_x_px",
    "centroid_x_pixels",
    "centroid_x_pixel",
    "centroid_x",
    "x_centroid_px",
)
_PIXEL_Y_CANDIDATES = (
    "centroid_y_fullres_px",
    "centroid_y_px",
    "centroid_y_pixels",
    "centroid_y_pixel",
    "centroid_y",
    "y_centroid_px",
)
_MICRON_X_CANDIDATES = (
    "centroid_x_um",
    "centroid_x_microns",
    "centroid_x_micron",
)
_MICRON_Y_CANDIDATES = (
    "centroid_y_um",
    "centroid_y_microns",
    "centroid_y_micron",
)
_CLASS_COLUMN_CANDIDATES = (
    "class",
    "classification",
    "classification_name",
    "path_class",
    "pathclass",
)


@dataclass(frozen=True)
class QuPathCentroidImportSummary:
    qupath_input_path: str
    input_row_count: int
    selected_row_count: int
    dropped_non_finite_count: int
    class_column: str | None
    requested_class_filter: str
    class_match_mode: str
    coordinate_unit: str
    x_column: str
    y_column: str
    available_classes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_header(value: object) -> str:
    text = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _slugify_token(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "cell"


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep).reset_index(drop=True)


def _resolve_column(table: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized_to_original = {_normalize_header(column): str(column) for column in table.columns}
    for candidate in candidates:
        resolved = normalized_to_original.get(candidate)
        if resolved is not None:
            return resolved
    return None


def _resolve_coordinate_columns(table: pd.DataFrame) -> tuple[str, str, str]:
    pixel_x = _resolve_column(table, _PIXEL_X_CANDIDATES)
    pixel_y = _resolve_column(table, _PIXEL_Y_CANDIDATES)
    if pixel_x is not None and pixel_y is not None:
        return pixel_x, pixel_y, "px"
    micron_x = _resolve_column(table, _MICRON_X_CANDIDATES)
    micron_y = _resolve_column(table, _MICRON_Y_CANDIDATES)
    if micron_x is not None and micron_y is not None:
        return micron_x, micron_y, "um"
    raise ValueError(
        "Unable to locate QuPath centroid columns. Expected pixel columns such as "
        "`centroid_x_px`/`centroid_y_px`, or micron columns such as "
        "`centroid_x_um`/`centroid_y_um`."
    )


def _resolve_class_column(table: pd.DataFrame) -> str | None:
    return _resolve_column(table, _CLASS_COLUMN_CANDIDATES)


def _class_mask(classes: pd.Series, *, class_filter: str, class_match_mode: str) -> np.ndarray:
    requested = str(class_filter).strip()
    if not requested:
        return np.ones(len(classes), dtype=bool)
    values = classes.fillna("").astype(str).str.strip()
    mode = str(class_match_mode).strip().lower() or "exact"
    if mode == "exact":
        return values.str.casefold().eq(requested.casefold()).to_numpy(dtype=bool)
    if mode == "contains":
        return values.str.casefold().str.contains(requested.casefold(), regex=False).to_numpy(dtype=bool)
    if mode == "regex":
        return values.str.contains(requested, flags=re.IGNORECASE, regex=True, na=False).to_numpy(dtype=bool)
    raise ValueError(f"Unsupported class match mode `{class_match_mode}`.")


def list_qupath_classes(qupath_input_path: Path) -> list[str]:
    table = _read_table(Path(qupath_input_path))
    class_column = _resolve_class_column(table)
    if class_column is None:
        return []
    return sorted(
        {
            str(value).strip()
            for value in table[class_column].fillna("").astype(str).tolist()
            if str(value).strip()
        }
    )


def load_qupath_centroid_rows(
    *,
    qupath_input_path: Path,
    source_id: str,
    class_filter: str = "",
    class_match_mode: str = "exact",
    row_id_prefix: str = "cell",
    pixel_size_um: float | None = None,
) -> tuple[pd.DataFrame, QuPathCentroidImportSummary]:
    table = _read_table(Path(qupath_input_path))
    input_row_count = int(len(table))
    class_column = _resolve_class_column(table)
    available_classes = tuple(list_qupath_classes(Path(qupath_input_path)))
    if str(class_filter).strip():
        if class_column is None:
            raise ValueError(f"Class filter `{class_filter}` was requested, but no class column was found.")
        selected = table.loc[
            _class_mask(table[class_column], class_filter=class_filter, class_match_mode=class_match_mode)
        ].copy()
        if selected.empty:
            raise ValueError(
                f"No rows matched class filter `{class_filter}`. Available classes: {list(available_classes)}"
            )
    else:
        selected = table.copy()
    selected = selected.assign(qupath_input_row_index=selected.index.to_numpy(dtype=np.int64)).reset_index(drop=True)
    x_column, y_column, coordinate_unit = _resolve_coordinate_columns(selected)
    selected["center_x_fullres_px"] = pd.to_numeric(selected[x_column], errors="coerce")
    selected["center_y_fullres_px"] = pd.to_numeric(selected[y_column], errors="coerce")
    if coordinate_unit == "um":
        if pixel_size_um is None or float(pixel_size_um) <= 0.0:
            raise ValueError("Micron centroid coordinates require `pixel_size_um` for pixel conversion.")
        selected["center_x_fullres_px"] /= float(pixel_size_um)
        selected["center_y_fullres_px"] /= float(pixel_size_um)
    finite = (
        np.isfinite(selected["center_x_fullres_px"].to_numpy(dtype=np.float64))
        & np.isfinite(selected["center_y_fullres_px"].to_numpy(dtype=np.float64))
    )
    dropped_non_finite_count = int((~finite).sum())
    selected = selected.loc[finite].reset_index(drop=True).copy()
    if selected.empty:
        raise ValueError("No finite centroid rows remained after filtering.")
    prefix = _slugify_token(row_id_prefix or class_filter or "cell")
    selected["row_id"] = [f"{source_id}:qupath:{prefix}:{idx:07d}" for idx in range(len(selected))]
    selected["graph_id"] = selected["row_id"].astype(str)
    selected["source_id"] = str(source_id)
    selected["centroid_x_fullres_px"] = selected["center_x_fullres_px"].to_numpy(dtype=np.float64)
    selected["centroid_y_fullres_px"] = selected["center_y_fullres_px"].to_numpy(dtype=np.float64)
    if class_column is not None:
        selected["qupath_class_name"] = selected[class_column].fillna("").astype(str).str.strip()
    summary = QuPathCentroidImportSummary(
        qupath_input_path=str(qupath_input_path),
        input_row_count=input_row_count,
        selected_row_count=int(len(selected)),
        dropped_non_finite_count=dropped_non_finite_count,
        class_column=class_column,
        requested_class_filter=str(class_filter).strip(),
        class_match_mode=str(class_match_mode).strip().lower() or "exact",
        coordinate_unit=coordinate_unit,
        x_column=str(x_column),
        y_column=str(y_column),
        available_classes=available_classes,
    )
    return selected, summary
