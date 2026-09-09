from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def predictions_to_qupath_geojson(
    predictions_csv: Path,
    output_path: Path,
    *,
    x_column: str = "centroid_x_fullres_px",
    y_column: str = "centroid_y_fullres_px",
) -> Path:
    table = pd.read_csv(predictions_csv)
    missing = [column for column in (x_column, y_column) if column not in table.columns]
    if missing:
        raise ValueError(f"Prediction table is missing coordinate column(s): {missing}")
    features = []
    for row in table.to_dict(orient="records"):
        properties = {
            "classification": {"name": str(row.get("predicted_gate_name", "prediction"))},
            "measurements": {
                "predicted_axis_coordinate": float(row["predicted_axis_coordinate"]),
                "predicted_epithelial_distance_clipped_1p0": float(
                    row["predicted_epithelial_distance_clipped_1p0"]
                ),
            },
        }
        if "row_id" in row:
            properties["name"] = str(row["row_id"])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row[x_column]), float(row[y_column])],
                },
                "properties": properties,
            }
        )
    payload = {"type": "FeatureCollection", "features": features}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload))
    return output_path
