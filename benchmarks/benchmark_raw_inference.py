"""Run each inference backend in a fresh process, starting from raw images.

No saved crops are used. OS file-cache state is not reset: report these as
raw-input, empty application-cache runs, not cold-disk benchmarks.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from crypt_villus_vit.provenance import file_record
from crypt_villus_vit.sources import load_source_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--cells-csv", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-cells", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--reference-workers", type=int, default=0,
                   help="Set equal to --workers for a worker-matched baseline.")
    p.add_argument("--backends", nargs="+", choices=("reference", "cached", "rust"),
                   default=["reference", "cached", "rust"])
    p.add_argument("--repeats", type=int, default=1)
    a = p.parse_args()
    if a.repeats < 1 or a.max_cells < 1:
        p.error("max-cells and repeats must be positive")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    rows = pd.read_csv(a.cells_csv)
    if len(rows) > a.max_cells:
        rows = rows.iloc[np.linspace(0, len(rows) - 1, a.max_cells, dtype=int)]
    # No labels or saved predictions enter the inference benchmark.
    rows = rows[[c for c in ("cell_id", "source_id", "centroid_x_fullres_px",
                              "centroid_y_fullres_px") if c in rows]].copy()
    rows["source_id"] = rows["source_id"].astype(str)
    if "cell_id" not in rows:
        rows.insert(0, "cell_id", [f"benchmark_{i}" for i in range(len(rows))])
    cells = a.output_dir / "cells.csv"
    rows.to_csv(cells, index=False)
    sources = load_source_manifest(a.source_manifest)
    records = []
    for source_id in rows.source_id.unique():
        spec = sources[source_id]
        records.append({"source_id": source_id, "image_path": str(spec.image_path.resolve()),
                        "pixel_size_um": spec.pixel_size_um, "channel_index": spec.channel_index})
    manifest = a.output_dir / "sources.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    report = {"timestamp_utc": datetime.now(UTC).isoformat(),
              "python": platform.python_version(), "checkpoint": file_record(a.checkpoint),
              "cells": file_record(cells), "sources": records,
              "cache_scope": "Fresh process and empty decoded-tile cache per run. OS cache uncontrolled.",
              "packages": {name: version(name) for name in
                           ("torch", "numpy", "tifffile", "imagecodecs", "zarr")},
              "runs": []}
    reference = None
    root = Path(__file__).resolve().parents[1]
    report["code"] = [file_record(root / name) for name in (
        "src/crypt_villus_vit/predict.py", "src/crypt_villus_vit/production_crops.py",
        "src/crypt_villus_vit/fast_crops.py", "rust_preprocess/src/lib.rs",
        "rust_preprocess/Cargo.lock", "uv.lock",
    )]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    for repeat in range(a.repeats):
        # Reverse on alternating repeats to reduce systematic order bias.
        for backend in a.backends if repeat % 2 == 0 else list(reversed(a.backends)):
            destination = a.output_dir / f"{backend}_{repeat}"
            cmd = [sys.executable, "-m", "crypt_villus_vit.cli", "predict-cells",
                   "--source-manifest", str(manifest), "--cells-csv", str(cells),
                   "--checkpoint", str(a.checkpoint), "--output-dir", str(destination),
                   "--device", a.device, "--batch-size", str(a.batch_size),
                   "--crop-backend", backend, "--num-workers",
                   str(a.workers if backend != "reference" else a.reference_workers),
                   "--no-scatter"]
            print(f"Starting {backend} repeat {repeat}: {len(rows):,} cells", flush=True)
            started = time.perf_counter()
            with (a.output_dir / f"{backend}_{repeat}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            wall = time.perf_counter() - started
            result = pd.read_csv(destination / "predictions.csv")
            if result.cell_id.tolist() != rows.cell_id.tolist():
                raise AssertionError("Prediction row identity changed")
            summary = json.loads((destination / "summary.json").read_text())
            columns = ([str(summary["prediction_column"])] if summary["task_type"] == "binary_classification"
                       else ["predicted_axis_coordinate", "predicted_epithelial_distance_clipped_1p0"])
            record = {"backend": backend, "repeat": repeat, "command": cmd, "wall_seconds": wall,
                      "cells_per_second": len(rows) / wall, "timing": summary["timing"],
                      "predictions": file_record(destination / "predictions.csv")}
            if reference is None:
                reference = result
                report["comparison_baseline"] = {"backend": backend, "repeat": repeat}
            if reference is not None:
                record["max_abs_prediction_difference"] = {
                    c: float(np.max(np.abs(reference[c].to_numpy() - result[c].to_numpy()))) for c in columns
                }
                for c in columns:
                    np.testing.assert_allclose(reference[c], result[c], atol=1e-6, rtol=1e-5)
                if "predicted_gate_name" in result:
                    assert result.predicted_gate_name.equals(reference.predicted_gate_name)
                if summary["task_type"] == "binary_classification":
                    probability = columns[0]
                    class_column = (probability.removesuffix("_probability") + "_class"
                                    if probability.endswith("_probability") else probability + "_class")
                    assert result[class_column].equals(reference[class_column])
            report["runs"].append(record)
            (a.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2))
            print(f"{backend}: {wall:.2f}s; {len(rows)/wall:.1f} cells/s; "
                  f"differences={record.get('max_abs_prediction_difference')}", flush=True)


if __name__ == "__main__":
    main()
