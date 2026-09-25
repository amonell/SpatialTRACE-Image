"""Time raw-image preparation through completed shard writes and provenance.

Each backend runs in a fresh process. The operating-system file cache is not
reset. No model fitting, GPU work, segmentation, or image conversion is timed.
"""
import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from crypt_villus_vit.provenance import file_record
from crypt_villus_vit.sources import load_source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--cells-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kind", choices=("pretraining", "supervised"), default="supervised")
    parser.add_argument("--max-cells", default=1024, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--reference-workers", default=0, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--backends", nargs="+", choices=("reference", "cached", "rust"),
                        default=["reference", "cached", "rust"])
    parser.add_argument("--repeats", default=1, type=int)
    args = parser.parse_args()
    if args.max_cells <= 0 or args.repeats <= 0:
        parser.error("max-cells and repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = pd.read_csv(args.cells_csv)
    if len(rows) > args.max_cells:
        rows = rows.iloc[np.linspace(0, len(rows) - 1, args.max_cells, dtype=int)]
    # Labels have no role in this throughput benchmark.
    rows = rows[[key for key in ("cell_id", "source_id", "centroid_x_fullres_px",
                                "centroid_y_fullres_px") if key in rows]].copy()
    if "cell_id" not in rows:
        rows.insert(0, "cell_id", [f"benchmark_{i}" for i in range(len(rows))])
    cells = args.output_dir / "cells.csv"
    rows.to_csv(cells, index=False)
    sources = load_source_manifest(args.source_manifest)
    records = [dict(source_id=sid, image_path=str(sources[sid].image_path.resolve()),
                    pixel_size_um=sources[sid].pixel_size_um, channel_index=sources[sid].channel_index,
                    section_id=sources[sid].section_id) for sid in rows.source_id.astype(str).unique()]
    manifest = args.output_dir / "sources.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    root = Path(__file__).resolve().parents[1]
    report = dict(timestamp_utc=datetime.now(UTC).isoformat(), kind=args.kind, cells=len(rows),
                  python=platform.python_version(), platform=platform.platform(),
                  cache_scope="Fresh process; empty decoded-tile cache. OS cache uncontrolled.",
                  timing_scope="CLI startup, cropping, shard writes and fsync, manifests and provenance hashes.",
                  inputs=[file_record(cells), file_record(manifest)],
                  code=[file_record(root / path) for path in (
                      "src/crypt_villus_vit/prepare_crops.py", "src/crypt_villus_vit/predict.py",
                      "src/crypt_villus_vit/fast_crops.py", "src/crypt_villus_vit/production_crops.py",
                      "rust_preprocess/src/lib.rs", "uv.lock")], runs=[])
    baseline = None
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    for repeat in range(args.repeats):
        for backend in args.backends if repeat % 2 == 0 else list(reversed(args.backends)):
            output = args.output_dir / f"{backend}_{repeat}"
            workers = args.reference_workers if backend == "reference" else args.workers
            command = [sys.executable, "-m", "crypt_villus_vit.cli", f"prepare-{args.kind}",
                       "--source-manifest", str(manifest), "--cells-csv", str(cells),
                       "--output-dir", str(output), "--crop-backend", backend,
                       "--num-workers", str(workers), "--batch-size", str(args.batch_size)]
            print(f"Starting {args.kind} {backend}: {len(rows):,} cells, {workers} workers", flush=True)
            started = time.perf_counter()
            with (args.output_dir / f"{backend}_{repeat}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            wall = time.perf_counter() - started
            provenance = json.loads((output / "run_provenance.json").read_text())
            identities = {Path(record["path"]).name: record["sha256"] for record in provenance["outputs"]
                          if record["path"].endswith(".npy") or Path(record["path"]).name == "manifest.csv"}
            if baseline is None:
                baseline = identities
                report["comparison_baseline"] = dict(backend=backend, repeat=repeat)
            if identities != baseline:
                raise AssertionError("Shard or manifest bytes differ from the comparison baseline")
            metadata = json.loads((output / "metadata.json").read_text())
            record = dict(backend=backend, workers=workers, repeat=repeat, command=command,
                          wall_seconds=wall, cells_per_second=len(rows) / wall,
                          crop_and_write_seconds=metadata["preparation_seconds"],
                          shard_bytes=sum(item["bytes"] for item in provenance["outputs"]
                                          if item["path"].endswith(".npy")),
                          identical_shard_and_manifest_bytes=True,
                          provenance=file_record(output / "run_provenance.json"))
            report["runs"].append(record)
            (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2))
            print(f"{backend}: {wall:.2f}s, {len(rows) / wall:.1f} cells/s; exact shard agreement", flush=True)


if __name__ == "__main__":
    main()
