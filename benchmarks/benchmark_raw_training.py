"""Benchmark actual supervised training from raw images, without saved shards.

Synthetic coordinate targets isolate software throughput from biological model
evaluation. All modes use identical rows, splits, initialization, batches, and
training settings. Do not use the fitted benchmark checkpoints as paper models.
"""
import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from crypt_villus_vit.provenance import file_record
from crypt_villus_vit.sources import load_source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--cells-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-cells", default=512, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--reference-workers", default=0, type=int)
    parser.add_argument("--ram-crop-cache-mib", default=8192, type=int)
    parser.add_argument("--tile-cache-mib", default=256, type=int)
    parser.add_argument("--modes", nargs="+", choices=("reference", "cached", "cached_ram", "rust_ram"),
                        default=["reference", "cached", "cached_ram"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_cells < 8:
        parser.error("At least eight cells are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = pd.read_csv(args.cells_csv)
    if len(rows) > args.max_cells:
        rows = rows.iloc[np.linspace(0, len(rows) - 1, args.max_cells, dtype=int)]
    rows = rows[[c for c in ("cell_id", "source_id", "centroid_x_fullres_px", "centroid_y_fullres_px")
                 if c in rows]].copy().reset_index(drop=True)
    # Deliberately synthetic labels: no accuracy or generalization claim.
    for column, axis in (("target_axis", "centroid_y_fullres_px"),
                         ("epithelial_distance_clipped_1p0", "centroid_x_fullres_px")):
        values = rows[axis].to_numpy(float)
        rows[column] = (values - values.min()) / max(float(np.ptp(values)), 1.0)
    rows["split"] = np.where(np.arange(len(rows)) % 8 == 0, "validation", "train")
    labels = args.output_dir / "labels.csv"
    rows.to_csv(labels, index=False)
    sources = load_source_manifest(args.source_manifest)
    manifest = args.output_dir / "sources.csv"
    pd.DataFrame([dict(source_id=sid, image_path=str(sources[sid].image_path.resolve()),
                       pixel_size_um=sources[sid].pixel_size_um, channel_index=sources[sid].channel_index)
                  for sid in rows.source_id.unique()]).to_csv(manifest, index=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    extra = []
    for key in ("local_crop_px", "context_crop_px", "fine_crop_px", "input_size_px", "fine_input_size_px",
                "patch_size_px", "embed_dim", "depth", "num_heads", "local_readout", "context_readout",
                "encoder_architecture"):
        extra += ["--" + key.replace("_", "-"), str(config[key])]
    for branch in ("local", "context", "fine"):
        if not config[f"use_{branch}_branch"]:
            extra += [f"--no-{branch}-branch"]
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    report = dict(timestamp_utc=datetime.now(UTC).isoformat(), cells=len(rows), epochs=args.epochs,
                  train_cells=int(rows.split.eq("train").sum()), validation_cells=int(rows.split.eq("validation").sum()),
                  batch_size=args.batch_size, device=args.device, checkpoint=file_record(args.checkpoint),
                  labels=file_record(labels), targets="Synthetic normalized x/y, solely for throughput testing.",
                  cache_scope="Fresh process for each mode; no prepared shards. OS file cache uncontrolled.",
                  timing_scope="CLI startup, optional RAM warmup, all training/validation epochs, checkpoints and provenance.",
                  code=[file_record(root / path) for path in (
                      "src/crypt_villus_vit/raw_training.py", "src/crypt_villus_vit/train.py",
                      "src/crypt_villus_vit/predict.py", "src/crypt_villus_vit/fast_crops.py", "uv.lock")], runs=[])
    baseline_history = baseline_weights = baseline_summary = None
    for mode in args.modes:
        backend = "rust" if mode == "rust_ram" else "reference" if mode == "reference" else "cached"
        workers = args.reference_workers if mode == "reference" else args.workers
        output = args.output_dir / mode
        command = [sys.executable, "-m", "crypt_villus_vit.cli", "--threads", "1", "train",
                   "--source-manifest", str(manifest), "--supervised-manifest", str(labels),
                   "--output-dir", str(output), "--initial-checkpoint", str(args.checkpoint),
                   "--device", args.device, "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
                   "--seed", "17", "--num-workers", str(workers), "--crop-backend", backend,
                   "--tile-cache-mib", str(args.tile_cache_mib), *extra]
        if mode.endswith("_ram"):
            command += ["--ram-crop-cache-mib", str(args.ram_crop_cache_mib)]
        print(f"Starting {mode}: {len(rows):,} cells, {args.epochs} epochs, {workers} workers", flush=True)
        started = time.perf_counter()
        with (args.output_dir / f"{mode}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        wall = time.perf_counter() - started
        summary = json.loads((output / "training_summary.json").read_text())
        history = pd.read_csv(output / "history.csv")
        weights = torch.load(output / "best_crypt_villus_vit_model.pt", map_location="cpu", weights_only=True)["model_state_dict"]
        if list(output.glob("*.npy")):
            raise AssertionError("Raw training wrote crop shards")
        compared = baseline_history is not None
        if not compared:
            baseline_history, baseline_weights, baseline_summary = history, weights, summary
            report["comparison_baseline"] = mode
        metrics = [c for c in history if not c.endswith("_seconds")]
        differences = {c: float(np.max(np.abs(history[c] - baseline_history[c]))) for c in metrics}
        max_weight_difference = max(float(torch.max(torch.abs(weights[key] - baseline_weights[key]))) for key in weights)
        # Report numerical drift instead of calling approximate agreement exact.
        np.testing.assert_allclose(history[metrics], baseline_history[metrics], atol=1e-6, rtol=1e-5)
        if summary["best_epoch"] != baseline_summary["best_epoch"]:
            raise AssertionError("Checkpoint-selection epoch changed")
        record = dict(mode=mode, workers=workers, command=command, wall_seconds=wall,
                      cache_warmup_seconds=summary["cache_warmup_seconds"],
                      epochs=history.to_dict(orient="records"), data_loading=summary["data_loading"],
                      best_epoch=summary["best_epoch"], compared_to_baseline=compared,
                      max_abs_metric_differences=differences, max_abs_selected_weight_difference=max_weight_difference,
                      no_prepared_shards=True, provenance=file_record(output / "run_provenance.json"))
        report["runs"].append(record)
        (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2))
        print(f"{mode}: {wall:.2f}s total; warmup {summary['cache_warmup_seconds']:.2f}s; "
              f"train epochs {history.train_seconds.round(2).tolist()}; weight difference {max_weight_difference:.3g}", flush=True)


if __name__ == "__main__":
    main()
