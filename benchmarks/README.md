# Raw-image benchmarks

Run from the repository with the environment described in
[Fast inference](../docs/fast_inference.md):

```bash
uv run --locked --extra cu126 --extra rust python benchmarks/benchmark_raw_inference.py \
  --source-manifest sources.csv --cells-csv cells.csv \
  --checkpoint weights/spatialtrace-image-xenium-v1.pt \
  --output-dir runs/benchmark --max-cells 1024 --repeats 2
```

The script samples cells evenly through the input table, drops annotation and
prediction columns, and runs inference in fresh subprocesses. It never loads
prepared crop arrays or changes the source image. Each output directory must be
new. Use `--reference-workers 4` for a worker-matched comparison.

The operating-system file cache is uncontrolled. These are raw-input benchmarks
with empty application caches, not cold-disk measurements. Cell detection and
conversion from unsupported image formats are outside the timed operation.

Every run writes its predictions, input provenance, a log, and timings. The
report compares predictions with the first backend, using `atol=1e-6, rtol=1e-5`,
and requires identical coordinate gates or classifier calls. Keep the reference
backend first when testing changes to preprocessing. Default backend order is
reversed on alternating repeats.

The benchmark table and cell-level outputs stay in the ignored `runs/` folder.
Do not commit microscopy data or per-cell results.

## Training-shard preparation

Run from the repository root:

```bash
uv run --locked --extra cpu --extra rust python benchmarks/benchmark_preparation.py \
  --source-manifest sources.csv --cells-csv cells.csv \
  --output-dir runs/preparation_benchmark --max-cells 1024
```

This compares serial reference cropping with eight-worker cached and Rust
cropping. It includes CLI startup, crop extraction, completed shard writes,
and provenance checksums. Every shard and the cell manifest must match the
first run byte-for-byte. It does not train a model.

Use `--kind pretraining` for local/context pairs; the default is supervised
local/context/fine crops. Without Rust installed, add
`--backends reference cached` and omit `--extra rust`.
Add `--reference-workers 8` for a worker-matched comparison or `--repeats 2`
to repeat in reversed backend order.

Each backend starts a fresh process and an empty decoded-tile cache. The OS
file cache is not cleared. Large runs create a separate shard copy for every
backend and repeat; check disk space first.
