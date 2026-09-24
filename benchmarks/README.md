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
