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

### Measured preparation times

September 24, 2026: one 34,063 × 34,137-pixel JPEG2000-compressed Xenium
DAPI image, Threadripper PRO 5995WX, no GPU. Preparation batches contained
128 cells; tile caches were 256 MiB per worker. Outputs were written to NVMe.

| Crops | Cells | Backend | Workers | Total time |
| --- | ---: | --- | ---: | ---: |
| Supervised | 1,024 | Reference | 0 | 320.5 s |
| Supervised | 1,024 | Cached | 8 | 20.8 s |
| Supervised | 1,024 | Cached + Rust | 8 | 19.3 s |
| Supervised | 50,000 | Cached | 8 | 96.6 s |
| Supervised | 50,000 | Cached + Rust | 8 | 98.0 s |
| Pretraining pairs | 256 | Reference | 0 | 55.7 s |
| Pretraining pairs | 256 | Cached | 8 | 16.2 s |
| Pretraining pairs | 256 | Cached + Rust | 8 | 16.2 s |
| Pretraining pairs | 50,000 | Cached + Rust | 8 | 66.1 s |

The reference uses the original crop routine; all runs use the same new shard
writer. These are single runs on a shared workstation, not cold-disk timings.
The small samples span the tissue; the full table has greater tile reuse.
The full 50,000-cell serial runs were not timed.

On the small comparisons, every shard and manifest matched the reference
byte-for-byte. All supervised shards also matched between cached and Rust
backends at 50,000 cells. The full outputs matched the serial reference at
1,024 supervised cells and 256 pretraining centers checked within those runs.

The large outputs contain 7.4 GB of supervised crops or 6.6 GB of pretraining
pairs. Their timings include writing and checksumming those files. They do not
include model training. The gains combine parallel loading, tile caching, and
spatial read ordering. Rust normalization added little on this image.

Exact measurements and check results are in
[`preparation_results_2026_09_24.json`](preparation_results_2026_09_24.json).

## Training directly from raw images

```bash
uv run --locked --extra cu126 python benchmarks/benchmark_raw_training.py \
  --source-manifest sources.csv --cells-csv cells.csv \
  --checkpoint weights/spatialtrace-image-xenium-v1.pt \
  --output-dir runs/raw_training_benchmark --max-cells 512 --epochs 3
```

This runs real optimizer steps and validation, starting each mode from the same
checkpoint. It compares the original loader with cached loading and an optional
shared RAM crop cache. No prepared shards are read or written. Targets are
synthetic coordinates derived from cell positions; fitted benchmark models are
for timing checks, not biological use.

Total time includes startup, RAM warmup, training, validation, checkpoint export,
and provenance. The report also separates warmup and individual epoch times.
It compares loss histories, selected epochs, and saved weights. Spatial ordering
is used only when filling RAM slots; optimizer batches keep the original order.

The default comparison uses zero reference workers and four optimized workers.
Add `--reference-workers 4` for a worker-matched comparison. Each mode runs in a
fresh process; the OS file cache is uncontrolled.

### Measured raw-training times

September 24, 2026: 512 cells from the same compressed DAPI image used above,
with 448 training cells and 64 validation cells. All modes used three epochs,
batch size 32, the released coordinate-model initialization, and full-parameter
training on an RTX A6000. Targets were synthetic. The workstation CPU was a
Threadripper PRO 5995WX.

| Loader | Workers | Total, three epochs | Third training epoch |
| --- | ---: | ---: | ---: |
| Original | 0 | 476.5 s | 132.9 s |
| Original | 4 | 151.0 s | 38.8 s |
| Cached tiles, persistent workers | 4 | 114.8 s | 31.0 s |
| Cached tiles + shared RAM crops | 4 | 33.1 s | 1.18 s |

The RAM-cached total includes 20.0 seconds to fill the cache. All 512 rows fit
in about 72 MiB; the configured 8 GiB ceiling did not allocate unused space.
The result was 14.4× faster than the serial loader and 4.6× faster than the
original four-worker loader. These measurements apply to this small test on
one shared workstation. Larger runs depend on cache capacity and model time.

All modes selected epoch 3. GPU loss differences were at most 7.5 × 10⁻⁹ and
metric differences at most 3.8 × 10⁻⁸. Selected GPU weights differed by up to
2.3 × 10⁻⁵, so this is numerical agreement, not bitwise identity. Separate CPU
tests matched batches, augmented crops, losses, and selected weights exactly.

Measurements and checks are recorded in
[`raw_training_results_2026_09_24.json`](raw_training_results_2026_09_24.json).
