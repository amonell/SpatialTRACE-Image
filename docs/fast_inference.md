# Faster inference from raw images

Use decoded-tile caching and spatially ordered processing to predict directly
from raw images. No saved crops are needed.
The trained model and its preprocessing settings stay the same.

## Install

From this repository:

```bash
uv sync --locked --extra cu126
```

Use `--extra cpu` instead of `--extra cu126` for CPU inference.

## Predict

Supply the original calibrated image and cell centroids, as described in
[Prepare your images](apply_to_own_data.md):

```bash
uv run --locked --extra cu126 spatialtrace-image predict-cells \
  --source-manifest sources.csv --cells-csv cells.csv \
  --checkpoint weights/spatialtrace-image-xenium-v1.pt \
  --output-dir runs/fast_predictions --device cuda \
  --crop-backend cached --num-workers 4 --batch-size 64 --no-scatter
```

Use the checkpoint appropriate for your task. The IF and region-classification
models use the same crop pipeline. Input conversion from VSI and cell detection
are separate steps; their time is not included in inference benchmarks.

`--crop-backend cached` enables the tile cache without installing Rust.
For optional Rust normalization, install [Rust](https://rustup.rs/), add
`--extra rust` to both uv commands, and choose `--crop-backend rust`.
`--crop-backend reference` retains the original preprocessing path and is still
the default. Optimized backends require checkpoints with the production input
protocol; they reject older checkpoints with different normalization rules.

Nearby cells are processed together, but output rows retain their original
order and identifiers. Image reads use the original pyramid levels. Per-crop
percentiles, padding, bilinear resizing, and uint8 rounding are preserved.
The model runs in its original precision. Reordering batches can introduce tiny
floating-point differences in GPU predictions.

## Memory and workers

The decoded-tile cache defaults to 256 MiB **per worker**. Four workers can use
up to 1 GiB for these caches, plus image decoding, Python, queued batches, and
model memory. Set `--tile-cache-mib` to change the cache limit. This is a cache
limit, not a total-process memory limit. Each worker opens one source at a time.

Try 2 or 4 workers first. More workers can duplicate decoding or saturate the
disk. The gain depends on compression, cell density, storage, and hardware;
uncompressed NumPy inputs may gain much less than tiled, compressed TIFFs.

## Measure on your data

```bash
uv run --locked --extra cu126 --extra rust python benchmarks/benchmark_raw_inference.py \
  --source-manifest sources.csv --cells-csv cells.csv \
  --checkpoint weights/spatialtrace-image-xenium-v1.pt \
  --output-dir runs/raw_benchmark --max-cells 1024 --repeats 2
```

This compares the original single-process path with cached and Rust paths using
four workers. Each run starts a fresh process with an empty application cache.
Add `--reference-workers 4` to compare all backends with the same worker count.
The operating-system file cache is not cleared. Timings include process startup,
raw crop preparation, GPU inference, CSV export, and provenance; the optional
scatter image is omitted from all runs. No reference annotations are used.

`benchmark.json` records commands, file identities, timings, and prediction
differences. Predictions are compared at `atol=1e-6, rtol=1e-5`; coordinate gate
assignments must agree. `summary.json` also separates time waiting for batches
from model/transfer time. With parallel workers, batch-wait time measures stalls,
not total CPU preprocessing work.

## Measured performance

On a 34,063 × 34,137-pixel JPEG2000-compressed Xenium DAPI OME-TIFF, using an
RTX A6000 and Threadripper PRO 5995WX, batch size 64:

| Cells | Backend | Loader workers | End-to-end time |
| ---: | --- | ---: | ---: |
| 1,024 | Original | 0 | 319.9 s |
| 1,024 | Original | 4 | 83.2 s |
| 1,024 | Cached | 4 | 34.4 s |
| 1,024 | Cached + Rust | 4 | 34.1–34.8 s |
| 50,000 | Cached | 4 | 140.5 s |
| 50,000 | Cached + Rust | 4 | 140.1 s |
| 50,000 | Cached + Rust | 8 | 106.4 s |
| 50,000 | Cached + Rust | 16 | 93.0 s |

With eight workers and batch size 256, the Rust path processed 50,000 cells in
83.2 seconds. The maximum coordinate difference from batch size 64 was
1.5 × 10⁻⁷; every gate assignment matched. On a workstation with sufficient RAM
and GPU memory, try `--num-workers 8 --batch-size 256`. Larger queued batches
use more memory in addition to the tile caches.

The 1,024-cell sample spans the tissue. The 50,000-cell table is denser and
allows greater tile reuse. Each run starts from the same original TIFF, with
no prepared crops. These are individual workstation measurements, not estimates
across specimens or hardware. The full 50,000-cell original-backend run was not
timed; its runtime should not be extrapolated from the smaller sample.

The optimized paths matched both original coordinate predictions exactly on
the 1,024-cell comparison, including gate assignments. Cached and Rust paths
also matched exactly on all 50,000 cells. Using 8 or 16 workers preserved those
predictions. Unit tests compare crop pixels exactly, including padding,
pyramids, Z-stacks, channels, and non-finite input values.

The main improvement is avoiding repeated TIFF decoding and overlapping crop
preparation with inference. Rust normalization alone did not materially improve
end-to-end time on this tissue. The `cached` backend therefore provides the main
gain without a Rust build. More workers helped the large run on this 64-core
machine, but increase startup time and memory use.

The machine-readable measurements are in
`benchmarks/results_2026_09_24.json` in the repository.
