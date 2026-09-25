# Release notes

## 1.0.0rc3 — 2026-09-25

- Faster raw-image inference with decoded-tile caching and parallel crop loading.
- Faster pretraining and supervised crop preparation. Saved shards keep the original crop values and cell order.
- Training directly from raw images with persistent workers and an optional shared RAM crop cache. No saved shards are required.
- Optional Rust normalization. The cached Python backend provides most of the measured speedup without a Rust build.

Existing inference and raw-image training commands keep the reference loader by default. Add `--crop-backend cached --num-workers 4` to use the faster loader. Crop-preparation commands use caching by default. See [training](docs/training.md) and [inference](docs/fast_inference.md).

The released model weights, architecture, figure code, and frozen figure inputs are unchanged. CPU training parity tests matched exactly; GPU runs showed small numerical differences. This release does not replace the manuscript's scientific results. See [tests](VALIDATION.md) and [timings](benchmarks/README.md).

## 1.0.0rc2 — 2026-09-19

Paper-figure reproduction release. Updated the name and figure labels to SpatialTRACE without changing model weights.

Use this version with the [archived figure inputs](https://doi.org/10.5281/zenodo.22850752). [Reproduction instructions](reproduction/README.md) include the fixed commit and setup commands.
