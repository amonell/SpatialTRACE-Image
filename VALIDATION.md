# Tests

Tested on Linux with Python 3.12, PyTorch 2.12.1, CPU, and an NVIDIA RTX A6000 with CUDA 12.6. Packages were installed as wheels in separate uv environments.

## Optimization branch

The `rust_optimization` branch was checked on September 24, 2026:

- 102 tests passed with the optional Rust extension installed.
- The standard CPU installation passed 87 tests; 15 Rust-specific tests were skipped.
- An installed wheel passed the 16-command CPU workflow, including crop preparation, pretraining, training from shards, cached raw-image training, and prediction.
- Cached and Rust preparation produced byte-identical `.npy` shards to the original crop routine. Tests cover mixed sources, pixel calibrations, image borders, reordered reads, and partial final shards.
- New preparation metadata rejects mismatched crop sizes or pixel calibration before supervised training. Existing prepared datasets remain supported.
- Cached raw-image training matched the original batches, augmented crops, loss histories, selected epochs, and selected weights in three-epoch CPU tests, including dropout. Persistent workers receive the correct augmentation epoch.
- Tests cover partial RAM caches, insufficient shared memory, excluded test rows, and worker cleanup after a training failure.
- Both [GitHub test jobs](https://github.com/amonell/SpatialTRACE-Image/actions/runs/36084984087) passed for implementation commit `be9ed3f`.
- A three-epoch GPU benchmark used 512 raw-image cells and synthetic targets. All loaders selected the same epoch; maximum metric differences were below 4 × 10⁻⁸. GPU weights were not bitwise identical. Full timings and differences are in `benchmarks/raw_training_results_2026_09_24.json`.

These are software checks. The small synthetic training runs do not measure
anatomical accuracy. Raw-image timings are documented in `benchmarks/README.md`.

## Original release checks

- 46 unit tests passed.
- The 19-command workflow passed on CPU and GPU. It covered pretraining, fine-tuning, prediction, QuPath export, and all four pretrained models.
- Tests covered checkpoint selection, data splits, pretrained weight loading, and download checksums.
- All released model parameters matched the original trained weights.
- GitHub Actions passed installation, unit tests, wheel building, and the example workflow.

Run the unit tests:

```bash
uv run --locked --extra cpu --extra dev pytest -q
```

Run the training and prediction checks:

```bash
uv run --locked --extra cpu python scripts/smoke_test.py \
  --work-dir runs/image_checks --device cpu
```

Add `--weights release-weights` to test the downloaded models. Use a new work directory. For GPU checks, use `--extra cu126` and `--device cuda`. Logs are saved in the work directory.

## Comparison with the paper code

On three reference cells, all six local and context crops matched the saved inputs exactly. A few fine-crop pixels differed by one uint8 level between JPEG2000 decoders. Maximum coordinate differences were below 4.5 × 10⁻⁵ on CPU and 1.1 × 10⁻⁵ on GPU. This checks implementation agreement rather than biological accuracy.

All eight figures matched the current page layouts pixel-for-pixel at 150 dpi. Figure checks also covered fonts, text, panel placement, and image resolution.

The SpatialTRACE release checks are in `validation/spatialtrace_rename.json`. The original implementation comparison is in `validation/acceptance.json`.
