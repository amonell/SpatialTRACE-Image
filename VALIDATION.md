# Tests

Tested on Linux with Python 3.12, PyTorch 2.12.1, CPU, and an NVIDIA RTX A6000 with CUDA 12.6. Packages were installed as wheels in separate uv environments.

## Software checks

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
