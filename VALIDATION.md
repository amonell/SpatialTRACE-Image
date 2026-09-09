# Release validation

Validated locally on Linux, Python 3.12, PyTorch 2.12.1, CPU and NVIDIA RTX A6000/CUDA 12.6. The package was built as a wheel and installed into separate CPU/GPU uv environments without an editable source install. Commands ran from fresh output directories. No publication or GitHub-hosted CI run is claimed.

- 46 unit tests passed.
- The 19-command acceptance workflow passed on both CPU and GPU: synthetic microscopy generation, coordinate training/inference, QuPath export, paired-crop preparation, representation pretraining, teacher-initialized coordinate and classifier fine-tuning, verified installation of all four release artifacts, and released-weight inference.
- Tests verify that exported final checkpoints contain the validation-selected weights, explicit test rows remain excluded, and shared-scale teacher transfer is strict.
- Download tests exercise HTTP transfer, checksum rejection, and non-overwrite behavior.
- All exported parameter tensors are exactly equal to their frozen originals.

On three prespecified real microscopy cells, all six local/context uint8 crops exactly matched the frozen prepared inputs. Sparse fine-crop pixels differed by one uint8 level between Bio-Formats and OpenJPEG decoding. Maximum coordinate differences from frozen predictions were below 4.5 × 10⁻⁵ on CPU and 1.1 × 10⁻⁵ on GPU. The report separates differences on identical prepared pixels from TIFF-decoder differences; prediction outputs are not claimed to be bitwise identical across runtimes. This is a bounded compatibility check, not a new performance estimate.

All eight paper figures were redrawn from relocated frozen inputs and matched the current full-page layouts pixel-for-pixel in a 72-dpi raster comparison. The figure runner additionally checks panel recomposition at 150 dpi, fonts, text, geometry and raster contracts. Figure data are separate from the model package.

Repeat the self-contained wheel acceptance workflow with:

```bash
python scripts/smoke_test.py --work-dir /new/temporary/image-run --device cpu
```

Add `--weights /path/to/release-weights` to test the four released artifacts; use a CUDA-installed interpreter and `--device cuda` for GPU acceptance. Reports and per-command logs are written inside the new work directory. The real-data compatibility checks require separately licensed author fixtures.
