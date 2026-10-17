Panel C reads the 4,843-cell frozen strict-test prediction table, the locked
split, and the existing `evaluation/strict_test/metrics.json`. Rendering performs
no inference or metric calculation. `--metrics` is required and hash-verified
against the same release as `--predictions`; the adjacent panel metric JSON is
an unchanged re-export of those frozen records.

The rendering script hash-locks the production-v2 prediction table, the strict
split manifest, and the released IF-adapted checkpoint. All 4,843 cells are
included without display downsampling. The scatter annotations contain only
Pearson r, MAE, and R²; Spearman rho and n remain in the adjacent metrics and
provenance JSON files. The x-axis is `Sparse IF reference`, and the y-axis is
`Image-model prediction`: the IF references are sparse annotations, not dense
graph-model reference coordinates.

Hexagonal-bin cell density uses viridis for the crypt-villus coordinate and
coolwarm for the epithelial-distance coordinate, matching the artwork palette.
The colorbars show logarithmic cell counts, not coordinate values. Arial text
and axes remain vector-based in the PDF; only the dense hexbins are rasterized.
