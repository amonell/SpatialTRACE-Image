Panel B reads the already-computed production-v2 strict-test transfer metrics and grouped bootstrap table. It does not rerun inference.

The display is now a compact 2×2 Spearman table: rows `Zero-shot` and
`IF fine-tuned`, columns crypt-villus and epithelial distance. Both rows use
the same released Xenium model lineage and locked 4,843-cell IF test set;
full lineage and crop-disjointness details remain in provenance rather than
long artwork headings. R² remains in the unchanged numeric table, not the plot.

PDF tiles are flat, opaque RGB vector rectangles, not a low-resolution indexed
palette image. The redundant colorbar is removed. Each numeric value remains
visible, and the Blues mapping is fixed to 0–1. The exact four RGB hex colors
are recorded in provenance. An additional `.svg` provides direct RGB vector
fills and editable Arial text for Illustrator. Place/open it in an RGB document
to preserve those colors; CMYK conversion may still alter the appearance.

Authoritative table: SATA
`Figure4/production_v2_final/transfer_comparison/final_lineage_transfer_metrics.tsv`,
SHA256 `28bf38f98bb78564550285f32d4d2c5564039bc4351719cd8dc1ce53e184188d`.
Grouped bootstrap: same directory `grouped_bootstrap_metrics.tsv`, SHA256
`2312ac71192f4873077b16656b37542d6e042363dbb47e256e2e82013c6501f3`.
Inputs/checkpoints/split must all pass the existing hash checks. No evaluation,
model selection, split change or retraining is triggered by rendering.
