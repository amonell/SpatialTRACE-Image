# Uploaded P14 analysis: polished figures

The two review figures preserve the calculations in `/tmp/analyze_rari_p14.py`.
They are candidates, not replacements for the manuscript's current Figure 4.

Outputs:

`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/user_rari_positioning_polished_v2/`

- `Additional_14_P14_positioning_polished.pdf` and `.png`: abundance, gate
  enrichment, CV profiles, and relative-position AUC.
- `Additional_15_IMAP_enrichment_polished.pdf` and `.png`: treatment IMAPs,
  their difference, and non-P14 gate composition.

Both PDFs contain embedded Arial and vector plots. PNGs are 450 dpi. Teal/rust
colors identify treatments; circle/diamond shapes identify experiments.
Gate comparisons show the uploaded script's BH-adjusted q values directly.
The numeric tables, pseudocounts, smoothing, masks, and map ranges are unchanged.
The output README explains the metrics and the exploratory testing design.

## Frozen analysis

The input release is SATA
`frozen_preprocessing/if_replicates_20261014/user_rari_positioning_v1/manifest.json`,
SHA256 `3fff265a5e93948569c14136b8c15aa49281e30bae178455e9609c89d09d174d`.
It contains a byte-identical copy of the uploaded script, its original figures,
mouse/gate/profile tables, statistics, and maps. All 25 OLS tests and the BH
corrections were independently checked. The uploaded script remains unchanged.

## Render

From `/home/amonell/Desktop/spatial_axes_manuscript`, with NumPy, pandas,
Matplotlib and Arial available:

```bash
python Figure_4/replicate_revision_20261014/figure_script/render_user_rari_positioning.py \
  --manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/user_rari_positioning_v1/manifest.json \
  --manifest-sha256 3fff265a5e93948569c14136b8c15aa49281e30bae178455e9609c89d09d174d \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/user_rari_positioning_polished_v2
```

Use a new output directory when rerunning; existing exports are preserved.
The renderer never reads raw cells, fits models, or recomputes tests.

```bash
python -m unittest discover -s frozen_preprocessing/if_replicates_20261014 \
  -p test_user_rari_positioning.py -v
```

Tests compare every plotted mouse value, mean profile, displayed gate q value,
and map array with the frozen inputs, and check point-axis bounds.
