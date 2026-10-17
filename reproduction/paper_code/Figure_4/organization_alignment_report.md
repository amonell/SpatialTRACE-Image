# Alignment with current Figure 4 artwork

The root SATA `Figures/Figure4.pdf`, locked in `final_figure_order.json`, defines
A microscopy, B sparse labels, C scatter, D maps/zooms, E lollipops and F P14
IMAPs. The assembled PDF remains unchanged; corrected components require artist
import. The historical PNG is a style/region reference only.

The scientific repair intentionally updates B–F to the buffered/shared-reference
IF checkpoint. It keeps the model architecture, strict test identities, tissue
regions, P14 detection calls and coordinate-gate boundaries fixed.

- A retains verified 50-µm external bars and original source RGB. No undocumented
  artist color transform is guessed.
- B now uses 749 retained training cells from three prespecified villi and its
  calibrated 200-µm bar. The old 821-cell example does not depict the repaired
  training split.
- C uses Sparse IF reference versus Image-model prediction and shows only
  Pearson r, MAE and R². Current values are in `panel_C/panel_C_metrics.json`
  on SATA, not the old artwork's manually typed metrics.
- D keeps the original zoom boxes/connectors and all cells. Its epithelial
  scale marks saturation at ≥0.5 instead of implying that 0.5 is the maximum.
- E retains the lollipop design but uses same-gate non-P14 denominators for
  abundance. Its right side is within-P14 composition shift, not abundance.
- F uses a common population-normalized probability-density scale shared with
  Extended Figure 4 D. Gate percentages are recomputed; labels remain outside
  the cells. The two panels no longer imply comparable color values from
  independently normalized condition-specific density estimates.

No coordinate gates are anatomically validated by these plots alone. The new
optional Extended Figure 4 F component supports review, not an accuracy claim.
Original components, models, predictions and source snapshots are recoverable
in the versioned archives. Rendering never changes fitted scientific outputs
or the assembled artwork.
