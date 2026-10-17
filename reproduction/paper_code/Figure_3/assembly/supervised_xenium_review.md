# Figure 3A approved supervised-side replacement

Published and verified on 6 September 2026. The user approved replacing the
supervised-side inputs and outputs with Xenium while retaining the Cont1
pretraining illustration and original layout.

The current complete page is:
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Regenerated_Figures/figure3a_supervised_xenium_2026_09_06/Figure3.pdf`.
Its PNG, review page, provenance and `supervised_replacement_audit.json` are
adjacent. The five canonical A–E layers are published under SATA
`Figure_3/panel_X/panel_X_final.pdf`, indexed in `final_panel_manifest.json`.
Previous exports and the initial visual proof were archived recoverably.

## Source and interpretation

The Xenium center is `sample_010_cell_0214763`, prepared index 4540, from Day90
SI replicate 2. Among the 25 verified supervised-input cells in the already
locked Figure 1d region, this is nearest the geometric window center. Selection
accesses identity and spatial columns only. The 2,744 cells in the resulting
3,132-pixel context field retain their original saved predictions from the
released checkpoint. No inference or model fitting was required for this
replacement. The three illustrated inputs match all original supervised shard
bytes exactly.

Predicted values are shown as fixed-radius vector centroid markers over native
DAPI. They are not nuclear segmentation masks. The original 0–1 viridis scale
is unchanged. The example is in-domain and held out of released SFT fitting and
selection, but was examined during development; it is not a fresh independent
performance estimate. Source locks, exact hashes, fields and preparation
commands are documented in `../panel_A/processing/README.md` and `../README.md`.

## Visual and automated checks

The final complete-page PNG was visually inspected. The regional maps show
spatial coordinate gradients, all three supervised images come from the new
cell, and the two head predictions use that cell's saved values. The old Cont1
outlines embedded in three input slots were removed structurally, without
moving the colored frames or transformer artwork. Pretraining remains unchanged.

- 49 shared and figure-specific regression tests passed.
- All seven intended image slots changed; zero pixels changed outside them.
- Figure 3B–E and all other figures' panels are pixel-identical: 40 unchanged
  canonical panels, one changed panel (3A).
- All 43 Figure 3 geometry anchors passed; panel recomposition is pixel-identical.
- 132 unique path/hash pairs passed verification, including direct snapshot
  inputs and canonical exports.
- All eight original assembled PDF hashes are unchanged. No manuscript prose,
  Methods, legends, checkpoints or existing frozen predictions were edited.

Regenerate the final page and canonical panels with:

```bash
python -B update_final_figure.py Figure_3 --overwrite
```

Standalone component regeneration uses the shared frozen helper as well:

```bash
python -B regenerate_all_organized_figure_components.py --figures Figure_3 --panels A --overwrite
```
