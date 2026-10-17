# Display-only update review

The user requested the original IMAP gate-label locations and clearer variation
in the fixed IF Peyer probability windows. This update does not change any
checkpoint, prediction, membership, window, density, contour, gate, or metric.

## IMAP registration

`Figure_4/imap_label_registration.json` records all 20 original label-box path
indices, bounds, text baselines, and line breaks from the current source PDFs:

- Main Figure4: SHA256 `e4ae363565e2231c7e36f475d5dadc2c973dd951e95d8ae599faba40ffae1697`.
- Extended Figure4: SHA256 `917587bf3b208a67256fca7da475bead1d79be0e56f6698d22e96b3e73d7abfd`.

`imap_labels.py` imports only the original box Bézier paths. It inserts fresh
Arial-Bold text from the frozen current percentage tables. Original historical
percentages and plotted cells are never imported. Main F and Extended D share
this helper with their standalone renderers. Original in-plot placement and the
`Musc` abbreviation explicitly supersede the previous external rails.

Nineteen boxes reproduce the exact original geometry. One approved text-fit
exception accommodates RARi non-P14 Crypt LP 10.5%, formerly 7.6%: its width
increases from 46.595 to 50.267 pt about the unchanged center. Its baseline,
Arial-Bold 6.6 pt font, padding, and corner curves remain unchanged. All axes,
condition titles, and colorbars remain fixed. The corresponding final-layout
anchor explicitly records the adjusted width.

Automated tests compare actual artist boxes and text baselines with their
registered targets to 0.001 pt, check that no axes move, and check the one
approved longer label fits. Initial exact-path errors were below 3×10⁻¹³ pt.

## Peyer probability display

The hash-locked predictions contain 2,518 DMSO and 2,263 RARi cells. DMSO spans
0.065–0.522 and RARi spans 0.067–0.431. The combined median is 0.106 and 99th
percentile is 0.420. Two DMSO values and no RARi values exceed 0.5.

Both windows now use the same linear magma display range of 0–0.5, with ticks
0, 0.25, and ≥0.5. The endpoint explicitly identifies color saturation; stored
probability values are never rescaled. Point size stays 5.5 pt², point order and
coordinates remain unchanged, and no smoothing is introduced. The original
native DAPI images remain the only raster layers; points and colorbars remain
vector. `Figure_4_Extended/panel_E/processing/probability_display.json` controls
both standalone and final-page drawing. The adjacent
`probability_range_audit.json` records input SHA256, exact ranges, quantiles, and
0.05-wide histogram bins across the complete [0,1] probability interval.

## Verification scope

Temporary complete-page and standalone proofs are rendered under
`/tmp/figure4-display-review.z5UtrJ/`. Full-page QA checks reference geometry,
embedded Arial, vector and raster contracts, stale text, and private content.
No assembled artwork PDF or canonical output is overwritten by this task.
Publication is handled separately by the root workflow after review.

Final temporary proofs are `main_v2/complete.pdf` and
`extended_v2/complete.pdf` (300-dpi review PNGs alongside). Both pass all
automated page checks. Ten unit tests pass. All three standalone components
also rerender successfully, including the widened label in
`standalone_extended_v2/panel_D/`. Measured label-target errors remain below
3×10⁻¹³ pt. Main F is entirely vector; Extended D has exactly two 600-ppi point
rasters; Extended E retains exactly two native DAPI rasters and vector dots.

A same-resolution pixel comparison with the prior released complete pages
finds zero changed pixels outside the main F gate-label region and Extended D
gate labels plus Extended E probability dots/colorbar. The DAPI images and all
other panels are pixel-identical to the previous release.
