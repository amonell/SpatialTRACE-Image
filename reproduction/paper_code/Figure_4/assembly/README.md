# Main Figure 4 page preparation

## Active nine-mouse update (15 October 2026)

E shows whole-section P14 abundance at upper right, F shows the P14 IMAP pair
at lower left, and G shows log2 gate enrichment directly beneath E. E/G read the frozen user analysis
through `../replicate_revision_20261014/approved_ab_release.json`. Circles mark
biological replicates; lines and whiskers show mean ± SEM. Only significant
experiment-adjusted P or BH-adjusted q values and their matching stars appear.
The IMAP F resolves `active_release.json` and shows
equal-mouse-mean coordinate densities and mean gate percentages on a shared
0–13 density scale, without clipping. There are four DMSO and five RARi mice;
tissue08 is excluded at the author's request. All gates and label positions are
unchanged within each map. All panels retain their frozen examples and evaluation.
Explanatory IMAP footers are in the caption, not the artwork. Original two-mouse
components are archived. Full reproduction commands and release hashes
are in `../replicate_revision_20261014/README.md`.

The statistical panels have no titles or sample-size labels. E/G keep the
DMSO/RARi legend; F keeps the condition and P14 headings. `design.py` defines
the aligned display geometry: top-row image tops, bottoms and scale bars;
paired scatter axes and bars; four identical zoom frames and their colorbars;
side-by-side IMAPs; and centered E/G plots in the right-hand column. E is now
128 × 120 pt, G is 176 pt wide, and F narrows to 152 pt per map at its previous
height. Its gate labels use 5.7 pt and keep their original relative positions.
E/G replicate markers are 25 pt², with the original values and jitter. The lower
F/G plots share top and bottom edges. B remains 26.3% wider than before the alignment pass.
Data, statistics and Extended Figure 4 are unchanged.

`prepare.py` regenerates seven independent, page-sized scientific PDF overlays from
the verified frozen IF release. It does not import old scientific plots, fit a
model, create metrics, or perform inference. `layout.json` locks the September 5
assembled reference hash, A4 page box, data axes, microscopy slots, color bars,
and individual Figure 4D frames and equal-aspect content rectangles.

```bash
python Figure_4/assembly/prepare.py --output-dir /tmp/figure4_assembly
```

The shared page compositor calls `prepare(output_dir, overwrite=False)` and
receives `elements`, `inputs`, `commands`, `warnings`, `rendered_anchors`, and
`intentional_differences`. PDF overlays have transparent page backgrounds and
their final physical dimensions. Full-page PDF and 600-dpi PNG publication are
handled by the root assembly runner; originals in SATA `Figures/` are untouched.

Panel A uses the registered original RGB snapshot crops and the locked
79-snapshot-pixel, 50-µm bars in a new external black footer. Panel B uses the
749-cell, three-villus retained-training illustration, enlarged to the same
141-pt image height as A while retaining its native aspect ratio. Its external
200-µm bar uses native IF calibration and scales with the image. Panel C reads the frozen
4,843-cell predictions and metrics. Panel D plots every frozen cell and the
unchanged 6,000 × 4,500-pixel zooms inside four equal 82 × 61.5-pt frames.
Whole-tissue maps retain equal physical aspect inside aligned frames. Panels E/G read
the frozen per-mouse abundance and enrichment values and experiment-adjusted
tests. Panel F reads filtered
equal-mouse mean density grids and mean gate percentages; its filled contours,
gate geometry and shared bottom bar remain vector.

Required frozen table fields and selected cells are validated by the renderer.
The adjacent `assembly_inputs_provenance.json` records input and output hashes,
code and font dependencies, exact commands, versions, actual rendered axes,
scale lengths, gate-label boxes, and counts. IMAP labels use `Musc` in the
original relative box and baseline positions; non-IMAP displays use `Muscularis`.
Percentage values remain fixed. `../imap_label_registration.json`
keeps original source bounds/lines alongside any explicit display exceptions;
`../imap_labels.py` imports the original box Bézier paths and inserts the current
frozen percentages. Most labels are inside their plots, matching the original.
Axes align without moving labels relative to their maps; headings are centered.
Standalone components share this registration.
The corresponding Extended Figure 4E RARi Crypt LP box has an explicit width
exception: 50.267 rather than 46.595 points, centered at its original position,
to accommodate the current two-digit percentage with unchanged font and padding.

Every returned element has an explicit `panel` letter, including panel labels.
The root runner can therefore export an exact, transparent panel layer from the
same fresh elements used for the complete page. These layers, rather than the
older standalone layouts, are the canonical artist-update files. Gate geometry
is read from the frozen gate-definition JSON; displayed percentages are read
from the frozen summary tables. They have one scientific source of truth.

Fast update-safety tests:

```bash
python -m unittest discover -s Figure_4/tests -v
```

Model accuracy remains a within-tissue crop-disjoint evaluation of an already
examined test set. E–G compare nine mice after annotator-supplied exclusions;
the original two contributed to model development. The comparison is exploratory,
and treatment randomization and annotation blinding are undocumented. Source QuPath
class labels are retained; upstream marker-threshold metadata remain unavailable.
The original RGB is preserved because the artist's color-export transform is
unregistered. These scope issues do not trigger new analyses during rendering.
