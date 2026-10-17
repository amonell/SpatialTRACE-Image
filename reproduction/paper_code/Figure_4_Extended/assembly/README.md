# Extended Figure 4 page preparation

## Active nine-mouse update and display cleanup (15 October 2026)

Panel E resolves the shared `Figure_4/replicate_revision_20261014/active_release.json`:
four DMSO and five RARi mice, excluding tissue08 at the author's request. It shows
equal-mouse-mean non-P14 densities and mean gate percentages with unchanged
gates and label positions. Its single shared 0–16 density colorbar occupies the
original right-hand slot. Sample-size labels and the explanatory footer are
omitted; these details remain in the caption. Neither map moves. This scale
differs from main Figure 4G's P14-specific 0–13 scale. Panel D retains the exact original DMSO/RARi fields,
channel contrast and 127 × 137.583-pt image rectangles at the author's request.
It is freshly rendered from the original frozen microscopy and cell-gate tables,
not copied from an older PDF. P14 cells
use larger filled markers, without outlines or a separate P14 legend entry.
Panel C centers the stage labels over epochs 1–7 and 8–11 using the frozen
stage lengths. Curves, divider and checkpoint markers are unchanged.
Panels A/B/D/F retain their original frozen model analyses, anatomical examples and qualitative Peyer
examples. The original two-mouse details below are historical where superseded
by this paragraph. See the shared replicate package README for commands.

```bash
python Figure_4_Extended/assembly/prepare.py --output-dir /tmp/figure4_extended_assembly
```

`prepare(output_dir, overwrite=False)` produces the A–F page inputs and returns
the shared compositor contract. `layout.json` records the September 5 reference
hash, page size, each data axis and shared bar, and every retained manual path.
The approved gate-backprojection addition is panel D; non-P14 IMAPs and Peyer
windows are E and F. The original hidden Illustrator `f.` object is omitted.

Panel A reuses only the original vector workflow path events 6–15. All source
text, images, scientific plots, and private Illustrator content are structurally
removed by `figure_assembly/static_art.py`. New Arial text reports 5,822 retained
training cells in 20 villi. The measured path bounds are checked again from the
clean extracted PDF. Panel B reads the frozen 4,843-cell Spearman table and uses
opaque RGB vector tiles. Panel C plots seven head-only and four full-model
epochs, with the frozen checkpoint selections at epoch 2 in each stage. Panel E
uses frozen non-P14 points, densities, and percentages. Dense points use 600-dpi
raster layers; gates, contours, and text stay vector. Its probability-density
scale is shared between the two non-P14 maps. Panel F plots the representation-pretrained
axis-architecture Peyer classifier in the unchanged 650-µm windows. DAPI is the only raster layer;
all prediction dots and the one shared bottom bar remain vector. The shared
linear display range is 0–1, with ticks at 0, 0.5 and 1.
All 4,781 probabilities come from the new frozen release; the current range
audit records counts above the former 0.5 display maximum. Markers and windows are unchanged. Rendering
performs no classifier inference or selection.

All data layers are newly rendered full-page transparent PDFs at their final
physical size; there is no copied old scientific background or white masking.
The root compositor writes the complete PDF and 600-dpi PNG. Provenance includes
all frozen inputs, code, fonts, outputs, actual axes, counts, density levels,
and gate-label boxes. The source probability-dot coordinates were independently
registered to the fixed native-centroid windows; their small imported x/y scale
difference is retained in the original plot rectangle. Native DAPI stays square.

The separate Peyer classifier windows are qualitative IF displays without
manual IF Peyer reference labels. Transfer metrics retain the repaired release's
within-tissue crop-disjoint scope. No scientific preprocessing is triggered.

Workflow counts are derived from the locked training manifest and checked
against its training summary. Epoch offsets, stage boundaries, and checkpoint
markers come from the two frozen histories and summaries. These annotations
contain no independent numerical copies to update. Gate geometry and labels
use the same frozen definitions and percentage tables as main Figure 4.
Their original box paths, positions, and line breaks come from
`Figure_4/imap_label_registration.json`; this restores the user's preferred
original in-plot placement. `Figure_4_Extended/panel_F/processing/probability_display.json`
is the single color-mapping configuration for standalone and final panel G.
One original label box needs additional width for the current value: RARi
non-P14 Crypt LP is 10.5% rather than the historical 7.6%. Its box expands from
46.595 to 50.267 points around the unchanged center, with the original baseline,
font size, padding, and corner curves. All other label boxes remain unchanged.

Every fresh element carries its visible A–F `panel` letter. The root runner's
per-letter transparent layers use these exact complete-page elements; the
standalone component scripts retain separate import layouts and should
not be used to infer final spacing or the final panel-B tile palette. C shares
the `history_labels.py` stage-position helper with final assembly. E uses the
same nine-mouse renderer as final assembly, before the fixed vertical translation.
