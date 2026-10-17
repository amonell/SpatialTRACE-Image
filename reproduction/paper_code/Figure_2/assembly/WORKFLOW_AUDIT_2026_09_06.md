# Independent audit and workflow repair

Scope: Figure 2 A–E, Extended Figure 2 A–D, Extended Figure 2 page 2 A–D,
and Extended Figure 3 A–G. The initial review used the actual final files in
SATA `Regenerated_Figures/complete_2026_09_05/`, built at
`complete_2026_09_05_20260906T044436.503031Z`. The original September 5 artwork
remains the layout authority. No scientific preprocessing, fitting, checkpoint
selection, predictions, manuscript text, or artist source was changed.

## Display and source checks

All 20 panels were visually inspected. The 75 previously registered image,
axes, and colorbar rectangles matched their original positions. Labels were
readable at the existing layout scale; the small Extended Figure 3G ROC legend
is inherited from the compact original layout. The Figure 2A retained vector
asset was inspected in isolation: it contained schematic boxes, arrows,
tensor illustrations, legend symbols, and zoom connectors, with no original
scientific image, scatter, curve, or hidden Illustrator data layer.

Eighty-nine recorded preparation inputs smaller than 100 MB were independently
rehash-checked and all matched; the one larger microscopy source was not
rehash-checked in this audit. Renderer code retains its existing image hash
check. Final numerical displays agreed with the current frozen records:

| Display | Verified source and result |
| --- | --- |
| Figure 2D | Section-disjoint OOF `metrics.json`: 25,436 cells; r 0.959 and 0.871, MAE 0.054 and 0.107, R² 0.918 and 0.757 |
| Figure 2E and extended C–D | Identical `per_fold_metrics.tsv`, SHA256 `462e31be3c8c6cfe0ec7aff09f282bfe5bb685f16f151bd3484f3478e75dcf94`; 160 runs, eight held-out sections, 1–45 training villi |
| Extended Figure 2B | The matched 160-run runtime record, median and interquartile range, including validation time |
| Extended Figure 2 page 2B–D | Corrected-label BinaryGAT: 22 epochs, selected epoch 10; class threshold 0.95; test-only balanced accuracy 0.91, 0.97, and pooled 0.95 across 500,050 cells |
| Extended Figure 3B–C | Frozen pretraining history: 35 epochs, selected epoch 33; released SFT: 20 epochs, selected epoch 9 |
| Extended Figure 3D | Matched 5% SFT initialization comparison, three paired runs, 5,107 evaluated cells; displayed means 0.860/0.864/0.875 and 0.692/0.704/0.718 |
| Extended Figure 3E | Five saved raw-crop shuffle repeats; mean r reductions local/context/fine = 0.432/0.316/0.007 and 0.577/0.090/0.037 |
| Extended Figure 3F–G | Separate corrected-label image classifier; fixed threshold 0.5; four saved strategies on identical 819 sparse-test cells with 91 positives; displayed AUROCs 0.74, 0.79, 0.74, and 0.81 |

## Verified workflow issues repaired

1. **Medium — prepared-input paths depended on output location.**
   `Figure_2/run_all_panels.py` and `Figure_3_Extended/run_all_panels.py`
   now accept independent prepared roots, with canonical frozen-input defaults.
   A new preview directory no longer needs a mirrored `stored_outputs` tree.
   `prepare_common.py` passes explicit roots; shell-command path rewriting was
   removed.
2. **Medium — three Figure 2A numerical insets lacked standalone generators.**
   `panel_A/figure_script/make_panel_A_schematic_insets.py` now generates three
   separate PDF/PNG components. Both ordinary regeneration and final assembly
   call this script. The old assembly-only plotting function was removed.
   Five original inset image/colorbar footprints were added to geometry QA.
3. **Medium — input locks differed between standalone and assembled routes.**
   The shared compact graph manifest is pinned, and A–C verify each selected
   input table against it. Figure 2E pins both metrics and checkpoint manifest.
   The corrected graph-component manifest, runtime manifest, paired-view
   manifest, and corrected image-classifier manifests are now pinned before
   trusting their embedded output hashes. A deliberately corrupted graph
   input is rejected by a regression test.
4. **Medium — panel ownership was missing for exact final-panel exports.**
   Every element in these four layouts now has a panel letter. Original
   Figure 2 vector paths were partitioned, without modifying their paint
   operations, into A schematic, B connectors, and C connectors. Individual
   final-panel exports therefore need not include neighboring manual art.
5. **Low — standalone Figure 2A/B PDF raster DPI was implicit.**
   Their PDF exports now honor `--dpi`, matching the PNG request. The final
   adapter already set DPI explicitly, so the assembled figure is unchanged.
6. **Low — panel-map descriptions lagged behind the scientific release.**
   The Figure 2A inset lineage, extended section-held-out terminology,
   graph-classifier test-only bar order, and initialization-comparison title
   are current. The Extended Figure 3 README now identifies the reviewed
   September 5 reference hash.

## Verification after repair

- Seven regression tests pass in `test_layout_registration.py`, including
  prepared-root independence, complete generator mapping, exact original
  geometry, disjoint manual-path ownership, font/colorbar behavior, and
  corrupted-input rejection.
- Fresh Figure 2 preparation and composition succeeded at
  `/tmp/figure2-workflow-review.sbjLzo/assembly/`; all 27 measured anchors
  passed with zero automated QA errors.
- The newly composed 150-dpi Figure 2 proof was **pixel-identical** to the
  actual existing final proof (`ImageChops.difference(...).getbbox() is None`).
- A direct Extended Figure 3 A-only run succeeded into
  `/tmp/figure2-workflow-review.sbjLzo/extended3_single_A/`, using its canonical
  prepared inputs without any copied or newly prepared input files.

## Explicit remaining workflow boundaries

The complete-page adapter controls final geometry and shared labels through
`layout.json`; it hides standalone figure-level titles/legends and uses
registered title anchors. Updating a shared final title therefore means
editing its layout element, not only the standalone renderer's title. This is
documented rather than treated as an automatic scientific update. The final
per-letter publishing/update CLI is owned by the shared root workflow.

The original Figure 2A MDE viewport is `x=[-2.2,1.9]`, `y=[-2.2,2.25]`. It
clips 3 of 989,431 supplied prepared embedding points. This inherited viewport
was preserved; neither the embedding nor its extremes were altered.

Low-level component `--overwrite` still requires prior archiving by its caller;
the public shared workflow supplies recoverable publication archiving. Rendering
does not refit any model. Figure 2D recomputes saved summary statistics as an
equality check. A subsequent authorized bounded conversion moved Extended
Figure 3G's ROC-coordinate generation into an explicit preparer. The resulting
ROC and display-metric tables are byte-identical to the prior exports and are
now hash-locked in SATA `Figure_3_Extended/stored_outputs/panel_G/frozen_roc_curves_v1/`.
The G renderer only reads those frozen tables and checks their saved metrics;
it performs no curve or metric generation. No model fitting, inference,
selection, or permutation was part of this conversion.
