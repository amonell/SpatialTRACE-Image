# Figure 1 / Figure 3 component-consistency audit

Audit date: 5 September 2026. Actual complete PDFs/300-dpi PNGs reviewed under
SATA `Regenerated_Figures/complete_2026_09_05/`; originals remain under SATA
`Figures/`. No frozen scientific values, model weights, original artwork, final
component exports, or manuscript documents were changed by this audit.

## Findings and corrections

1. **High — Figure 1 A/B/C individual exports disagreed with the final page.**
   The A script plotted the generic full zoom, B colored villi_006 and _004,
   and C plotted the generic zoom rather than the D training window. Corrected
   `panel_A/figure_script/make_panel_A_components.py` and the B counterpart to
   read the same `assembly/source_region_lock.json` as assembly; C now reads
   the same frozen training table and `panel_D/processing/figure1d_region_lock.json`.
   `common.load_source_regions` and `require_frozen_input` validate source
   evidence and the shared scientific input hashes.
2. **High — Figure 3 A's individual paired-view example had incorrect geometry.**
   The old component code rotated student local, flipped student context and
   left both teachers untransformed. The first consistency repair made all four
   views share the final renderer's 90-degree rotation. **That repair was
   scientifically incorrect:** its regression test reproduced the simplified
   renderer assumption instead of validating the actual training protocol.
   The subsequent cross-figure audit verified the frozen dataset pair and now
   `Figure_3/figure_inputs.py:paired_views` supplies clean variant-0 teachers and
   variant-1 students with shared rotation, contrast, brightness and blur,
   exactly matching Extended Figure 3a. The hash-locked center/rows are unchanged.
   Both renderers read the same existing `paired_view_repair_v1` arrays and
   manifest. Black patches cover 102/256 student patches as an illustration;
   actual training replaces randomly masked tokens, not image pixels.
3. **High — Five required Figure 3 A scientific components were missing from
   individual export/mapping.** Added the fine SFT input, two single-nucleus
   head predictions, and two regional predictions. These read the already
   frozen 2,755-nucleus Cont1 inference and segmentation; no inference occurs
   during rendering. `prediction_overlays` is shared with assembly. The panel
   map now lists all 13 generated A component stems.
4. **Medium — Figure 3 D/E selected zoom geometry during every export.**
   Both component scripts formerly invoked `spatial_maps.zoom_limits`.
   `Figure_3/zoom_region_lock.json` now records the unchanged accepted bounds,
   with the original provenance SHA recorded as evidence. Both components and
   assembly read it. The former mutable panel-D output provenance is no longer
   a runtime geometry dependency that a rerender can invalidate.
5. **Medium — Figure 1 C called two unused numeric alternatives active.**
   The final artwork contains original conceptual neighbor/expression diagrams,
   not the alternative numeric graph and gene-count bar plot. Removed those
   two alternatives from default rendering and active panel_order components;
   their historical stems are explicitly marked inactive. Existing generated
   outputs and scientific input tables are preserved, not deleted.
6. **Medium — Figure 1 A PDF point layers omitted the requested DPI.** Its PDF
   export now explicitly passes `dpi`, like its PNG export and other panels.
7. **Medium — Figure 3 C's individual export had an extra visible metric and
   recomputed the frozen metric table.** It now displays Pearson r, MAE and R²,
   matching the final page, and exports values from the hash-locked metrics
   JSON without recalculation. Spearman remains in the numeric record.
8. **Workflow — two independent styling implementations could still drift.**
   Both assembly renderers now export `scientific_panel_A.pdf`–`E.pdf` directly
   from the same live artists as the complete scientific layer. The explicit
   `scientific_axis_panels` map assigns every scientific axis to exactly one
   panel; `scientific_panel_layers` returns output-relative paths. Root's
   canonical per-panel workflow adds the original manual vectors/text and
   composes final pages from these same panel layers. Standalone legacy-style
   import crops are documented separately, rather than called exact final pages.

## Panel-by-panel checks

| Panel | Actual scientific/manual source and result |
|---|---|
| 1A | Day90 whole-section graph cells, registered narrower gray villus crop, native frozen Peyer DAPI; original outlines/arrows retained. A source-image registration r=0.973883. |
| 1B | Original single illustrated continuous annotation (villus_004) and corrected-polygon Peyer labels. Other annotated villi can lie within the crop; only _004 is displayed as colored. |
| 1C | Sparse label graph, original conceptual neighbors/expression bars, graph-model training-region coordinates and corrected Peyer GAT probabilities. Original C and D training raster **and alpha hashes are identical**, directly supporting their shared table/window. |
| 1D | Original fixed Day90 paired training and distinct inference regions; released Xenium checkpoint and frozen bounded predictions. No window selection or new inference. |
| 1E | Original raw RGB IF crop and repaired-IF frozen predictions; continuous/discrete graphs are deliberately original conceptual drawings, not empirical quantitative panels. |
| 3A | Exact Cont1 nucleus352082 and native 4096/2048/512/128px fields, paired geometry, frozen zero-shot head/regional predictions; original manual boxes/arrows/scale bars. Its IF example was exposed in pretraining and is illustrative, not a performance test. |
| 3B | Frozen 30,000-cell UMAP geometry and current graph-reference metadata; no UMAP fitting. |
| 3C | All 50,000 locked sample_008 cells and current graph-reference release. Visible r/MAE/R² match the frozen metrics. Graph-derived labels are correctly named; they are not direct manual annotations. |
| 3D/E | All 50,000 cells in each full map; accepted matching zoom; unchanged image predictions/current graph references. Labels say Example section, consistent with prior development exposure. |

Figure 3 A's four scale-bar lengths were independently measured from final
PDF vector paths and compared with displayed image widths/native pixel metadata
(0.3250059901547313 µm/px). The nominal 200/100/50/10 µm bars agree to rounding
precision. Figure 3's 6 transformer blocks, 8 attention heads, 256D transformer
features, central4×4 readout, 128D fine branch and 640D concatenation agree with
the frozen model configuration/source. The 81,656 training rows, 32,011 paired
centers per scale and 512,176 augmented crops agree with saved summaries.

Crypt-villus maps use 0–1. Epithelial UMAP/spatial maps intentionally use the
documented 0–0.7 scale, while the separate Figure 3 A schematic predictions use
their own 0–1 scale. Dense raster layers retain vector text/colorbars. No missing
data-obscuring labels, cropped metric boxes, stale manual metric labels, or
colorbar seams were found in the actual final pages.

## Verification and diagnostic outputs

- Seven regression tests: `python -B -m unittest Figure_3.test_component_consistency -v`.
  The revised paired-view test checks exact clean teacher identities against
  original crops and exact visible student pixels against frozen photometric
  arrays, plus mask membership and rejection of unlocked inputs.
- Targeted component renders: `/tmp/figure13-final-component-audit.bPdoAx/`.
  This includes F1 A/B/C and F3 A/C/D/E; the final C checks are in
  `Figure_1/panel_C_final_audit/` and `Figure_3/panel_C_final_audit/`.
- New assembly proofs/QA: `assembly_Figure1/` and `assembly_Figure3/` beneath
  that directory. Both have no automated QA errors.
- The first consistency-only proofs were pixel-identical to their baseline.
  The final paired-protocol correction intentionally supersedes that Figure 3
  result: `/tmp/figure3-paired-protocol-proof.4AOVcQ/complete.pdf` and
  `complete.png` differ from `reaudit_2026_09_05/Figure3.png` at 74,667 pixels,
  **all inside the four original paired-thumbnail rectangles** (zero outside,
  using only floor/ceil rasterization of the exact registered point bounds).
  Counts are 15,407 student-local; 15,180 student-context; 22,672 teacher-local;
  and 21,408 teacher-context. All other artwork and data are pixel-identical.
  `qa/qa.json` has no errors; `standalone/` contains a successful separate
  panel-A regeneration with the same corrected frozen views. This proof has
  not overwritten canonical output files.
- Recomposition from five independently exported scientific panel layers is
  also **pixel-for-pixel identical** to the corresponding full scientific-layer
  composition for both figures: zero changed pixels.

Pixel comparisons establish appearance preservation only; the missed paired
protocol demonstrates why they do not establish scientific correctness. The
paired repair is supported independently by the frozen dataset source and
actual prepared arrays. Current repaired labels/references remain intentional
changes from obsolete scientific content in the original artwork.
