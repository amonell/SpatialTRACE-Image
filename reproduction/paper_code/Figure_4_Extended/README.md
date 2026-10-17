# Extended Figure 4

The 16 October 2026 owner-approved order is **A–G**. The new first panel puts the
CCR9 flow validation beside the mouse-level comparison, and the manuscript cites
it in the paragraph introducing main Figure 4A. Prior A–F are now B–G.

| Panel | Content | Active component renderer |
| --- | --- | --- |
| A | Representative IEL CCR9 flow pair and mouse-level CCR9+ percentage | `panel_A/figure_script/make_panel_A_iel_ccr9.py` |
| B | IF fine-tuning workflow, with original manual vectors | `assembly/prepare.py::workflow` |
| C | Frozen zero-shot versus IF-fine-tuned Spearman comparison | `panel_C/figure_script/make_panel_C_final_transfer.py` |
| D | Frozen two-stage IF training history | `panel_D/figure_script/make_panel_D_if_training_history.py` |
| E | Author-selected microscopy fields and fixed IMAP gate backprojection | `panel_E/figure_script/make_panel_E_gate_backprojection.py` |
| F | Frozen nine-mouse non-P14 IMAPs | `panel_F/figure_script/make_panel_F_non_p14_imaps.py` |
| G | Fixed qualitative Peyer classifier windows | `panel_G/figure_script/make_panel_G_peyer_conditions.py` |

`panel_order.json`, `assembly/layout.json`, and the root `final_figure_order.json`
register this order. The original `Figures/Figure4_extended.pdf` is preserved,
SHA256 `917587bf3b208a67256fca7da475bead1d79be0e56f6698d22e96b3e73d7abfd`.

## Render and publish

From `/home/amonell/Desktop/spatial_axes_manuscript`:

```bash
/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B update_final_figure.py Figure_4_Extended --dry-run
/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B update_final_figure.py Figure_4_Extended --overwrite
```

`assembly/prepare.py` renders all data from frozen numerical and microscopy inputs,
extracts the original workflow vectors structurally, and composes the complete
page. It calls the CCR9 renderer with PyFlowjoPrism's explicit Python interpreter.
No old figure PDF is substituted for a regenerated scientific panel.

Final outputs are under SATA
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended/panel_X/`
as `panel_X_final.pdf`, `panel_X_final.png`, and provenance JSON. The complete
PDF is `Regenerated_Figures/Current/Figure4_extended.pdf`. The Current collection
retains exactly eight complete PDFs; technical runs and earlier versions are
preserved separately in `Figure_Run_Archive` and `polished_panel_archive`.

## Frozen inputs

Panel A uses `panel_A/processing/iel_ccr9_release.json`, which hashes the approved
PyFlowjoPrism workspace, compensation, eight-row exclusion table, representative
selection, statistical results, original FCS and rendering code. Required fields
are `sample_id`, `condition`, `mouse`, `included`, `donor_events`,
`ccr9_positive_events`, and `ccr9_percent`. Seven mice are plotted; DMSO mouse 8
remains excluded for failed cell recovery. The primary two-sided Welch result
is reused, not recalculated. The endpoint is exploratory, and its exact-rank
sensitivity is retained in the Methods and linked statistics.

C–G retain the released IF and classifier inputs, checkpoint hashes, splits,
training histories, fixed native fields, density arrays, cell assignments and
probabilities. Their release pointers are resolved by `Figure_4/release.py`,
`Figure_4/field_selection.py`, and
`frozen_preprocessing/peyer_representation_v1/release.py`. Training, checkpoint
selection, full inference, gate tuning and model comparisons are not rendering
steps. The exact hashes of every resolved input appear in the final provenance.

The CCR9 renderer loads the three locked plotting modules from the original
`2026-10-16_FlowJo_reference_PDFs/project/` source archive. Their hashes must
match the existing release lock; newer edits in the live PyFlowjoPrism project
are not substituted for those modules.

Panel E keeps the author-selected original DMSO and RARi option 32 fields used in main
Figure 4D. Each native field remains 6000 × 4500 pixels with a calibrated 200-µm
bar. Its shared anatomical-localization heading is omitted at the owner's request.
F keeps one shared 0–16 probability-density bar. G keeps the same classifier,
probabilities, fixed windows, 0–1 color scale and 0.5 calling threshold. Its display
configuration is `panel_F/processing/probability_display.json`.

The new row is placed first; the workflow and non-P14 panels are translated,
training axes are shorter, and the fixed Peyer windows use smaller square display
rectangles. Existing scientific values do not change. All text is Arial. Counts
stay in captions and supporting tables, without `n=` annotations in figures.
Panel A uses four-point summary markers and an aligned layout with `IEL P14 CD8 T cells` centered over the flow pair for
its compact manuscript size.

Panel B sits closer to the training history and farther from panel A.
`workflow_offset_y_pt = 148` in `assembly/layout.json` places it 18 pt below its
original position. The original vectors, heading, internal text and panel letter
move together; all other panel positions remain fixed.

## Interpretation and document linkage

CCR9 is the percentage within live CD8+ CD45.1+ IEL donors, not an absolute cell
count or a direct receptor-engagement assay. Representatives are selected by
proximity to each group's median percentage, independently of significance.
Python smoothing approximates the FlowJo renderer; the exact recovered display
settings and numerical limitations are recorded in the flow-analysis release.

IF coordinate evaluation reuses the locked, previously examined within-section
crop-disjoint test regions. The displayed microscopy fields are author-selected
examples, not independent accuracy evaluations. IF Peyer maps remain qualitative
because corresponding patch status was not independently annotated.

`integrate_ccr9_documents.py` makes only the requested addition and shifted
Extended Figure 4 references, stages clean DOCX files for QA, and preserves every
other Word package part. It adds the CCR9 readout beside main Figure 4A, supplies
the new A legend, and records the analysis methods. Superseded documents remain
in their existing document archives. The scope and exact paragraph edits are in
SATA `audits/extended4_iel_ccr9_2026-10-16/document_changes.json`.
