# Figure 4 reproducible components

Current D fields (16 October): original DMSO plus author-selected RARi option 32,
matched exactly to Extended Figure 4E. Both final and standalone renderers use
`panel_D/processing/active_fields.json`; its frozen manifest SHA256 is
`10c30344d79527ab09082734fe69f44af9a63864e1328422d6f9b910ba0868dc`.
See [the shared-field instructions](panel_D/processing/README.md). This
author-approved choice supersedes historical zoom descriptions below;
predictions, gates, statistics and all other panels are unchanged.

For the **finished panel layout**, run from the home project:

```bash
python -B update_final_figure.py Figure_4 --overwrite
```

This regenerates the page and publishes `panel_X_final.pdf`,
`panel_X_final.png`, and `panel_X_final_provenance.json` to
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/panel_X/`.
These PDFs contain only their own panel on the registered full-page canvas;
import at the page origin without rescaling. `final_panel_manifest.json` in
the SATA figure directory identifies each final renderer and its source hashes.
Use `--dry-run` to inspect commands without writes. Previous final exports
are archived; original assembled PDFs remain untouched. See
[the final-figure editing guide](../FINAL_FIGURE_EDITING.md).
The component commands below provide smaller individual assets, not a second
authoritative final-page layout.

Code root: `/home/amonell/Desktop/spatial_axes_manuscript` (CODE).
Output root: `/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript` (SATA).
These are separate directories. Components are generated into
`SATA/Figure_4/panel_X/`; large frozen results are under
`SATA/Figure_4/stored_outputs/scientific_repair_v1/`.

The original assembled `Figures/Figure4.pdf` remains artist-maintained and is
never overwritten by these scripts. `final_figure_order.json` preserves its
original order separately from the approved generated A–G order in
`panel_order.json`. Every image/data component has real generation code; no
component is created by copying an existing PDF/PNG.

For the complete publication layout, use `assembly/prepare.py` through the root
complete-figure runner. Its per-letter `panel_X_final.*` layers contain the exact
fresh elements used in the complete page, including final spacing and panel
labels. The standalone scripts below intentionally provide separately sized
component views: microscopy pairs have individual footers, whole-tissue maps
have separate coordinate exports, and IMAPs have their own labels and bars.
They share the current frozen science but are not substitutes for the exact
final-layout layers. See `assembly/README.md` for that update workflow.

## Panel map

| Panel | Generated component | Input |
|---|---|---|
| A | Original DMSO/RARi multichannel crops, 50-µm external bars | Registered QuPath snapshots, original source RGB |
| B | Sparse IF training labels over DAPI, 200-µm external bar | 749 retained training cells in three fixed SI1 villi |
| C | Two coordinate scatters; Pearson r, MAE, R² only | Unchanged 4,843-cell crop-disjoint cohort, repaired IF predictions/references |
| D | Four whole-tissue maps with locked boxes, connectors and magnified views | All 300,429 cells; same repaired IF checkpoint |
| E | Whole-section P14 abundance, upper right | Nine mouse-level P14/non-P14 ratios; experiment-adjusted OLS P on log2 abundance |
| F | DMSO/RARi P14 IMAPs, side by side at lower left | Same filtered nine-mouse cohort; equal-mouse mean densities and percentages; shared 0–13 density scale |
| G | P14 enrichment by compartment, directly beneath E | Same mice and five fixed gates; log2 P14-share/non-P14-share ratios; experiment-adjusted OLS and BH q values |

E/G use `replicate_revision_20261014/approved_ab_panels.py` and the immutable
analysis selected by `approved_ab_release.json` (manifest SHA256
`3fff265a5e93948569c14136b8c15aa49281e30bae178455e9609c89d09d174d`).
Tables are under SATA `frozen_preprocessing/if_replicates_20261014/user_rari_positioning_v1/`.
The approved candidate's points, error bars, palette and significance labels are
retained. Each circle is a mouse; horizontal lines and whiskers show mean ± SEM.
Only significant P/q values and their matching stars are shown.

F uses `replicate_revision_20261014/final_panels.py`. Its `active_release.json`
hash-locks the filtered cohort and density display. The cohort has four DMSO and five RARi mice,
1,568,177 cells and 43,424 P14 cells. Tissue08 remains excluded. All scientific
inputs and original microscopy fields are unchanged. See the
[replicate release README](replicate_revision_20261014/README.md) for paths,
columns, limitations and the review-before-publication command.

Both the standalone E/F/G scripts and complete-page runner read these same
frozen inputs. The previous two-mouse E/F scripts are preserved in the lineage
archive. The caption, rather than text below F, describes IMAP normalization.

The current display grid is in `assembly/design.py`. B is 193.34 pt wide (26.3%
larger than the previous export) without recropping or stretching. Its overlay
and 200-µm bar scale with the image. Top-row images share top/bottom edges and
scale-bar baselines. Paired plots, magnified frames
and colorbars share explicit alignments. Statistical ticks and annotations are
7 pt, axis labels 7.5 pt, and condition headings 8 pt. E/G replicate dots are
25 pt² (47% larger in diameter than the previous 11.5-pt² dots), with unchanged
values, jitter and summary lines. E is 128 × 120 pt; G is 176 pt wide below the
same horizontal center. F's maps narrow to 152 pt each without reducing their
height; 5.7-pt gate labels keep their original relative positions and fit inside
their boxes. F/G share top/bottom plot edges. Tests cover these display constraints.

Panel A preserves its DMSO 180-degree rotation and unrotated RARi region.
Neither snapshot-to-slide registration nor an additional artist color transform
is invented. Scale calibration reproduces the original 79-native-pixel,
50-µm snapshot bar. B uses the verified 0.16247698804244626 µm/full-resolution
pixel calibration. Always scale microscopy and bars together in Illustrator.

B's field is x=16540:22092, y=48600:52649, containing SI1 villi 000, 003 and 005.
Selection used geometry/reference coverage before repaired test metrics.
The archived 821-cell/villi-009/010/031 example is not the repaired training set.

C axes are `Sparse IF reference` and `Image-model prediction`; epithelial
density uses coolwarm. D retains the original 6000×4500-pixel zooms and marks
epithelial color saturation at ≥0.5. F retains vector density contours, gates,
colorbars and Arial text in PDF, plus a 600-dpi PNG; use the PDF in Illustrator.
Other dense PDF layers honor their requested 300-dpi export, not a 100-dpi default.

IMAP labels now follow the user's original box locations and line breaks in the
current `Figures/Figure4.pdf`, with the current frozen percentages. Most labels
are inside the data axes. `imap_label_registration.json` and `imap_labels.py`
share the original vector-box geometry between standalone and final rendering.
The two map frames narrow without shifting labels relative to their maps.

## Scientific release and hash locks

The authorized IF repair retains the production shared-scale-aware architecture
and original Xenium initialization. It removes 2,753 training crops overlapping
validation, leaving 5,822 training, 2,312 validation and the unchanged 4,843 test
identities. All local/context/fine split-pair overlaps are zero. A single
retained-training epithelial-ratio p99, 5.755600843392969, calibrates both
sections and all partitions. Native crop widths are 1024/4097/256 pixels,
resized to 256/256/128 with the frozen crop normalization and centering.

Seven head-stage and four full-stage epochs completed; validation selected
epoch 2 in each. Standard math attention resolved the numerical backward
failure without changing the locked training configuration. No test-based
checkpoint, region, gate or model selection occurred.

| Source | Exact path relative to root | SHA256 |
|---|---|---|
| IF checkpoint | SATA `Figure_4/stored_outputs/scientific_repair_v1/stage2_full/best_crypt_villus_vit_model.pt` | `b512cb49918a77602575a6768ee6ccc1b5fcda04d3fdd10221510aafec0fc7b2` |
| Training/preprocessing lock | SATA `Figure_4/stored_outputs/scientific_repair_v1/lock.json` | `e38db97a2fbd6a2ec3154c7062113d4fc50a5e24872df81ecf44955e900593f8` |
| Strict identity manifest | SATA `splits/if_final_strict_test_v1.json` | `7d9c9120c5ef55a2c30a2fe2b3c0e0fe3b16a111ca61ff5339029ae6e4a2ac48` |
| Prespecified evaluation | SATA `Figure_4/stored_outputs/scientific_repair_v1/evaluation_specification.json` | `b30e64a1a4a38da5f5b6eedae769a5872e8498c856e128139e495c9f3abf42a4` |
| DMSO snapshot | SATA `Data/Histology/histology_snapshots/DMSO.png` | `284b4661d6c1448b8a4ce01bc4a90dfdc49974cf05cee0ea8164c2f87744acca` |
| RARi snapshot | SATA `Data/Histology/histology_snapshots/RAR.png` | `951188d7134fbc709576859bac86033ea2924d793d1f6d5f7e92eb139902d06e` |
| Original zoom configuration | CODE `Figure4/imap_if/config_all_cells_production_v2_if_sft.yaml` | `6447d3cc3ea0e4318be13db878fd0a68da0ad4981562721f33a7e885587dce3d` |
| Frozen style reference | SATA `Figures/Old_Figures/2026_09_04_Figures/Figure4.png` | `2f18c07127cddb4641b42f5fc1690eb3a52ea62debaf9c15027d8bb3034ba858` |

Every renderer requires the completed, hash-verified
`SATA/Figure_4/stored_outputs/scientific_repair_v1/evaluation/release_manifest.json`.
It resolves strict predictions/metrics, whole-tissue predictions, gate tables,
common density arrays, reference version and checkpoint. It never silently uses
the historical `Figure4/production_v2_final/` outputs. Panel provenance records
the full input/output hashes and exact commands.

Required fields: C uses `row_id`, `eligible_final_test`, `target_axis`,
`epithelial_distance_clipped_1p0`, `predicted_axis_coordinate` and
`predicted_epithelial_distance_clipped_1p0`. D additionally requires
`source_id`, `object_id`, full-resolution centroid coordinates, `x_um`,
`y_um` and `class`. E/G read `per_mouse_metrics.csv`, `per_mouse_gate_counts.csv`
and `statistics.csv`: `mouse_id`, `condition`, `experiment`, `p14_per10k`,
`imap_gate`, `log2_enrich`, `metric`, `gate`, `p_ols_batch` and `q_ols_batch`.
F reads the frozen mean density grids and gate-percentage summaries.
B reads the locked illustration
array/cell table and source-image calibration; it never selects example cells.

## Regeneration

From CODE, use the figure environment. To inspect all explicit input commands:

```bash
Figure2/.venv/bin/python Figure_4/run_all_panels.py \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4 \
  --dry-run
```

To archive and replace selected components from frozen outputs:

```bash
Figure2/.venv/bin/python Figure_4/reexport_panels.py \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4 \
  --panels B C D E F G --overwrite
```

The wrapper archives previous components, verifies untouched panels/artwork,
renders from numeric/image inputs and updates the component index. A fresh
directory can use `run_all_panels.py` directly without `--overwrite`.
Rendering never fits, selects, constructs a split, or performs full inference.

Expected PDF/PNG stems:

- A: `panel_A_dmso_IF_overlay`, `panel_A_rari_IF_overlay`.
- B: `panel_B_sparse_IF_training_labels`.
- C: `panel_C_strict_IF_test_scatter`; `panel_C_metrics.json`.
- D: `panel_D_{dmso,rari}_{crypt_villus,epithelial_distance}_whole_and_zoom`.
- E: `panel_E_P14_abundance`.
- F: `panel_F_P14_IMAPs`.
- G: `panel_G_P14_enrichment`.

Each directory includes provenance (`assembly_inputs_provenance.json` for E–G),
embedded Arial vector PDF and PNG. No manually assembled arrows/layout are redrawn.

## Evaluation and interpretation limits

This is a corrected evaluation of a previously examined within-section
crop-disjoint IF cohort, not a new untouched independent test. Epithelial
references are computed from QuPath epithelial classifications, not directly
hand-drawn distance labels. The common IF calibration does not establish
numerical equivalence to the differently calibrated Xenium reference.

Whole-tissue panels are descriptive outputs, not further independent tests.
E–G include nine biological replicates from two experiments after the supplied
tissue removals. E/G use the frozen uploaded analysis, adjusted for experiment;
the original two sections contributed to image-model development. Treatment
randomization and annotation blinding are undocumented. P14 detection calls and
gates remain fixed. Anatomical gate names are provisional,
not independently validated compartment assignments. Abundance is cell-relative,
not tissue-area density; convex hull area is not used. Composition does not
measure absolute abundance. P14 and non-P14 condition maps use the same
per-mouse-normalized estimator, with separate 0–13 and 0–16 color ranges.

See `organization_alignment_report.md` for intentional differences from stale
artist labels. Original models, predictions, regions and assembled artwork are
preserved; the artist must import the corrected components.
