# New IF section review panels

## Current Figure 4 (15 October 2026)

The author-approved A/B candidate is integrated as E (whole-section P14
abundance, upper right) and G (gate enrichment, directly below E). The IMAP
pair is F, side by side at lower left. `approved_ab_release.json`
locks the selected candidate provenance and the frozen uploaded analysis.
E/G retain the approved palette, biological-replicate circles, mean ± SEM,
and significance-only stars with P/q values. The tests are experiment-adjusted
OLS, with BH correction across five gates for G. No new tests or fits run during
figure generation. Earlier exact-permutation panels below are historical.

F retains its frozen grids, percentages and 0–13 colorbar. Its explanatory
footers are in the caption. The latest display pass enlarged B and aligned
image edges, axes, zooms and colorbars across the page, with more legible
statistical labels. No scientific values or other current figures changed.
The complete page is SATA `Regenerated_Figures/Current/Figure4.pdf`.
Regenerate it with `python -B update_final_figure.py Figure_4 --overwrite`.
The standalone E/F/G commands are listed by `Figure_4/run_all_panels.py --dry-run`
with an explicit output root; they use the same frozen inputs and plot helpers.

The source analysis is SATA
`frozen_preprocessing/if_replicates_20261014/user_rari_positioning_v1/manifest.json`,
SHA256 `3fff265a5e93948569c14136b8c15aa49281e30bae178455e9609c89d09d174d`.
The caption describes the normalization, tests, mouse counts and equal-weight
IMAP means. Integration QA is in SATA
`Figure_Run_Archive/if_replicates_20261015_approved_efg_v1/visual_qa_and_publication.json`.
The current candidate collection is resolved by `candidate_plot_set.json`;
its paper-figure copies are historical snapshots, not the current publication.

## Previous potential plot set: spatial correlations

The current PDF-only folder is SATA
`Figure_Run_Archive/if_replicates_20261014/potential_plots_post_removals_v5/pdfs/`.
The parent folder's `All_IF_candidate_plots.pdf` has 44 pages from 35 PDFs.
The first two pages are the new 2D P14 spatial-distribution correlation matrices;
all previous candidate pages follow unchanged.

`Additional_12_P14_2D_spatial_correlation.pdf` compares each pair of SIs using
Pearson correlation between their full joint crypt–villus × epithelial-distance
P14 maps. Each map is normalized to unit mass. The maps use the existing locked
100 × 100 coordinate grid and Gaussian smoothing (sigma 2 grid bins, reflect
boundary mode). All retained P14 cells are included. Rows and columns use the
same treatment-grouped SI order, with the original mice labeled. Both PDFs use
the full −1 to 1 Pearson scale, with every coefficient printed.

`Additional_13_P14_2D_correlation_occupied_bins.pdf` repeats the comparison on
the 4,405 bins containing at least one observed P14 cell across all nine SIs.
One shared, treatment-independent mask is applied to every pair. Correlations
can change by as much as 0.32 after excluding unoccupied bins. This sensitivity
therefore accompanies the primary full-grid matrix. Neither map normalization
nor the sensitivity uses non-P14 background correction. These comparisons
describe distribution shape; total P14 abundance is a separate endpoint.

Pair correlations share mice and neighboring grid values are dependent. No
bin-level P values, confidence intervals or treatment significance tests are
reported. Correlation is between two tissues' 2D density maps, not between CV
and epithelial distance within one tissue or matched cells across tissues.
The original RARi map also correlates strongly with DMSO maps; these matrices
should not be interpreted as clean treatment-only separation.

Frozen values are in SATA
`frozen_preprocessing/if_replicates_20261014/p14_spatial_correlation_v1/manifest.json`,
SHA256 `63051d8652df29017a28fe0d229f27cff53347c6792f4c9aa471a4a30adc4ff8`.
Inputs are the post-removal `sections.csv`, `densities.npz` and per-cell Parquets.
The per-cell fields are `is_p14`, `predicted_axis_coordinate` and
`predicted_epithelial_distance_clipped_1p0`. Preparation confirms that every P14
cell is inside the grid and reconstructs each saved map from the coordinates.
Both correlation matrices, all 36 unique pairs, descriptive pair-group summaries,
coverage checks, normalized grids and the exact shared mask are frozen before
rendering. Independent tests reproduce the matrices with centered dot products
and SciPy Pearson coefficients.

Regenerate into new directories; existing outputs are protected:

```bash
/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  frozen_preprocessing/if_replicates_20261014/prepare_p14_spatial_correlation.py \
  --release /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/nine_mouse_removal_release_v1/manifest.json \
  --release-sha256 613bc495ad9fb518e0d47d34f1ed72fd860f002502dd04ba626132bb7879ccc0 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/p14_spatial_correlation_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/replicate_revision_20261014/figure_script/render_p14_spatial_correlation.py \
  --manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/p14_spatial_correlation_v1/manifest.json \
  --manifest-sha256 63051d8652df29017a28fe0d229f27cff53347c6792f4c9aa471a4a30adc4ff8 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/p14_spatial_correlation_v1
```

Rendering exports both named PDFs, their 300-dpi PNGs and provenance. All tiles
and labels remain vector in PDF, with embedded Arial. The collection contains
copies; `extend_candidate_collection.py` records its exact command and verifies
all 44 assembled pages against their sources. The eight current manuscript
figures, predictions, gates and model weights are unchanged.

## Previous potential plot set: twenty visualization options

The current PDF-only folder is SATA
`Figure_Run_Archive/if_replicates_20261014/potential_plots_post_removals_v4/pdfs/`.
The parent folder's `All_IF_candidate_plots.pdf` contains 42 pages in 33 PDFs.
The twenty new options are first; the previous 22 pages follow unchanged.

For just the twenty options, open SATA
`Figure_Run_Archive/if_replicates_20261014/p14_visualization_options_v2/Twenty_P14_visualization_options.pdf`.
Its `pdfs/` directory holds the twenty individual PDFs, with a page-by-page index
in the parent README. Each page also has a 300-dpi PNG. All plotting text is Arial.
The PDFs retain vector plots, labels and axes.

The options cover gate enrichment, P14 composition, cell-count-normalized
density, per-mouse regional contrasts, heatmaps, experiment-specific comparisons,
and a combined density/composition view. Options 01, 09, 12 and 14 give the
clearest complementary views. The observed pattern is lower Top LP density
and greater relative muscularis enrichment under RARi. Density is lower under
RARi in muscularis as well; a larger muscularis P14 share is a relative change.
The existing experiment-stratified gate tests remain nonsignificant after Holm
correction. No new significance tests or paper-figure edits were made.

Input tables and descriptive summaries are frozen in SATA
`frozen_preprocessing/if_replicates_20261014/p14_visualization_suite_v1/manifest.json`,
SHA256 `d0145653b1620a080eb86b3960dbc11553a0916745e6805d199b2186d3e970e4`.
Inputs resolve through the previously locked heatmap, composition, fold-change
and observed/expected releases. Required fields are mouse IDs, source IDs,
condition, cohort, gates, cell counts, density, gate percentages and enrichment.
Preparation saves per-mouse ratios and contrasts, condition means and SEM,
and experiment-specific composition changes before rendering.

Regenerate into new output directories:

```bash
/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  frozen_preprocessing/if_replicates_20261014/prepare_p14_visualization_suite.py \
  --release-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/p14_visualization_suite_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/replicate_revision_20261014/figure_script/render_p14_visualization_suite.py \
  --manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/p14_visualization_suite_v1/manifest.json \
  --manifest-sha256 d0145653b1620a080eb86b3960dbc11553a0916745e6805d199b2186d3e970e4 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/p14_visualization_options_v2
```

The suite plan lists all twenty output stems (`Option_01_…` to `Option_20_…`).
Both commands protect existing directories. `qa_p14_visualization_suite.py`
independently reproduces the table arithmetic, checks all fixed axis bounds,
and verifies the twenty combined pages against their source PDFs. All twenty
layouts were inspected. `extend_candidate_collection.py` adds their copies
to the full collection and verifies all 42 pages. The eight current manuscript
figure PDFs remain unchanged. Exact commands and hashes are stored beside each
output; `candidate_plot_set.json` records the current collection.

## Previous potential plot set: row-normalized heatmap addition

The current PDF-only folder is SATA
`Figure_Run_Archive/if_replicates_20261014/potential_plots_post_removals_v3/pdfs/`.
The parent folder's `All_IF_candidate_plots.pdf` has 22 pages and 13 bookmarks.
Page 1 is `Additional_11_P14_density_row_normalized_heatmap.pdf`; the fold-change
plot is page 2. All previous candidate pages are unchanged.

The heatmap shows nine SIs as rows and five gates as columns. Density is P14
per 10,000 non-P14 cells within the same gate. Each row is divided by its mean
across the five gates, giving a row mean of 1. DMSO rows appear above RARi;
the original pair is labeled. The shared color scale runs from 0 to 2.5,
centered on 1, with all 45 values printed. Tiles and labels are vector in PDF.
This view compares regional patterns after removing overall density differences
between mice. It uses equal gate weights, unlike the whole-tissue denominator
of observed/expected enrichment. No new tests or batch correction are applied.

Frozen data are in SATA
`frozen_preprocessing/if_replicates_20261014/p14_density_heatmap_after_removals_v1/`.
The manifest SHA256 is
`b44f301f9a17c0cb62331421829d55d6d403c22565349961b4f0bb0cb45d744a`.
It contains raw and normalized density tables, row identities and the exact
normalization definition. The source cohort, mouse IDs, treatment labels,
counts and gate assignments remain locked to the post-removal release.

Preparation and rendering commands (existing output directories are protected):

```bash
/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  frozen_preprocessing/if_replicates_20261014/prepare_p14_density_heatmap.py \
  --release /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/nine_mouse_removal_release_v1/manifest.json \
  --release-sha256 613bc495ad9fb518e0d47d34f1ed72fd860f002502dd04ba626132bb7879ccc0 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/p14_density_heatmap_after_removals_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/replicate_revision_20261014/figure_script/render_p14_density_heatmap.py \
  --manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/p14_density_heatmap_after_removals_v1/manifest.json \
  --manifest-sha256 b44f301f9a17c0cb62331421829d55d6d403c22565349961b4f0bb0cb45d744a \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/p14_density_heatmap_v1
```

Rendering writes `Additional_11_P14_density_row_normalized_heatmap.pdf`, a
300-dpi PNG and provenance. `extend_candidate_collection.py` copies the reviewed
PDFs and rebuilds the book; its exact command is in `plot_set_manifest.json`.
Independent tests reproduce all 45 cells of the heatmap from the cell counts.
All 22 book pages match their source PDFs, all 13 PDFs have embedded Arial
and in-bounds text, and the eight current paper figures remain unchanged.

## Previous potential plot set: density fold-change addition

The current PDF-only folder is SATA
`Figure_Run_Archive/if_replicates_20261014/potential_plots_post_removals_v2/pdfs/`.
The parent folder's `All_IF_candidate_plots.pdf` has 21 pages and 12 bookmarks.
Page 1 is `Additional_10_P14_density_fold_change.pdf`; all 20 earlier pages
follow unchanged. `candidate_plot_set.json` records the current paths.

The new plot shows RARi/DMSO fold change in P14 cells per 10,000 non-P14 cells
within the same gate. Each mouse's density is calculated separately. Within
each experiment, densities are averaged by condition; both condition means are
then standardized to the same experiment weights (original 2/9, new 7/9).
Bars show the ratio of these standardized means. The two points show the
original and new experiment ratios, not individual mice or confidence limits.

The author confirmed that the original DMSO and RARi mice belong to one earlier
experiment and that treatment was assigned to individual mice within each
experiment. The new plot therefore uses exact within-experiment permutations:
70 assignments, a log-ratio statistic, twice the smaller tail, and Holm correction
across five gates. All adjusted P values are 0.143. This analysis is exploratory;
the previous composition plots and paper figures retain their earlier tests.
Model predictions, gates, tissue removals and manuscript documents are unchanged.

Frozen statistics: SATA
`frozen_preprocessing/if_replicates_20261014/p14_density_fold_change_after_removals_v1/manifest.json`,
SHA256 `6065e6afffd23fc61f3e2fe67545e702a0798aa7bdda51e213194c826b80ed79`.
The source cohort manifest remains locked at
`613bc495ad9fb518e0d47d34f1ed72fd860f002502dd04ba626132bb7879ccc0`.
Required source fields are mouse and source IDs, condition, cohort, gate,
`n_p14`, `n_non_p14`, and `p14_per_10000_non_p14`. The renderer reads
`density_fold_change.csv`, `experiment_fold_change.csv` and the analysis plan;
it performs no statistical tests. Numeric summaries and all 70 assignments per
gate are saved in the frozen release. No interval is estimated from the two
experiment points. Background-cell counts measure cellular availability rather
than compartment area; removal-annotation blinding remains undocumented.

Regenerate into new directories; these commands refuse to replace existing outputs:

```bash
/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  frozen_preprocessing/if_replicates_20261014/prepare_p14_density_fold_change.py \
  --release /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/nine_mouse_removal_release_v1/manifest.json \
  --plan frozen_preprocessing/if_replicates_20261014/p14_density_fold_change_plan_v1.json \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/p14_density_fold_change_after_removals_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/replicate_revision_20261014/figure_script/render_p14_density_fold_change.py \
  --manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/p14_density_fold_change_after_removals_v1/manifest.json \
  --manifest-sha256 6065e6afffd23fc61f3e2fe67545e702a0798aa7bdda51e213194c826b80ed79 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/p14_density_fold_change_v1
```

Rendering produces `Additional_10_P14_density_fold_change.pdf`, its 300-dpi PNG
and `provenance.json`. The current collection was assembled with
`extend_potential_plot_set.py`; its exact command is in `plot_set_manifest.json`.
All 21 combined pages match their source PDFs pixel-for-pixel, all 12 source
PDFs have embedded Arial and in-bounds text, and the eight current paper PDFs
are unchanged. Independent tests in `test_p14_density_fold_change.py` reproduce
the denominators, ratios, complete permutation distributions and Holm adjustment.

## Previous potential plot set: composition addition

All candidate PDFs are together at SATA
`Figure_Run_Archive/if_replicates_20261014/potential_plots_post_removals_v1/pdfs/`.
`All_IF_candidate_plots.pdf` in the parent folder combines them into 20 pages
with bookmarks. The new grouped percentage bars and their statistics table
are pages 1–2. The folder contains 11 PDFs:

- `Additional_01`–`06`: refreshed regional comparisons, individual P14 maps,
  composition bars, new-section maps, gate overlays and individual non-P14 maps.
- `Additional_07_removals_and_enrichment_review.pdf`: the existing ten-page
  post-removal analysis and exclusion review.
- `Additional_08_P14_composition_bars.pdf`: DMSO/RARi bars for each gate's
  percentage of the mouse's total P14 cells, with individual mice and mean ± SEM.
- `Additional_09_P14_composition_statistics.pdf`: primary and sensitivity tests.
- `Figure4.pdf` and `Figure4_extended.pdf`: copies of the current paper figures.

Every candidate uses the filtered cohort. The older `nine_mouse_update_v4/pdfs/`
set is preserved as a pre-removal version. This addition does not change paper
figures, model predictions, gates or manuscript documents.

Composition statistics are frozen in SATA
`frozen_preprocessing/if_replicates_20261014/p14_composition_after_removals_v1/manifest.json`
(SHA256 `c951835cd8a902c9273ac042b376c103c33e1f47df1f482ed053158e2cff266e`).
The plan is `frozen_preprocessing/if_replicates_20261014/p14_composition_plan_after_removals_v1.json`.
The primary comparison uses four DMSO and five RARi mice, exact two-sided
mouse-label permutations and Holm correction across five gates. The denominator
is each mouse's total retained P14 count. No gate has adjusted P < 0.05.
This is an exploratory within-P14 distribution endpoint, separate from the
background-normalized enrichment endpoint in the paper figure.

Preprocessing: `prepare_p14_composition_statistics.py` under the frozen package.
Rendering: `figure_script/render_filtered_plot_set.py` and
`figure_script/render_p14_composition_candidate.py`. Compilation and checks:
`assemble_potential_plot_set.py` and `qa_potential_plot_set.py`.
Each accepts explicit input/output paths; exact commands, hashes and versions
are saved beside its outputs. `plot_set_manifest.json` records the PDF copies
and page order; `visual_qa.json` records the compiled-page comparison.

## Current Figure 4: all tissue removals applied

`active_release.json` locks three manifests under SATA
`frozen_preprocessing/if_replicates_20261014/`:

| Release | Manifest SHA256 |
|---|---|
| `nine_mouse_removal_release_v1/manifest.json` | `613bc495ad9fb518e0d47d34f1ed72fd860f002502dd04ba626132bb7879ccc0` |
| `density_over_expected_after_removals_v1/manifest.json` | `03680195a5e1cca0f6475b798b0b6359d396fd430fcc0189509ae08d7b6d346b` |
| `figure_display_after_removals_v1/manifest.json` | `bb66eb85f1fcf4d1101a6b79116d04eb7e11b7c354596f7f4acf92ab404aac5f` |

The cohort contains four DMSO and five RARi mice: the original pair and new
tissues 01–07. All 77 supplied removal regions are applied. The retained
1,568,177 cells include 43,424 P14 cells. Tissue08 remains excluded; its source
data are preserved.

Main E keeps the dumbbell/lollipop layout. At the author's request, the left
subplot again shows P14 per 10,000 non-P14 cells in the same gate, with mouse
means ± SEM from the filtered cohort. The right retains the observed/expected
endpoint: `(P14 in gate / total P14) / (non-P14 in gate / total non-P14)`,
calculated separately for each mouse. It shows DMSO minus RARi ± SEM.
Each gate has one Holm-adjusted P label in the right subplot only. Tests enumerate
all 126 mouse-label assignments, with the unchanged doubled-smaller-tail
two-sided definition and Holm correction across five gates. No adjusted P is
below 0.05. New-only and cohort-stratified results remain in the frozen tables.

Main F and Extended E show filtered equal-mouse mean densities and mean gate
percentages. The shared ranges remain 0–13 for P14 and 0–16 for non-P14;
the filtered maxima are 12.747 and 15.742, respectively. No density is clipped.
Main A–D and Extended A–D/F retain their original data and layout, including
the original Extended D microscopy fields. Model weights, gates and retained-cell
predictions are unchanged.

`final_panels.py` reads `sections.csv`, `gates.csv`, `group_summary.csv`,
`densities.npz`, and `density_over_expected_statistics.csv` through these
manifests. Group summaries require condition, gate, metric, mean and SEM.
Statistics require analysis, metric, gate, group sizes, means, SEM,
`difference_dmso_minus_rari`, `difference_sem`, raw P and `holm_adjusted_p`.
Rendering verifies the cohort/statistics match and runs no tests or inference.

Render the restored-left main figure, inspect it, then publish:

```bash
/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  regenerate_complete_figures.py --figures Figure_4 \
  --run-id if_replicates_20261014_left_abundance_v1

/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  Figure_4/replicate_revision_20261014/publish_left_abundance.py \
  --run /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014_left_abundance_v1 \
  --previous-run /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014_post_removals_v2 \
  --publish-visually-reviewed
```

Use a new run ID for another render. Omit the last flag for a read-only
publication audit. The audit requires all page pixels outside E's left subplot
to be identical. Publication archives old exports and updates only `Figure4.pdf`
in SATA `Regenerated_Figures/Current/`. Per-letter PDF/600-dpi PNG exports and
provenance go to SATA `Figure_4/panel_X/`. Extended Figure 4 remains the reviewed
post-removal export from `if_replicates_20261014_post_removals_v2`.

This remains an exploratory mouse-level comparison. Background-cell counts
measure cellular availability, not compartment area. Treatment randomization
and removal-annotation blinding are undocumented. Removals followed inspection
of earlier results; the original pair contributed to model development.

The following sections document earlier releases. Their review PDFs, analysis
definitions and counts are historical, not the current manuscript figures.

## Historical combined review PDF and independent P-value audit

The 11-page review PDF is at SATA
`Figure_Run_Archive/if_replicates_20261014/IF_quantification_review_v1/IF_quantification_review.pdf`.
It contains a contents page, a P-value audit, both complete figures and all seven
current companion pages. Bookmarks and native PDF page dimensions are retained.
This is a review compilation, not panel regeneration or a replacement publication.

The independent audit verified the per-cell gate assignments and denominators,
then reproduced all 30 tests using a separate enumeration and statsmodels Holm
correction. The repeated Figure 4e abundance P=0.040 labels are not a copying
error. Four raw absolute-tail P values are 1/126 and one is 4/126; Holm maps all
five to 5/126. A different two-sided convention, twice the smaller signed tail
(SciPy's default), gives adjusted P=0.079 for all five. Neither convention has
been selected after seeing which is significant. The current paper figures and
frozen primary analyses remain unchanged, and the review explains why the
significance claim is sensitive to the convention.

Audit preprocessing and compilation are separate:

```bash
python frozen_preprocessing/if_replicates_20261014/audit_regional_statistics.py \
  --pointer lineage_archive/if_replicates_20261014_before_removal_figures/Figure_4/replicate_revision_20261014/active_release.json \
  --current-figures /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Regenerated_Figures/Current \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/independent_statistics_audit_v1

python Figure_4/replicate_revision_20261014/build_if_review_pdf.py \
  --publication /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/nine_mouse_update_v4/publication.json \
  --audit-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/independent_statistics_audit_v1/manifest.json \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/IF_quantification_review_v1
```

These directories already exist. Both commands refuse replacement. The audit
manifest SHA256 is `1e964d1b5289249b4675dcf136e5b38f4567cbb18944d41bac7f02501e6f2668`.
It locks the audit CSV, all exact null distributions, conclusions, source data
and code. The compilation manifest locks every input/output PDF and page order.
`qa_if_review_pdf.py --review-dir PATH` compares all nine copied figure pages
pixel-for-pixel against their source PDFs and checks bookmarks and text bounds.

## Historical pre-removal release: nine mice, tissue08 excluded

The author requested tissue08's exclusion after the early previews. The current
analysis has four DMSO and five RARi mice, including the original pair. Source
annotations and predictions are preserved; the 41 Transferred calls in tissue 08
are not rewritten as zero. `active_release.json` points to the frozen release:

`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/nine_mouse_release_v2/manifest.json`

SHA256: `e791f7552ed5ccee1f90dbe1e62a1a1bcc6221881e81c1380a87b70abb908efe`.

The canonical Figure4 runner uses `final_panels.py` for E/F; Extended Figure4
uses it for E. Main E restores the abundance dumbbells and signed composition
lollipops, using mouse means and SEM. Its connections join group means, not paired
animals. Individual mice remain in the companion panels.

Main F uses equal-mouse-mean frozen P14 densities on a shared 0–13 scale. The
largest density is 12.122; no values are clipped or changed. Extended E retains
the non-P14 0–16 scale. Label positions, percentages and gate thresholds are
unchanged. Extended D restores the exact original DMSO/RARi microscopy fields,
contrast, marker styling and aspect ratio from the hash-locked
`gate_backprojection_v1/manifest.json`. New-section fields remain in the companion
panels. Other panels retain the original model examples and evaluations.

`active_release.json` also locks `regional_statistics_v1/manifest.json` (SHA256
`4a3697641a4dabf9454e7881544cd3cd95136364f8e06cf192794bcc4c3f1e47`).
The analysis enumerates all 126 unpaired mouse-label allocations. Two-sided P
values use absolute mean differences; Holm correction covers five gates per
endpoint. All nine mice are the primary analysis. New-only and cohort-stratified
tests are reported alongside it, without selecting the most favorable result.
These exploratory tests assume exchangeability; treatment randomization is not
documented. The original mice contributed to model development, and tissue08's
exclusion followed interim review.

Regenerate the complete pages and canonical final panels:

```bash
/home/amonell/Desktop/spatial_perturb/2026_07_02_Liver_Perturb/.venv/bin/python -B \
  regenerate_complete_figures.py --figures Figure_4 Figure_4_Extended \
  --publish-panels --overwrite
```

Create the six companion PDFs from the same frozen release:

```bash
/tmp/spatialtrace-main-release.jChYCO/cuda/bin/python \
  Figure_4/replicate_revision_20261014/figure_script/render_replicate_extras.py \
  --input /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/nine_mouse_release_v2 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/nine_mouse_update_v3
```

Use a new output directory for another rendering. Companion pages are regional
consistency, individual P14 IMAPs, regional composition, all seven new whole-section
maps, all seven new fixed-field gate backprojections, and individual non-P14 IMAPs.
Their PDFs share the output's `pdfs/` folder; PNGs and provenance stay outside it.
The frozen release contains all per-cell and per-mouse tables, densities, small
microscopy arrays and hashes. Raw-slide access or model execution is not needed
for rendering. New tissue predictions have no coordinate reference labels;
original model accuracy is a separate analysis. Main E displays the primary
Holm-adjusted P values. The statistical companion reports both sensitivity
analyses, means, SEM and differences. Both full-cohort and new-only summaries
remain saved for cohort checks.

Statistics are explicitly frozen before plotting:

```bash
python frozen_preprocessing/if_replicates_20261014/prepare_regional_statistics.py \
  --release /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/nine_mouse_release_v2/manifest.json \
  --plan frozen_preprocessing/if_replicates_20261014/regional_statistics_plan_v1.json \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/regional_statistics_v1
```

That frozen output already exists. The preparation command refuses replacement.
The renderer reads `regional_statistics.csv` (analysis, metric, gate, group sizes,
means, SEM, mean difference, test statistic, raw P and Holm-adjusted P) and performs
no tests:

```bash
python Figure_4/replicate_revision_20261014/figure_script/render_regional_statistics.py \
  --manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/regional_statistics_v1/manifest.json \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/regional_statistics_review_v1
```

This writes `Additional_07_regional_statistics.pdf`, its PNG and provenance.

The sections below document the earlier eight-section/ten-mouse plan and interim
snapshot. They are retained for provenance, not the active final cohort.

This package renders the October 2026 sections from frozen inference outputs.
It does not replace the submitted Figure 4 or Extended Figure 4. The scanned
labels identify images 01–04 as RARi and 05–08 as DMSO. The author confirmed
eight different mice and inclusion of the original pair, giving five mice per
condition. The verified key is stored with the preprocessing outputs.

## Panels

All pages are generated from cell tables and native microscopy arrays:

- `Additional_01_all_section_maps.pdf`: DAPI/E-cadherin, crypt–villus predictions
  and epithelial-distance predictions for all eight sections. Every prediction
  cell is shown. Coordinate ranges are shared across sections.
- `Additional_02_P14_IMAPs.pdf`: individual-section P14 coordinate distributions
  and percentages in the original Figure 4 gates.
- `Additional_03_non_P14_IMAPs.pdf`: matching background-cell distributions.
- `Additional_04_gate_backprojection.pdf`: microscopy beside the same field
  colored by coordinate gate. E-cadherin remains visible; P14 uses larger filled
  points. Display fields were fixed from raw images before inference.
- `Additional_05_section_abundance_composition.pdf`: gate-specific abundance,
  within-P14 composition and whole-section P14 fraction. Numbers above the
  whole-section points are P14 counts.
- `Additional_06_ten_mouse_comparison.pdf`: gate-specific abundance and
  composition for all ten mice, with individual points and mean ± SEM. Original
  mice use open squares and new mice use filled circles.

## Inputs

SATA `frozen_preprocessing/if_replicates_20261014/analysis/manifest.json` locks the
per-cell Parquet files and summary tables. Required fields include `source_id`,
`cell_id`, `centroid_x_fullres_px`, `centroid_y_fullres_px`, `is_p14`, `gate_name`,
`predicted_axis_coordinate`, and `predicted_epithelial_distance_clipped_1p0`.
Image caches and exact calibration are locked by the corresponding image
manifests. The alignment review fixes display fields. Renderers stop if an input
checksum changes.

```bash
/tmp/spatialtrace-main-release.jChYCO/cuda/bin/python \
  Figure_4/replicate_revision_20261014/figure_script/render_review_panels.py \
  --root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014
```

The output's `pdfs/` folder contains only PDFs. PNG previews and provenance are
kept beside it. Existing exports are not overwritten; choose a new review output
directory for a changed rendering. Arial text and axes stay vector; dense
non-P14/spatial points are rasterized at 600 dpi. P14 IMAP points remain vector.

The first five pages show the eight new sections; the sixth includes the original
pair. Each section represents one distinct mouse, as confirmed by the author.
Section 08's 41 P14 cells are retained. These data contain no new coordinate
ground truth, so the paper's original accuracy panels must remain explicitly
separate from this new-section application.

## Interim regional preview

`render_early_preview.py` reads a separate snapshot of fully completed coordinate
runs. It does not wait for the Peyer's-patch classifier and does not read partial
prediction files. The original pair is shown as a labeled reference. All mice
available when the snapshot starts are included; unfinished sections are listed
in its manifest. No treatment tests are performed on the incomplete cohort.

The three pages show regional P14 percentages, gate-specific abundance and
enrichment (`Preview_01_regional_consistency.pdf`), individual P14 coordinate
maps (`Preview_02_individual_P14_IMAPs.pdf`), and P14 versus background regional
composition (`Preview_03_regional_composition.pdf`). Enrichment is the fraction
of P14 in a gate divided by the fraction of non-P14 in that gate, within a mouse.
It adjusts for background-cell representation, not anatomical area. Agreement
between distributions does not establish coordinate accuracy.

Prepare the snapshot explicitly before rendering:

```bash
/tmp/spatialtrace-main-release.jChYCO/cuda/bin/python \
  frozen_preprocessing/if_replicates_20261014/prepare_early_preview.py \
  --root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014 \
  --original-release /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/scientific_repair_v1/evaluation/release_manifest.json \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/early_preview_v1

/tmp/spatialtrace-main-release.jChYCO/cuda/bin/python \
  Figure_4/replicate_revision_20261014/figure_script/render_early_preview.py \
  --input /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/if_replicates_20261014/early_preview_v1 \
  --output /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_Run_Archive/if_replicates_20261014/early_preview_v1
```

Both commands refuse existing output directories. Use a new version suffix for
a later snapshot. Its manifest locks the completed predictions, the scanned-label
sample key and the paper gate definitions. The renderer verifies snapshot hashes.
PDFs are in `pdfs/`; PNGs and provenance stay outside that folder. Preview files
never replace final panel components or the submitted figure PDFs.
