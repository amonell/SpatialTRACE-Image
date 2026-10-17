# Complete, fixed-position manuscript figures

`regenerate_complete_figures.py` regenerates data components and assembles all
eight pages from frozen scientific inputs. The September 5 PDFs under SATA
`Figures/` remain immutable layout references. They are never copied into a
complete-page background, overwritten, or used as scientific input tables.

## Regeneration

From the home project, using the configured Python environment:

```bash
python -B regenerate_complete_figures.py --verify-only
python -B regenerate_complete_figures.py --dry-run
python -B regenerate_complete_figures.py
python -B regenerate_complete_figures.py --figures Figure_3 Figure_3_Extended
python -B regenerate_complete_figures.py --publish-panels --overwrite
python -B update_final_figure.py Figure_3 --overwrite
```

The default output is a timestamped run under SATA `Regenerated_Figures/`.
`--output-root` changes the parent directory; `--run-id` chooses the dedicated
run name. Existing runs are refused unless `--overwrite` is given. That flag
archives only a directory carrying this runner's ownership manifest. It never
deletes earlier assets or alters original artwork.

Fresh panel assets live under each mirrored SATA figure package's
`stored_outputs/complete_figure_runs/<generation>/`. The run directory contains
the eight vector PDFs, PNG previews, per-page provenance, `complete_figure_run.json`,
and `review.html`. Figure 4 pages have 600-dpi PNGs; other full-page PNGs use
300 dpi. PDF layers retain the figure-specific vector/raster requirements.

Every complete figure is now assembled from its own final per-letter panel
PDFs, generated from the same fresh scientific artists, text, and original
manual vectors. The runner requires pixel-identical recomposition against the
direct fresh assembly. It saves these files under `panels/Figure_N/` in the run.
`--publish-panels` additionally publishes `panel_X_final.pdf`, its PNG and
provenance into the canonical SATA `Figure_N/panel_X/` directory. A package-level
`final_panel_manifest.json` records every file, hash, source script, layout and
exact update command. These panels use a transparent full-page registered
canvas; import at the page origin with no resizing. They contain only their own
panel's content, including manual schematic artwork where applicable.

`update_final_figure.py Figure_3 --overwrite` is the public one-figure update
command. It refreshes that figure's full panel set together so shared fonts and
layouts stay consistent. Previous final files are recoverably archived, and a
failed publication restores the previous files. Unrelated data, artist folders,
other figures, original artwork and frozen inputs remain in place. Low-level
component runners still provide independently sized microscopy/data assets;
their styling is not the final registered page styling. Use the new final panel
exports when updating the finished figure.

## Layout and preparation contract

Each organized package has `assembly/layout.json` and `assembly/prepare.py`.
The manifest locks the reference hash, visible panel order, A4 page dimensions,
expected text and measured geometry anchors. Coordinates are points from the
page's top-left. Text y values are baselines. No automatic grid, tight-layout
or whole-panel font scaling is applied by the compositor.

`prepare(output_dir, overwrite=False)` returns an object with:

- `elements`: freshly generated PDF layers and positioned Arial text;
- `inputs`: explicit source paths, SHA256 hashes and roles;
- `rendered_anchors`: actual measured final axes rectangles;
- `commands`, `warnings`, and `intentional_differences` as applicable.

PDF elements use `id`, `kind="pdf"`, output-relative `path`, and
`rect_pt=[x,y,width,height]`. Optional `source_rect_pt`, `fit="contain"` and
`alignment="top-left"` support bounded placement; isotropic final-size
components are preferred. Text elements specify `text`, `x_pt`, `y_pt`,
`font_size_pt`, optional Arial-family `font`, `align`, RGB `color`, and
`rotation_degrees`. Element `z` controls stacking, with list order as the tie-break.

Static vectors are structurally selected from the hash-locked reference into
fresh assets using `static_art.py`. Old text, images, scientific path groups,
private Illustrator payload and unrelated resources are excluded. Shapes are
not algorithmically redrawn. Every scientific visual is regenerated from its
numeric, spatial or microscopy inputs. An ambiguous static group is a review
failure, not an excuse to copy the original page.

## Verification and limitations

The root runner verifies scientific release gates before creating outputs.
Rendering performs no training, checkpoint selection, split construction,
full-dataset inference, or unrequested preprocessing. Package renderers retain
their frozen-lineage checks. Current scientific values replace older values
still displayed in the layout references.

Automatic QA checks page dimensions, measured anchors (0.1-point tolerance),
embedded Arial, expected/obsolete text, private source payload, and uniform
placement. Side-by-side pages, overlays and contrast-enhanced differences are
provided for visual inspection. Global pixel equality is not a pass criterion:
the user approved current science and Arial in place of historical values and
mixed fonts. Figure 4b additionally has an explicitly approved aspect-ratio
correction; the crop is not stretched or recut.

A successful script run reports `AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED`.
That is not a claim that every manual shape and label has passed human/visual
inspection. Failures retain their artifacts and diagnostic logs for repair.

Synthetic tests (no manuscript rendering):

```bash
python -B -m unittest discover -s tests -p 'test_complete_figure_assembly.py' -v
```
