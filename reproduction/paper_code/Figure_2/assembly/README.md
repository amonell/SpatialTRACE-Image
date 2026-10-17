# Fixed-coordinate complete-page assembly

`layout.json` registers the reviewed September 5 PDF hash, its A4 page geometry,
panel order, component and axes positions in top-left PDF points, and explicit
scientific/presentation differences. Every data component is rendered anew from
the verified frozen inputs; the reference PDF is a layout reference only.

The public API is `prepare(output_dir: Path, overwrite=False)` in `prepare.py`.
It returns fresh component elements, input hashes, exact subprocess commands,
actual final Matplotlib axes bounds, warnings and intentional differences for
the shared `figure_assembly` compositor. Output PDF paths are relative to the
supplied preparation directory. The shared compositor creates a new document;
the artist's assembled file is never overwritten.

For an isolated render check from the home project root:

```bash
python -B -c "from pathlib import Path; from Figure_2.assembly.prepare import prepare; prepare(Path('/tmp/Figure_2_assembly_check'))"
```

The default invokes the existing active panel scripts through
`Figure_2/assembly/fixed_render.py`. Their scientific-input and checkpoint checks
remain active. The adapter changes only canvas size, axes positions, Arial
typography and declared presentation options; it disables automatic/tight
layout. Every PDF has its `.geometry.json` beside it. Existing renderer
provenance records retain data paths/hashes and evaluation scope. No training,
checkpoint selection, inference-corpus creation or preprocessing is invoked.

The main Figure 2 additionally uses a reviewed original-path allowlist for its
manual schematic and inset connectors, partitioned into panel-A, panel-B, and
panel-C vector assets. The extracted assets contain neither
original text nor images/private Illustrator source. Its labels are retyped in
Arial, and its data examples are fresh plots. The other three packages do not
need an original-artwork background or static vector asset.

Every returned element, including each external title or legend label, has an
explicit `panel` letter. Figure 2A's three numerical insets are rendered by
`panel_A/figure_script/make_panel_A_schematic_insets.py`, also called by the
ordinary panel runner. Their five original image/colorbar footprints are
registered alongside the existing axes. No assembly-only scientific plotting
function remains. Prepared roots are passed explicitly to the runners; the
adapter does not rewrite paths in printed shell commands.

Acceptance requires the shared compositor's geometry, expected/forbidden text,
Arial embedding and private-content checks, followed by visual review of the
new full page. Declared negative-R² values, current repaired classifier results
and current evaluation labels must remain visible even when the older artwork
shows different values or experiment ranges.

`source_registration.json` independently records original image CTMs, measured
vector-spine bounds, and direct PDF text-paint baselines. These are the source
anchors in `layout.json`, not measurements inferred from the regenerated page.
Preparation writes `source_geometry_comparison.json` and stops if any registered
physical footprint differs from its original source by more than 0.1 pt.
Transparent page-sized logical components avoid clipping titles/labels while
remaining one-to-one physical-size PDF placements. Original Illustrator image
scaling is reflected in the registered data viewport. All frozen rows are
supplied to the plot; the source-registered MDE viewport clips three extreme
points from the 989,431-row prepared embedding. Other maps retain their locked
full-section or zoom bounds. This viewport choice does not refit the embedding.
The original low-resolution raster appearance and Arial glyph-width changes are
separate from this geometric test and still require visual review.

Lightweight registration/export regression tests require no training data:

```bash
MPLCONFIGDIR=/tmp/spatial_axes_assembly_mpl python -B -m unittest Figure_2.assembly.test_layout_registration -v
```
# Classifier label alignment

The two lines `Separate Peyer` and `classifier` share the center of their
original box at x = 517.6989 pt. Their original 8-point Arial and 9.6-point
baseline spacing are retained. `manual_text_replacements` in `layout.json`
controls the two explicit alignment overrides; other manual text is unchanged.
