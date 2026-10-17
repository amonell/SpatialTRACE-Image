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
python -B -c "from pathlib import Path; from Figure_2_Extended.assembly.prepare import prepare; prepare(Path('/tmp/Figure_2_Extended_assembly_check'))"
```

The default invokes the existing active panel scripts through
`Figure_2/assembly/fixed_render.py`. Their scientific-input and checkpoint checks
remain active. The adapter changes only canvas size, axes positions, Arial
typography and declared presentation options; it disables automatic/tight
layout. Every PDF has its `.geometry.json` beside it. Existing renderer
provenance records retain data paths/hashes and evaluation scope. No training,
checkpoint selection, inference-corpus creation or preprocessing is invoked.

The main Figure 2 additionally uses a reviewed original-path allowlist for its
manual schematic and inset connectors. The extracted asset contains neither
original text nor images/private Illustrator source. Its labels are retyped in
Arial, and its data examples are fresh plots. The other three packages do not
need an original-artwork background or static vector asset.

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
scaling is reflected in the registered data viewport; all frozen rows remain.
The original low-resolution raster appearance and Arial glyph-width changes are
separate from this geometric test and still require visual review.

Lightweight registration/export regression tests require no training data:

```bash
MPLCONFIGDIR=/tmp/spatial_axes_assembly_mpl python -B -m unittest Figure_2.assembly.test_layout_registration -v
```
