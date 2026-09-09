# Reproduce the paper figures

This directory contains the rendering code and registered layouts for all eight current paper figures. It redraws plots and microscopy layers from frozen numeric/image inputs and assembles them with the approved schematic vectors. It does not copy finished scientific panels or rerun training, checkpoint selection, inference, UMAP fitting, ablations, or statistical analyses.

## Inputs and installation

The separately distributed `paper_bundle_v3` contains 152 hash-locked inputs (15.3 GB uncompressed). Public hosting and data licensing await author confirmation; no download URL is claimed yet. Only TissueMapper-Image distributes pretrained weights, under GPL-3.0-only. Graph checkpoint binaries are excluded from the figure bundle: their recorded identities are checked as provenance, while frozen prediction tables, metrics, and all other distributed inputs retain full content-hash verification.

```bash
uv sync --locked --extra cpu --extra figures
uv run --locked --extra cpu --extra figures python reproduction/render.py --bundle /path/to/paper_bundle_v3 --output-dir runs/paper --verify-only
uv run --locked --extra cpu --extra figures python reproduction/render.py --bundle /path/to/paper_bundle_v3 --output-dir runs/paper
```

Tested on Linux with Python 3.12. Install Poppler's `pdftoppm`, `pdffonts`, `pdftotext`, and `pdfimages` commands separately (the `poppler-utils` package on Debian/Ubuntu). Exact typography requires licensed Arial fonts. Fonts are not redistributed. If they are not installed at the standard Linux path, append `--arial-dir /path/to/fonts`, containing `Arial.ttf`, `Arial_Bold.ttf`, `Arial_Italic.ttf`, and `Arial_Bold_Italic.ttf`. Font substitution is rejected. Figure rendering uses the CPU and does not need a GPU.

Use a new output directory for every run. Verification fails on altered frozen data. Files in the bundle are read-only during rendering; historical workstation paths inside immutable provenance are resolved against the bundle, and attempted reads from the original workstation are rejected.

## Select figures

Both model repositories carry the same small rendering snapshot so either can rebuild the complete paper. To render only the graph figures:

```bash
uv run --locked --extra cpu --extra figures python reproduction/render.py --bundle /path/to/paper_bundle_v3 --output-dir runs/graph_figures --figures Figure_2 Figure_2_Extended Figure_2_Extended_2
```

For the image figures, select `Figure_3 Figure_3_Extended Figure_4 Figure_4_Extended`. `Figure_1` is the shared overview. Omitting `--figures` renders all eight.

## Outputs and updates

Each figure directory contains its complete PDF and PNG, `reproduction.json`, QA reports, and freshly generated `components/`. `components/final_panels/panel_X_final.pdf` is the registered, editable PDF layer for each panel letter. These layers retain the full page canvas: import at the origin without rescaling. Recomposition is checked against the direct full-page rendering at 150 dpi. Automated checks cover geometry, embedded Arial fonts, required/obsolete text, panel letters, clipping, private PDF content, and raster-resolution requirements.

`paper_code/Figure_N/assembly/layout.json` controls the approved positions and wording. Panel-specific plotting functions live under that package; shared rendering helpers are under `paper_code/figure_assembly/`. `code_manifest.json` identifies the tested source snapshot. For an intentional code/layout edit, pass `--allow-code-changes`; input-data checks remain mandatory. Rebuild into a new folder and review the resulting PDF. Changing scientific results requires a separately verified preprocessing release, not a plotting-time refit.

The paper's held-out-section definitions, development-set exposure, graph-derived image references, jointly fitted scVI features, and single-section-per-condition IF comparison are unchanged. Generic training tutorials test the software and do not reproduce the paper's full model-development experiments. See the model cards and Methods for the scope of those results.
