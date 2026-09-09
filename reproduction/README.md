# Reproduce the figures

Render all eight figures from saved predictions, metrics, images, and schematic artwork.

## Setup

Get `paper_bundle_v3` from the authors (15.3 GB). Install Poppler (`poppler-utils` on Debian or Ubuntu) and Arial fonts.

Then install the Python packages:

```bash
uv sync --locked --extra cpu --extra figures
```

If Arial is installed outside the standard Linux font directory, add `--arial-dir /path/to/fonts` to the commands below. That directory must contain `Arial.ttf`, `Arial_Bold.ttf`, `Arial_Italic.ttf`, and `Arial_Bold_Italic.ttf`.

## Render

Check the inputs, then render:

```bash
uv run --locked --extra cpu --extra figures python reproduction/render.py \
  --bundle /path/to/paper_bundle_v3 --output-dir runs/paper --verify-only

uv run --locked --extra cpu --extra figures python reproduction/render.py \
  --bundle /path/to/paper_bundle_v3 --output-dir runs/paper
```

Use a new output directory for each run. Input checksums must match the supplied bundle.

Both repositories include the same figure code. To render selected figures, add `--figures` followed by their names:

- Overview: `Figure_1`
- Graph: `Figure_2 Figure_2_Extended Figure_2_Extended_2`
- Image: `Figure_3 Figure_3_Extended Figure_4 Figure_4_Extended`

## Outputs

Each figure folder contains:

- The full-page PDF and PNG.
- `components/`: individual plots and images.
- `components/final_panels/panel_X_final.pdf`: one editable layer per panel.
- `reproduction.json` and QA reports.

Panel layers use the full-page canvas. Import them at the origin without rescaling.

## Update a panel

Positions and labels are set in `paper_code/Figure_N/assembly/layout.json`. Plotting code is in the same figure package; shared helpers are in `paper_code/figure_assembly/`.

After editing code or layout, add `--allow-code-changes` and render into a new folder. Input checksums are still checked. Review the PDF for spacing, labels, and image quality.

Rendering uses saved scientific results. New predictions or analyses require updated preprocessing outputs. The bundle contains Graph checkpoint IDs for provenance, with the weight files excluded.
