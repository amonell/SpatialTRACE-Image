# DAPI/E-cadherin backprojection revision (6 September 2026)

## Current fields (16 October 2026)

The author finalized review option 32 for RARi and retained the original DMSO field.
Both main D and extended E now use `Figure_4/panel_D/processing/active_fields.json`
through `Figure_4/field_selection.py`. Its approved manifest is
SATA `Figure_4/stored_outputs/panel_D/approved_fields_option32_20261016/manifest.json`,
SHA256 `10c30344d79527ab09082734fe69f44af9a63864e1328422d6f9b910ba0868dc`.
DMSO bounds are `x=45000:51000, y=35000:39500`; RARi bounds are
`x=31000:37000, y=47000:51500`. Every view has the same 4:3 native aspect;
the former extended-only vertical margin is removed. The selected fields
contain 3,776 DMSO and 2,603 RARi cells, with unchanged values and gate calls.
Marker sizes, DAPI/Cy5 display levels, gate colors and calibrated 200-µm bars
are preserved. No P14 legend entry or outlines are added.

Regenerate with `Figure_4_Extended/run_all_panels.py --panels E --output-root
/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended`.
For the full page, use `update_final_figure.py Figure_4_Extended --overwrite`.
The standalone script receives `--field-pointer` pointing to the same home
JSON. Output stems are unchanged: `panel_E_gate_backprojection.pdf/png` and
its provenance; final-page layers are `panel_E_final.pdf/png`.
See `Figure_4/panel_D/processing/README.md` for exact preparation commands,
required table columns and source hashes. Selection followed author review
of microscopy and predictions; the panels are qualitative illustrations,
not independent compartment-accuracy measurements.

## Preserved earlier field lineage

The historical descriptions and commands below retain the old geometry;
they no longer control the active D panels.

On 14 October the author restored these exact original fields in the complete
Extended Figure 4. The nine-mouse cohort update affects the quantitative IMAPs,
not this representative anatomical display. The canonical renderer uses the
same frozen manifest (SHA256
`f3a67a9481d9969126e2df03d2e7e598158bcd286e92295d717779c7c8817d5b`),
original aspect ratio, contrast and filled markers. New-section backprojections
remain available as companion panels. No field is selected using prediction
quality or treatment-test results.

Completed: the user confirmed Cy5 as E-cadherin. The channel key is frozen at
SATA `frozen_preprocessing/model_design_comparisons_v1/if_channel_key.json`,
SHA256 `4577078f3efa1432ab66866206a88e04f8208478f052e414b354de69a5648ec6`.
The generated component shows DAPI in blue and E-cadherin in gray, retaining
epithelial staining beneath gate-colored cell centroids. The current styling
uses larger filled P14 markers without outlines; all colored points are 50%
larger in diameter than before. Cell memberships and frozen values are unchanged.
The DMSO field contains 5,244 cells (337 P14); RARi contains 4,885 (87 P14).
The paired original and overlaid images use identical display limits. The QA
record beside the prepared manifest verifies every cell field against its
source table, the native DAPI registration and the physical scale calibration.

The confirmed DAPI/E-cadherin component is generated panel D. Its retained
preparer at `panel_G/processing/prepare_gate_backprojection.py` uses the repaired IF
release's saved `coordinate_gate` values and `class == Transferred` P14 calls.
The five gate assignments and their boundaries remain unchanged.

Native registered microscopy comes from SATA
`frozen_preprocessing/model_design_comparisons_v1/if_anatomy_channels_v1/manifest.json`.
Both original Figure 4D windows receive the same 1,000-pixel vertical margin to
include underlying tissue. Their original DAPI cores match byte-for-byte.
The microscopy manifest hashes each native channel array and the source VSI/ETS.
Display limits are shared between conditions and between original/overlay views.

Preparation command (requires the confirmed channel-key JSON):

```bash
Figure2/.venv/bin/python Figure_4_Extended/panel_G/processing/prepare_gate_backprojection.py \
  --release-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/scientific_repair_v1/evaluation/release_manifest.json \
  --channel-manifest /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/model_design_comparisons_v1/if_anatomy_channels_v1/manifest.json \
  --channel-key /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/frozen_preprocessing/model_design_comparisons_v1/if_channel_key.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1
```

Render via `Figure_4_Extended/run_all_panels.py --panels D --output-root
/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended`.
Outputs are `panel_E_gate_backprojection.pdf`, `.png` and
`panel_E_gate_backprojection_provenance.json`. Cells and gate assignments remain
vector; raw microscopy is raster. P14 cells use larger gate-colored filled
points without outlines. "Muscularis" is the display label for the
unchanged saved Muscularis gate. This is qualitative anatomical correspondence,
not an independent accuracy estimate. The generated complete page includes this
as D; E/F now contain non-P14 IMAPs and Peyer windows. Frozen preprocessing
paths retain their historical panel_F names. Active outputs are under SATA
`Figure_4_Extended/panel_E/`; the renderer is
`panel_E/figure_script/make_panel_E_gate_backprojection.py`.
