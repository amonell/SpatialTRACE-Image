# Current shared fields (16 October 2026)

Main Figure 4D and Extended Figure 4E use the author's original DMSO field
(`x=45000:51000, y=35000:39500`) and review option 32 for RARi
(`x=31000:37000, y=47000:51500`). These are identical 6000 × 4500-pixel
fields in both figures. Native centroids use inclusive lower and exclusive
upper bounds; all 3,776 DMSO and 2,603 RARi cells in the fields are displayed.

Both renderers resolve `active_fields.json` through `Figure_4/field_selection.py`.
It locks SATA `Figure_4/stored_outputs/panel_D/approved_fields_option32_20261016/manifest.json`,
SHA256 `10c30344d79527ab09082734fe69f44af9a63864e1328422d6f9b910ba0868dc`.
The manifest hashes native DAPI/Cy5 arrays, cell tables, unchanged prediction
sources, checkpoint, gates and the author's selection. Required fields are
`centroid_x_fullres_px`, `centroid_y_fullres_px`, `x_um`, `y_um`,
`predicted_axis_coordinate`, `predicted_epithelial_distance_clipped_1p0`,
`coordinate_gate`, `class`, `source_id`, and `object_id`.

These are author-selected qualitative examples following prediction and
microscopy review, not an independent estimate of anatomical accuracy. No
prediction, gate boundary, cohort summary or evaluation metric changes.
The original DMSO field is byte-exact to its earlier microscopy crop.

Regenerate the current components with:

```bash
python -B Figure_4/run_all_panels.py --panels D \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4
```

Use `Figure_4/reexport_panels.py --panels D --overwrite` with the same
`--output-root` to archive existing component exports first. Complete pages
use `update_final_figure.py Figure_4 --overwrite` and the corresponding
`Figure_4_Extended` command. Preparation is separate:

```bash
python -B Figure_4/panel_D/processing/prepare_panel_D_rari_selection.py \
  --selection Figure_4/panel_D/processing/approved_selection_option32_20261016.json \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4/stored_outputs/panel_D/approved_fields_option32_20261016
```

Preparation refuses an existing destination. It subsets frozen arrays and
tables only and checks exact agreement with reviewed option 32. The previous
R1 release remains intact. Current component filenames remain the four
`panel_D_{dmso,rari}_{crypt_villus,epithelial_distance}_whole_and_zoom.pdf/png`
pairs, `panel_D_zoom_regions.tsv`, and `panel_D_provenance.json`.

## Historical geometry, preserved as source lineage

The following describes the superseded region lock, not the active RARi view.

Panel D reads all 300,429 frozen production-v2 whole-tissue IF predictions and
uses matched physical fields and fixed color limits.

The required `--zoom-config` is the original production-v2 configuration:
`Figure4/imap_if/config_all_cells_production_v2_if_sft.yaml`, SHA256
`6447d3cc3ea0e4318be13db878fd0a68da0ad4981562721f33a7e885587dce3d`.
Its zoom geometry is independently recorded in the corresponding saved
`imap_if/outputs/all_cells_production_v2_if_sft/imap_if_run_metadata.json`.

The locked regions (level-0 pixel coordinates) are DMSO
`x=45000:51000, y=35000:39500` and RARi
`x=17000:23000, y=48000:52500`. Both are converted using the saved
`0.16247698804244627` micrometres per pixel; each covers
974.861928 × 731.146446 micrometres. No new region selection or inference is
performed. Centroids on the region boundary are included. Only zoom geometry
and pixel size are read from the legacy configuration: its `plot_include_classes`
filter is deliberately not applied to the current all-cell prediction tables.

The renderer creates four individual whole-tissue/zoom components in displayed
order (DMSO crypt-villus, DMSO epithelial distance, RARi crypt-villus, RARi
epithelial distance), a zoom-count/bounds TSV, and provenance. The redundant
combined preview is archived and is no longer generated in the final tree.
Full and zoom maps use `viridis` on 0–1 for crypt-villus and `coolwarm` on 0–0.5
for epithelial distance, identically across conditions. The plot contains all
163,992 DMSO and 136,437 RARi cells; the zoom is an additional view, not a change
to the full-tissue cell set. Rectangles and connectors map the original region
to the corresponding enlargement.
