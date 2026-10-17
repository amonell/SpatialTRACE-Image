Panel F uses the representation-pretrained, shared-scale Peyer image classifier
with the axis backbone. SATA `frozen_preprocessing/peyer_representation_v1/image_windows/manifest.json`
locks the checkpoint, new probabilities and unchanged 2,518-cell DMSO /
2,263-cell RARi windows. These are qualitative IF outputs; no independent IF
Peyer labels exist. New probabilities cover only these windows, not a whole-tissue
classifier corpus. `Figure_4_Extended/panel_F/processing/probability_display.json` sets a shared linear 0–1 display
range, covering the new release's full probability range. Saved probabilities are not
rescaled. Current ranges and saturation counts are recorded in final provenance
and `panel_F/processing/probability_range_audit.json`. Both renderers use
`Figure_4_Extended/probability_display.py`; neither rescales predictions.
Prediction points and colorbars remain vector; the native DAPI crop is the only
raster PDF layer, with a 600-dpi companion PNG. No fitting or inference occurs
during rendering, and no display window was reselected from predictions.

The retained gate-backprojection preparation/audit code in this directory
describes the historical frozen source lineage for current panel D. It is not
invoked by final figure rendering. See `panel_E/processing/README.md` from
the figure root for the active gate-review component.
