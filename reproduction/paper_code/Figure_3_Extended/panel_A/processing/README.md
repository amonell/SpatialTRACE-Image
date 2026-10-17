Panel A reads frozen actual paired-dataset views for the exact Cont1 center in
Figure 3a. `frozen_preprocessing/scientific_repair_v1/prepare_paired_views.py`
prepared those arrays before plotting. The authoritative SATA source is
`Figure_3_Extended/stored_outputs/paired_view_repair_v1/manifest.json` and its
hashed `paired_views.npz`. Both student scales use the same selected geometric
and intensity transform; teacher views are clean. Opaque gray (`#808080`) blocks
illustrate the approximately 40% learned-token masking. This is a display
overlay; the frozen images, patch memberships and training inputs are unchanged.

The current hash-locked PDF orders the four views in one horizontal row:
student local, teacher local, student context, teacher context. The renderer
uses that 1×4 layout. The source center and mask fraction are unchanged; the
inconsistent illustrative per-scale geometry has been corrected. There is no
model inference or fitting in rendering. Exact commands and fields are in the
figure README and package runner.
