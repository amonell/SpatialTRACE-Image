# F: P14 IMAPs

The unchanged frozen nine-mouse cohort is selected by
`Figure_4/replicate_revision_20261014/active_release.json`. Its manifest is SATA
`frozen_preprocessing/if_replicates_20261014/nine_mouse_removal_release_v1/manifest.json`,
SHA256 `613bc495ad9fb518e0d47d34f1ed72fd860f002502dd04ba626132bb7879ccc0`.
The pointer also locks density display settings. The cohort manifest hashes
`densities.npz`, `group_summary.csv`, `sections.csv` and gate definitions.

Each condition uses the equal-weight mean of individually normalized mouse
grids. Gate percentages are means of within-mouse percentages. Four DMSO and
five RARi mice contribute; tissue08 is excluded. The shared P14 color range is
0–13. Non-P14 maps retain their separate 0–16 range. No inference, smoothing,
gate assignment or statistics are recomputed by the renderer.

Original gate-label boxes and their positions within each map are unchanged.
DMSO and RARi appear side by side with a shared y-axis and colorbar.
Both maps narrow to 152 pt while retaining their height and relative label-box
positions. Gate text uses 5.7 pt and headings are centered. Explanatory text is in the figure caption. The colorbar and its
label remain. PDF contours, gates, colorbar and text are vector; PNG is 600 dpi.

Run `python update_final_figure.py Figure_4 --overwrite` for the complete page
and SATA `Figure_4/panel_F/panel_F_final.pdf` plus PNG/provenance. The standalone
runner exports `panel_F_P14_IMAPs.pdf` and PNG from these frozen numeric inputs.
