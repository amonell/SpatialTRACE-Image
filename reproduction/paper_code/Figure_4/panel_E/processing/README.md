# E: whole-section P14 abundance

No analysis runs during rendering. The approved analysis is frozen under SATA
`frozen_preprocessing/if_replicates_20261014/user_rari_positioning_v1/`.
Its manifest SHA256 is `3fff265a5e93948569c14136b8c15aa49281e30bae178455e9609c89d09d174d`;
`Figure_4/replicate_revision_20261014/approved_ab_release.json` pins its path.
The manifest hashes the uploaded analysis, per-mouse tables and test results.

`per_mouse_metrics.csv` supplies `mouse_id`, `condition`, `experiment`,
`n_cells`, `n_p14`, `p14_per10k`. `statistics.csv` supplies the
`log2_p14_per10k` row and its experiment-adjusted `p_ols_batch`.
There are four DMSO and five RARi mice after all supplied removals.

Run `python update_final_figure.py Figure_4 --overwrite` for the complete page
and SATA `Figure_4/panel_E/panel_E_final.pdf` plus PNG/provenance. The standalone
runner exports `panel_E_P14_abundance.pdf` and PNG from the same numeric inputs.
Abundance is relative to non-P14 cell count, not physical tissue area.
