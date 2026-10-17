# G: gate enrichment

No analysis runs during rendering. The approved analysis is frozen under SATA
`frozen_preprocessing/if_replicates_20261014/user_rari_positioning_v1/`.
Its manifest SHA256 is `3fff265a5e93948569c14136b8c15aa49281e30bae178455e9609c89d09d174d`;
`Figure_4/replicate_revision_20261014/approved_ab_release.json` pins its path.

`per_mouse_gate_counts.csv` supplies `mouse_id`, `condition`, `experiment`,
`imap_gate` and `log2_enrich`. `statistics.csv` supplies the five `log2_enrich`
rows, `p_ols_batch` and BH-adjusted `q_ols_batch`. The frozen uploaded analysis
uses log2(((gate P14 + 0.5) / total P14) / (gate non-P14 / total non-P14)).
Rendering never changes gates, recomputes tests or pools mice as cells.

Run `python update_final_figure.py Figure_4 --overwrite` for the complete page
and SATA `Figure_4/panel_G/panel_G_final.pdf` plus PNG/provenance. The standalone
runner exports `panel_G_P14_enrichment.pdf` and PNG. Positive enrichment is
relative allocation, not an increase in absolute P14 abundance.
