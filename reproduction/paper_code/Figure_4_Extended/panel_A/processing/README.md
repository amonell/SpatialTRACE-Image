# Frozen IEL CCR9 validation

`iel_ccr9_release.json` pins the completed owner-approved analysis at
`/mnt/sata1/Analysis_Alex/Desktop_2/PyFlowjoPrism/outputs/2026-10-16_SpatialTRACE_RARi_IEL_CCR9`.
The readout is CCR9+ cells as a percentage of live CD8+ CD45.1+ IEL donors.
DMSO 5–7 and RARi 1–4 are independent biological replicates. DMSO 8 is retained
in the audit table but excluded for owner-confirmed failed cell recovery.

The lock identifies the canonical SpatialTRACE notebook/tab, edited matrix revision,
complete native gate hierarchy, measurement table, exact representative selection,
statistical result and raw FCS hashes. No new experimental assignment is inferred.
The input table requires `sample_id`, `condition`, `mouse`, `included`,
`donor_events`, `ccr9_positive_events`, and `ccr9_percent`. The statistical JSON
supplies `p`; the representative JSON supplies the approved `selected` records.

The renderer checks the two representatives' actual full-event donor/CCR9 gate
counts against the frozen release. It reuses all seven mouse measurements and
the two-sided Welch P = 0.0137. The exact-rank sensitivity P = 0.0571 and the
exploratory endpoint selection remain disclosed in the Methods and source results.
It performs no fitting, gate adjustment, representative selection or new test.

Shared contour settings are 5%, 256 bins, smoothing on and outlier dots. The
Python Gaussian sigma of five bins is an approximation, not FlowJo's native
smoothing code 13. Panel headings are DMSO/RARi; no mouse IDs or `n=` annotations
appear in the panel. The compact manuscript summary uses four-point dots with narrower jitter; this
is an owner-requested exception to the larger standalone summary default.

Output: SATA `Figure_4_Extended/panel_A/panel_A_final.pdf` and PNG/provenance,
plus the complete-page technical run's `panel_A_iel_ccr9.pdf`, PNG, geometry,
measurement CSV and provenance. Regenerate through the package README's public
final-figure command. Code snapshots and the pre-integration source live under
home `lineage_archive/extended4_iel_ccr9_2026-10-16/pre_change`.
