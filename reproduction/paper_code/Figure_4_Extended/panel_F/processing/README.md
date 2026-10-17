# Non-P14 IMAPs: frozen nine-mouse inputs

Both standalone and final renderers use the post-removal cohort selected by
`Figure_4/replicate_revision_20261014/active_release.json`. The manifest is SATA
`frozen_preprocessing/if_replicates_20261014/nine_mouse_removal_release_v1/manifest.json`,
SHA256 `613bc495ad9fb518e0d47d34f1ed72fd860f002502dd04ba626132bb7879ccc0`.
It hashes the cohort tables, gate definitions and density arrays. The pointer
also locks display settings.

Required inputs include `densities.npz` (grid edges, centers, and the two
`{condition}__non_p14__mean_mouse` arrays) and `group_summary.csv`
(`subset`, `condition`, `metric`, `gate`, `mean`).
Four DMSO and five RARi mice contribute equally. The retained non-P14 counts
are 748,234 and 776,519. Tissue08 is excluded. Gate labels are means of
within-mouse percentages; each mouse density was normalized before averaging.
Rendering performs no inference, density estimation, gate assignment or tests.

The 15 October display cleanup keeps both original map rectangles and all
registered gate-label boxes. Sample sizes and normalization are described in
the figure caption. One vertical colorbar in the original right-hand slot
shows the unchanged 0–16 scale, with ticks at 0, 4, 8, 12 and 16.
Dense density layers and PNGs are 600 dpi; text, gates, contours and the
colorbar remain vector. The historical two-mouse component is archived.

For canonical final layers and the complete page, run:

```bash
python -B update_final_figure.py Figure_4_Extended --overwrite
```

Final exports are SATA `Figure_4_Extended/panel_F/panel_F_final.pdf`, PNG
and provenance. To generate a separate import component from the same inputs:

```bash
python -B Figure_4_Extended/run_all_panels.py --panels E \
  --output-root /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_4_Extended
```

The standalone script accepts an explicit `--cohort-pointer` and `--output-dir`.
It exports `panel_F_non_P14_IMAPs.pdf`, a 600-dpi PNG and
`assembly_inputs_provenance.json`. Use the archival wrapper documented in
the figure README before replacing existing component exports. The standalone
PDF uses original pre-translation coordinates; final assembly translates its
intact layer downward by 173 pt.

The retained `probability_display.json` belongs to panel F and preserves its
historical path after relettering. This cleanup does not alter that file.
