#!/usr/bin/env python3
"""Render the frozen exploratory panel E comparisons; never calculate tests here."""
from __future__ import annotations
import argparse
import json
from paper_paths import Path, lock_record, rendering_script
import sys

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Figure_4.common import configure_figure_fonts, require_file, file_record, write_provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if manifest["status"] != "frozen_exploratory_mouse_statistics":
        raise RuntimeError("Statistics must be frozen before rendering")
    files = {Path(r["path"]).name: require_file(r["path"], r["sha256"]) for r in manifest["outputs"]}
    table = pd.read_csv(files["regional_statistics.csv"])
    plan = json.loads(files["analysis_plan.json"].read_text())
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    configure_figure_fonts()
    fig = plt.figure(figsize=(10.5, 7.4))
    fig.text(.05, .95, "Figure 4E · Mouse-level treatment comparisons", fontsize=14, weight="bold")
    fig.text(.05, .913, "DMSO: 4 mice  |  RARi: 5 mice  ·  One section per mouse; tissue08 excluded", fontsize=9)
    for y, metric, title, units in [
        (.54, "p14_per_10000_non_p14", "Gate-specific P14 abundance", "P14 per 10,000 non-P14 cells in the same gate"),
        (.20, "p14_composition_percent", "Within-P14 composition", "% of each mouse’s P14 cells; differences in percentage points")]:
        ax = fig.add_axes([.05, y, .9, .26]); ax.axis("off")
        ax.text(0, 1.15, title, weight="bold", fontsize=11, transform=ax.transAxes)
        ax.text(0, 1.04, units, fontsize=8, transform=ax.transAxes)
        rows = []
        for gate in plan["gates"]:
            sub = table.loc[table.metric.eq(metric) & table.gate.eq(gate)].set_index("analysis")
            main = sub.loc["all_mice"]
            rows.append([gate, f"{main.dmso_mean:.1f} ± {main.dmso_sem:.1f}",
                         f"{main.rari_mean:.1f} ± {main.rari_sem:.1f}",
                         f"{main.mean_difference_dmso_minus_rari:+.1f} ± {main.difference_sem:.1f}",
                         f"{main.p_holm_five_gates:.3f}",
                         f"{sub.loc['cohort_stratified'].p_holm_five_gates:.3f}",
                         f"{sub.loc['new_only'].p_holm_five_gates:.3f}"])
        headers = ["Gate", "DMSO\nmean ± SEM", "RARi\nmean ± SEM", "DMSO − RARi\nmean ± SEM",
                   "All mice\nadjusted P", "Cohort-stratified\nadjusted P", "New mice only\nadjusted P"]
        chart = ax.table(cellText=rows, colLabels=headers, bbox=[0, 0, 1, 1],
                         colWidths=[.14, .155, .155, .17, .11, .14, .13], cellLoc="center")
        chart.auto_set_font_size(False); chart.set_fontsize(8)
        for (row, col), cell in chart.get_celld().items():
            cell.set_edgecolor("#dddddd"); cell.set_linewidth(.4)
            if row == 0:
                cell.set_facecolor("#eaf0f2"); cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f6f6f6")
            if col == 0:
                cell.set_text_props(ha="left")
    notes = [
        "Two-sided exact mouse-label permutation tests; Holm correction across five gates separately for each endpoint and analysis.",
        "All mice: 126 allocations. Cohort-stratified: 70 allocations, weighted within-cohort mean difference. New mice: 35 allocations (3 DMSO, 4 RARi).",
        "SEM describes variation across mice, not cells. The plotted difference is unadjusted; the stratified test uses a cohort-adjusted statistic.",
        "Exploratory analysis following descriptive review. Randomization is undocumented; label tests assume exchangeability under the null.",
        "Original mice contributed to model development. Tissue08 was excluded after interim inspection. No new coordinate ground truth is available."
    ]
    for i, note in enumerate(notes):
        fig.text(.05, .145-i*.024, note, fontsize=7.5, va="top")
    outputs = [out/"Additional_07_regional_statistics.pdf", out/"Additional_07_regional_statistics.png"]
    for path in outputs:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    write_provenance(out/"provenance.json", script=Path(__file__),
                     inputs=[file_record(a.manifest, "frozen_statistics_manifest")] +
                            [file_record(path, name) for name, path in files.items()],
                     outputs=outputs, overwritten=[], extra={"analysis_performed_during_render": False})


if __name__ == "__main__":
    main()
