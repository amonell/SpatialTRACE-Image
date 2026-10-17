"""Nine-mouse Figure 4 panels; read only the explicitly frozen release."""
from __future__ import annotations
import json
from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from Figure_4.assembly.common import PAGE, colorbar, top_text
from Figure_4.assembly import design
from Figure_4.imap_labels import draw_registered_gate_labels
from Figure_4.spatial import GATE_COLORS, GATE_ORDER, gate_background

RELEASE_POINTER = Path(__file__).with_name("active_release.json")
CONDITION_COLORS = {"DMSO": "#4F7F8A", "RARi": "#A55D54"}


def load_inputs(page, release_pointer=RELEASE_POINTER):
    pointer = json.loads(page.read(release_pointer, role="nine_mouse_release_pointer").read_text())
    rec = pointer["manifest"]
    release = json.loads(page.read(rec["path"], rec["sha256"], "frozen_nine_mouse_release").read_text())
    if release["status"] != "frozen_nine_mouse_figure_release" or release["mice_per_condition"] != {"DMSO": 4, "RARi": 5}:
        raise RuntimeError("Unexpected replicate release")
    files = {Path(r["path"]).name: r for r in release["outputs"]}
    def read(name):
        rec = files[name]
        return page.read(rec["path"], rec["sha256"], "replicate_" + name)
    data = {"release": release, "files": files, "read": read,
            "sections": pd.read_csv(read("sections.csv")), "gates": pd.read_csv(read("gates.csv")),
            "stats": pd.read_csv(read("group_summary.csv")), "density": np.load(read("densities.npz")),
            "density_display": json.loads(read("density_display.json").read_text()),
            "display_fields": json.loads(read("display_fields.json").read_text())}
    data["stats"] = data["stats"].loc[data["stats"].subset == "all_retained_mice"]
    rec = pointer["statistics_manifest"]
    analysis = json.loads(page.read(rec["path"], rec["sha256"], "frozen_mouse_statistics").read_text())
    if analysis["status"] != "frozen_density_over_expected_statistics":
        raise RuntimeError("Expected frozen observed/expected mouse-level statistics")
    analysis_files = {Path(r["path"]).name: r for r in analysis["outputs"]}
    def read_analysis(name):
        rec = analysis_files[name]
        return page.read(rec["path"], rec["sha256"], "regional_analysis_" + name)
    data["tests"] = pd.read_csv(read_analysis("density_over_expected_statistics.csv"))
    data["analysis_plan"] = json.loads(read_analysis("analysis_plan.json").read_text())
    if data["analysis_plan"]["input_manifest_sha256"] != pointer["manifest"]["sha256"]:
        raise RuntimeError("Statistics and cohort do not match")
    rec = pointer["display_manifest"]
    display = json.loads(page.read(rec["path"], rec["sha256"], "frozen_post_removal_display").read_text())
    if display["status"] != "frozen_post_removal_figure_display":
        raise RuntimeError("Expected verified post-removal display limits")
    rec = next(r for r in display["outputs"] if Path(r["path"]).name == "density_display.json")
    data["P14_density_display"] = config = json.loads(
        page.read(rec["path"], rec["sha256"], "post_removal_density_display").read_text())
    if (config["cohort_manifest_sha256"] != pointer["manifest"]["sha256"] or
            config["statistics_manifest_sha256"] != pointer["statistics_manifest"]["sha256"]):
        raise RuntimeError("Display release does not match the cohort/statistics")
    if config["non_P14_limits"] != data["density_display"]["shared_limits"] or config["clipped_values"]:
        raise RuntimeError("Inconsistent density display")
    primary = data["tests"].loc[data["tests"].analysis.eq("all_mice")]
    if (len(primary) != 5 or set(primary.gate) != set(GATE_ORDER) or
            not primary.metric.eq("observed_over_expected_p14_density").all() or
            not primary.n_dmso.eq(4).all() or not primary.n_rari.eq(5).all()):
        raise RuntimeError("Unexpected primary endpoint, gates or mouse counts")
    for condition in ("DMSO", "RARi"):
        summary = data["stats"].loc[data["stats"].condition.eq(condition) &
                                  data["stats"].metric.eq("p14_regional_enrichment")].set_index("gate")
        for field in ("mean", "sem"):
            np.testing.assert_allclose(primary.set_index("gate").loc[list(GATE_ORDER), condition.lower()+"_"+field],
                                       summary.loc[list(GATE_ORDER), field])
    if "nancy_if_20261014_08" in set(data["sections"].source_id):
        raise RuntimeError("Excluded tissue08 entered the figure")
    page.details["replicate_cohort"] = {"n_mice": 9, "mice_per_condition": release["mice_per_condition"],
        "n_cells": release["n_cells"], "n_p14": release["n_p14"], "excluded": release["excluded"],
        "aggregation": "Equal mouse means; individual mice and original/new identity retained in frozen tables",
        "cohort_manifest": pointer["manifest"], "statistics_manifest": pointer["statistics_manifest"],
        "display_manifest": pointer["display_manifest"], "manual_removals_applied": True}
    page.warnings.extend(release["warnings"])
    page.warnings.extend(analysis["warnings"])
    page.read(Path(__file__), role="nine_mouse_panel_renderer")
    return data


def regional_summary(page, data):
    """Original dumbbell/lollipop design; all estimates and tests are frozen."""
    figure = page.figure()
    left = page.axis(figure, "E_abundance", [49.495, 669.367, 148.248, 132.912])
    right = page.axis(figure, "E_composition", [206.117, 669.366, 148.248, 132.912])
    primary = data["tests"].loc[data["tests"].analysis.eq("all_mice")]
    enrichment = primary.loc[primary.metric.eq("observed_over_expected_p14_density")].set_index("gate")
    abundance = data["stats"].loc[data["stats"].metric.eq("p14_per_10000_non_p14")]
    if len(abundance) != 10 or abundance.duplicated(["gate", "condition"]).any():
        raise RuntimeError("Expected one frozen abundance summary per gate and condition")
    abundance = abundance.set_index(["gate", "condition"])
    upper = float((abundance["mean"] + abundance["sem"]).max())
    left.set_xlim(0, upper*1.18)
    limit = max(.25, float((enrichment.difference_dmso_minus_rari.abs()+enrichment.difference_sem).max())*1.17)
    right.axvspan(-limit, 0, color="#f6eeee", zorder=0)
    right.axvspan(0, limit, color="#eef4f5", zorder=0)
    right.axvline(0, color="#555555", lw=.6, zorder=1)
    right.set_xlim(-limit, limit)
    for y, gate in enumerate(GATE_ORDER):
        means = [abundance.loc[(gate, condition), "mean"] for condition in ("RARi", "DMSO")]
        left.plot(means, [y, y], color=GATE_COLORS[gate], lw=1.8, zorder=2)
        for condition in ("DMSO", "RARi"):
            row = abundance.loc[(gate, condition)]
            left.errorbar(row["mean"], y, xerr=row["sem"], fmt="o", ms=3.6,
                          color=CONDITION_COLORS[condition], markeredgecolor="#222222", markeredgewidth=.3,
                          elinewidth=.7, capsize=2, capthick=.6, zorder=4)
        row = enrichment.loc[gate]
        value, sem = row.difference_dmso_minus_rari, row.difference_sem
        right.plot([0, value], [y, y], color=GATE_COLORS[gate], lw=1.8, zorder=2)
        right.errorbar(value, y, xerr=sem, fmt="o", ms=3.6,
                       color=CONDITION_COLORS["DMSO" if value >= 0 else "RARi"],
                       markeredgecolor="#222222", markeredgewidth=.3, elinewidth=.7, capsize=2, capthick=.6, zorder=4)
        # One adjusted P value per gate, in the difference panel only.
        right.text(.98, y+.28, f"P = {row.holm_adjusted_p:.3f}",
                   transform=right.get_yaxis_transform(), ha="right", va="center", fontsize=5.8, color="#444444")
    for axis in (left, right):
        axis.set(ylim=(4.52, -.40), yticks=range(5), yticklabels=list(GATE_ORDER) if axis is left else [])
        axis.tick_params(labelsize=6.7)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#e5e5e5", lw=.45, zorder=0)
        axis.set_axisbelow(True)
    left.set_xlabel("P14 per 10,000 non-P14 cells\nin the same gate", fontsize=6.7, labelpad=3)
    right.set_xlabel("DMSO − RARi\n(observed / expected)", fontsize=6.7, labelpad=3)
    for x, title in ((123.619, "Gate-specific P14 abundance"), (280.241, "Enrichment difference")):
        top_text(figure, x, 664.57, title, ha="center", fontsize=7.6, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", color=CONDITION_COLORS[c], markersize=3,
                      label=f"{c} (n = {data['release']['mice_per_condition'][c]})") for c in ("DMSO", "RARi")]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(495 / PAGE[0], 1 - 778 / PAGE[1]),
                  frameon=False, fontsize=6.5, ncol=2, columnspacing=.8, handletextpad=.4, handlelength=1.2)
    for y, label in ((795, "Mouse means and differences ± SEM"),
                     (806, "P: exact permutation; Holm across 5 gates"),
                     (817, "Expected: background-cell distribution")):
        top_text(figure, 495, y, label, ha="center", fontsize=5.9)
    page.details["replicate_regional_summary"] = {"n_mice": 9,
        "left_endpoint": "p14_per_10000_non_p14",
        "left_normalization": "Within each mouse and gate: 10,000 times P14 count divided by non-P14 count",
        "left_displayed_summary": abundance.reset_index().to_dict(orient="records"),
        "left_inference": "Descriptive mouse means and SEM; observed/expected P values apply only to the right subplot",
        "right_endpoint": data["analysis_plan"]["endpoint"],
        "endpoint": data["analysis_plan"]["endpoint"],
        "normalization": data["analysis_plan"]["observed_over_expected"],
        "uncertainty": data["analysis_plan"]["summary"],
        "difference": "Difference of mouse-mean observed/expected ratios, DMSO minus RARi",
        "test": data["analysis_plan"]["primary_test"], "multiplicity": data["analysis_plan"]["multiplicity"],
        "two_sided_definition": data["analysis_plan"]["two_sided_definition"],
        "displayed_P": "Holm-adjusted, once per gate in the difference plot",
        "displayed_statistics": enrichment.loc[list(GATE_ORDER)].reset_index().to_dict(orient="records"),
        "legacy_layout_aliases": {"E_abundance": "P14 per 10,000 non-P14 cells in the same gate", "E_composition": "Observed/expected difference"},
        "statistical_scope": "Exploratory; cohort-stratified and new-only sensitivity tests saved alongside primary results",
        "connections": "Treatment group means; mice are not paired"}
    page.save(figure, "panel_E_abundance_composition")


def mean_imap(page, figure, data, condition, population, rect, *, title_y, show_xticks=True,
              title_x=None, xlabel_anchor=None, show_sample_size=True,
              show_ylabel=True, show_yticklabels=True):
    p14 = population == "P14"
    prefix = "F" if p14 else "D"  # Extended-page adapter translates its legacy D source anchors to E.
    suffix = "p14" if p14 else "non_p14"
    grid = data["density"][f"{condition}__{suffix}__mean_mouse"]
    config = data["density_display"]
    limits = data["P14_density_display"]["P14_shared_limits"] if p14 else config["shared_limits"]
    definition_record = data["release"]["gate_definition"]
    definitions = json.loads(page.read(definition_record["path"], definition_record["sha256"], "unchanged_paper_gates").read_text())
    ax = page.axis(figure, prefix + "_" + condition, rect)
    gate_background(ax, definitions)
    centers = data["density"]["centers"]
    masked = np.ma.masked_less(grid.T, config["lower_display_threshold"])
    norm = Normalize(*limits)
    if p14:
        ax.contourf(centers, centers, masked, levels=np.linspace(config["lower_display_threshold"], limits[1], 35),
                    cmap="viridis", norm=norm, alpha=.86, zorder=2)
    else:
        ax.pcolormesh(data["density"]["edges"], data["density"]["edges"], masked,
                      cmap="viridis", norm=norm, shading="flat", rasterized=True, alpha=.86, zorder=2)
    levels = [v for v in config["contour_levels"] if grid.min() < v < grid.max()]
    if levels:
        ax.contour(centers, centers, grid.T, levels=levels, colors="#666666", linewidths=.45, alpha=.8, zorder=4)
    ax.axvline(.28, color="#333333", lw=.55, zorder=5)
    ax.axhline(.4, color="#333333", lw=.55, zorder=5)
    ax.plot([.28, 1], [.25, .25], color="#333333", lw=.55, zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
    ax.tick_params(labelbottom=show_xticks, labelleft=show_yticklabels)
    if show_ylabel:
        ax.set_ylabel("Crypt-villus coordinate", labelpad=3)
    if xlabel_anchor is not None:
        top_text(figure, *xlabel_anchor, "Epithelial-distance coordinate", fontsize=7.5 if p14 else 8,
                 ha="center" if p14 else "left")
    count = data["release"]["mice_per_condition"][condition]
    title = f"{condition} {population}"
    if show_sample_size:
        title += f" · n = {count} mice"
    top_text(figure, rect[0] + rect[2] / 2 if title_x is None else title_x, title_y,
             title, ha="center", fontsize=8, fontweight="bold")
    metric = "p14_composition_percent" if p14 else "non_p14_composition_percent"
    table = data["stats"].loc[(data["stats"].condition == condition) & (data["stats"].metric == metric)].set_index("gate")
    fractions = table["mean"].to_dict()
    np.testing.assert_allclose(sum(fractions.values()), 100)
    registration = draw_registered_gate_labels(ax, fractions, population, condition, GATE_COLORS,
                                               font_size_pt=design.IMAP_LABEL_SIZE if p14 else None)
    page.read(registration["source"]["path"], registration["source"]["sha256"], "original_IMAP_label_artwork")
    for box in registration["boxes"]:
        page.anchors.append({"id": prefix + "_" + condition + "_label_" + box["gate"].replace(" ", "_"),
                             "rect_pt": box["rect_pt"]})
    page.details[f"replicate_{condition}_{suffix}_imap"] = {"mice": count, "mean_mouse_gate_percentages": fractions,
        "density_aggregation": "Equal weight per mouse, not pooled cell counts", "density_limits": limits,
        "gate_labels": registration, "individual_cells_available_in_companion_IMAPs": True}
    return plt.cm.ScalarMappable(norm=norm, cmap="viridis")


def p14_imaps(page, data):
    figure = page.figure()
    mean_imap(page, figure, data, "DMSO", "P14", design.IMAP_RECTS[0],
              title_y=design.IMAP_TITLE_Y, show_sample_size=False)
    artist = mean_imap(page, figure, data, "RARi", "P14", design.IMAP_RECTS[1],
                       title_y=design.IMAP_TITLE_Y, xlabel_anchor=design.IMAP_XLABEL,
                       show_sample_size=False, show_ylabel=False, show_yticklabels=False)
    cax = page.axis(figure, "F_shared_density_bar", design.IMAP_BAR)
    colorbar(figure, cax, artist, horizontal=True, label="Probability density",
             ticks=data["P14_density_display"]["P14_ticks"])
    page.read(Path(design.__file__), role="author_approved_display_geometry")
    page.save(figure, "panel_F_P14_IMAPs")


def controls(page, data, *, panel='E'):
    figure = page.figure()
    for condition, rect, title_y in [
        ("DMSO", [44.887, 322.236, 173.601, 111.996], 303.017),
        ("RARi", [287.556, 322.236, 173.601, 111.996], 303.1273)]:
        artist = mean_imap(page, figure, data, condition, "non-P14", rect, title_y=title_y,
                          title_x=126.07029 if condition == "DMSO" else 374.92162,
                          xlabel_anchor=(80.6858 if condition == "DMSO" else 331.6822, 457.705),
                          show_sample_size=False)
    cax = page.axis(figure, 'D_shared_density_bar', [465.85, 328.96, 4.92, 98.52])
    colorbar(figure, cax, artist, label="Probability density", ticks=[0, 4, 8, 12, 16])
    page.details['non_P14_colorbar_display'] = {'shared_between_conditions': True,
        'limits': data['density_display']['shared_limits'], 'ticks': [0, 4, 8, 12, 16],
        'sample_sizes_and_normalization_in_caption': True}
    page.save(figure, f"panel_{panel}_non_P14_IMAPs")


def composite(dapi, ecad, levels):
    d = np.clip(np.asarray(dapi, dtype=np.float32) / levels["dapi_upper"], 0, 1)
    e = np.clip(np.asarray(ecad, dtype=np.float32) / levels["ecad_upper"], 0, 1)
    return np.clip(np.stack([.65*d, .65*d+.8*e, .9*d+.8*e], axis=-1), 0, 1)


def backprojection(page, data, layout):
    figure = page.figure()
    settings = layout["gate_projection_layout"]
    for i, condition in enumerate(("DMSO", "RARi")):
        sid = data["release"]["representative_new_sections"][condition]
        field = data["display_fields"][sid]
        arrays = np.load(data["read"](f"{sid}_microscopy.npz"))
        image = composite(arrays["dapi_crop"], arrays["ecad_crop"], field["levels"])
        frame = pd.read_parquet(data["read"](f"{sid}.parquet"))
        x, y, width, height = field["bounds_xywh_px"]
        cells = frame.loc[(frame.centroid_x_fullres_px >= x) & (frame.centroid_x_fullres_px < x+width) &
                          (frame.centroid_y_fullres_px >= y) & (frame.centroid_y_fullres_px < y+height)]
        for j in range(2):
            ax = page.axis(figure, f"D_{condition}_{'image' if j == 0 else 'gates'}", settings["axes_rect_pt"][2*i+j])
            ax.imshow(image, extent=(x, x+width, y+height, y), interpolation="nearest")
            if j:
                ax.scatter(cells.centroid_x_fullres_px, cells.centroid_y_fullres_px, c=cells.gate_name.map(GATE_COLORS),
                           s=np.where(cells.is_p14 == 1, settings["P14_marker_area_pt2"], settings["cell_marker_area_pt2"]),
                           linewidths=0, edgecolors="none", zorder=2)
            ax.set(xlim=(x, x+width), ylim=(y+height, y), aspect="equal")
            ax.axis("off")
            ax.set_title(f"{condition} {sid[-2:]} · " + ("DAPI / E-cadherin" if j == 0 else "Cells by IMAP gate"),
                         fontsize=settings["title_size_pt"], pad=6)
            bar_px = 200 / field["pixel_size_x_um"]
            start = x + .04*width
            ax.plot([start, start+bar_px], [-.035, -.035], transform=ax.get_xaxis_transform(),
                    color="black", lw=1.4, clip_on=False)
            ax.text(start+bar_px/2, -.05, "200 µm", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=settings["scale_label_size_pt"])
        page.details[f"new_backprojection_{condition}"] = {"source_id": sid, "cells": len(cells),
            "bounds_xywh_px": field["bounds_xywh_px"], "gate_assignments_changed": False,
            "native_images": True, "P14": "Larger filled gate-colored markers; no outlines", "E_cadherin_channel": "Cy5"}
    handles = [Line2D([], [], marker="o", ls="", color=GATE_COLORS[g], label=g, markersize=4) for g in GATE_ORDER]
    x, y = settings["legend_anchor_pt"]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(x/PAGE[0], 1-y/PAGE[1]),
                  ncol=5, frameon=False, fontsize=settings["legend_size_pt"], columnspacing=1.2, handletextpad=.35)
    page.save(figure, "panel_D_gate_backprojection")
