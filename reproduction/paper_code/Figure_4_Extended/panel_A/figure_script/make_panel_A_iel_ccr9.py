#!/usr/bin/env python3
"""Render the approved IEL CCR9 validation from its frozen flow-analysis release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import platform
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from paper_paths import Path, lock_record, rendering_script

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.font_manager import FontProperties, findfont


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(record):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"Frozen CCR9 input changed: {path}")
    return path


class FrozenRendererFinder(importlib.abc.MetaPathFinder):
    """Load only the three explicitly hash-locked renderer modules from copies."""
    def __init__(self, records):
        self.modules = {'pyflowjo_prism.' + Path(r['path']).stem: Path(r['path'])
                        for r in records}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_file_location(fullname, self.modules[fullname])
        return None


def render(lock_path, layout_path, output_dir, overwrite=False, renderer_source_dir=None):
    lock = json.loads(Path(lock_path).read_text())
    if lock["status"] != "FROZEN_OWNER_APPROVED_IEL_CCR9":
        raise RuntimeError("IEL CCR9 release is not approved")
    paths = {key: checked(record) for key, record in lock["inputs"].items()}
    renderer_sources = []
    for original in lock["renderer_sources"]:
        record = dict(original)
        if renderer_source_dir is not None:
            record['original_path'] = original['path']
            record['path'] = str(Path(renderer_source_dir).resolve() / Path(original['path']).name)
        checked(record)
        renderer_sources.append(record)
    sys.path.insert(0, str(Path(lock["pyflowjo_root"]) / "src"))
    finder = FrozenRendererFinder(renderer_sources)
    if any(name in sys.modules for name in finder.modules):
        raise RuntimeError('Renderer modules were imported before their frozen sources were selected')
    sys.meta_path.insert(0, finder)
    try:
        from pyflowjo_prism import FlowStyle, Project, SummaryStyle, plot_summary, shared_compensation_matrix
        for name, path in finder.modules.items():
            if Path(sys.modules[name].__file__).resolve() != path.resolve():
                raise RuntimeError('A renderer module was not loaded from its verified source')
    finally:
        sys.meta_path.remove(finder)

    layout = json.loads(Path(layout_path).read_text())
    settings = layout["iel_ccr9_layout"]
    width, height = layout["page_size_pt"]
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / f"panel_A_iel_ccr9.{ext}" for ext in ("pdf", "png")]
    outputs += [out / "panel_A_iel_ccr9_measurements.csv", out / "panel_A_iel_ccr9_provenance.json",
                out / "panel_A_iel_ccr9.geometry.json"]
    if any(path.exists() for path in outputs):
        if not overwrite:
            raise FileExistsError("CCR9 component exists; pass --overwrite")
        archive = out.parent / "polished_panel_archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_iel_ccr9")
        archive.mkdir(parents=True)
        for path in outputs:
            if path.exists():
                shutil.copy2(path, archive / path.name)
    font = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "pdf.fonttype": 42,
                         "font.size": 7, "axes.linewidth": .6, "svg.fonttype": "none",
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "mathtext.fallback": None})
    table = pd.read_csv(paths["measurements"])
    table = table.loc[table.included].copy()
    if set(table.loc[table.condition.eq("DMSO"), "mouse"]) != {5, 6, 7} or set(
            table.loc[table.condition.eq("RARi"), "mouse"]) != {1, 2, 3, 4}:
        raise RuntimeError("Included mice differ from the approved exclusion")
    selected = json.loads(paths["representatives"].read_text())["selected"]
    statistics = json.loads(paths["statistics"].read_text())
    fcs_root = paths["DMSO_FCS"].parent
    if paths["RARi_FCS"].parent != fcs_root:
        raise RuntimeError("Approved representative FCS files do not share one source directory")
    project = Project.from_workspace(paths["workspace"], fcs_root=fcs_root,
                                     namespace="Sep26_IEL_CCR9")
    # In the portable bundle, workspace URIs retain their original provenance
    # strings. Resolve every sample through the bundle-aware Path adapter before
    # FlowKit opens a file.
    for sample in project.samples.values():
        resolved = Path(sample.source["path"])
        if not resolved.is_file():
            raise FileNotFoundError(f"Workspace FCS source is unavailable: {resolved}")
        sample.source["path"] = str(resolved)
    if shared_compensation_matrix(project).revision != lock["matrix_revision"]:
        raise RuntimeError("Active compensation differs from the approved release")
    figure = plt.figure(figsize=(width / 72, height / 72), facecolor="none")
    anchors, plotted = [], []

    def axis(ident, rectangle):
        x, y, w, h = rectangle
        ax = figure.add_axes([x / width, 1 - (y + h) / height, w / width, h / height])
        anchors.append({"id": ident, "rect_pt": rectangle})
        return ax

    for condition in ("DMSO", "RARi"):
        row = selected[condition]
        record = project.sample(row["sample_id"])
        donor = record.count(project.resolve_population(row["sample_id"], "CD45_1_Donor"))
        positive = record.count(project.resolve_population(row["sample_id"], "CCR9+ P14s"))
        if donor != row["donor_events"] or positive != row["ccr9_positive_events"]:
            raise RuntimeError("Representative gate membership changed")
        ax = axis("A_" + condition + "_flow", settings["flow_axes_pt"][condition])
        project.plot_flow(row["sample_id"], population="CD45_1_Donor", x="PE-Cy7-A", y="BV786-A",
                          gates=["CCR9+ P14s"], annotations=False, ax=ax, title=condition,
                          xlim=(-500, 100000), ylim=(-300, 100000),
                          style=FlowStyle(font_size=6.5, title_size=8, bins=256, smoothing=5,
                                          contour_step=.05, contour_width=.2, gate_width=.65,
                                          dot_size=.16, max_dots=4000))
        ax.set_xlabel("CCR9", labelpad=2)
        ax.set_ylabel("CD45.1", labelpad=2)
        ax.text(.5, .965, f'CCR9+ {row["ccr9_percent"]:.1f}%', transform=ax.transAxes,
                ha="center", va="top", fontsize=7, fontweight="bold")
        plotted.append({"condition": condition, "sample_id": row["sample_id"], "mouse": row["mouse"],
                        "donor_events": donor, "CCR9_positive_events": positive,
                        "percentage": row["ccr9_percent"], "contour_levels": len(ax.flowprism_metadata["contours"]["levels"])})
    summary = axis("A_CCR9_summary", settings["summary_axis_pt"])
    style = replace(SummaryStyle(), font_size=7, title_size=8, tick_rotation=0,
                    marker_size=settings["summary_marker_size_pt"], bar_linewidth=1.0,
                    error_width=.8, cap_size=3, axis_width=.7,
                    colors=("#b4b4b7", "#80a6bc"), edgecolors=("black", "black"))
    plot_summary(table, x="condition", y="ccr9_percent", kind="bar", paired=False,
                 order=["DMSO", "RARi"], style=style, ax=summary, annotate=False,
                 title="", ylabel="CCR9+ donor cells (%)", ylim=(0, 100))
    for x, points in enumerate(c for c in summary.collections if isinstance(c, PathCollection)):
        offsets = points.get_offsets()
        offsets[:, 0] = x + np.linspace(-settings["summary_jitter"], settings["summary_jitter"], len(offsets))
        points.set_offsets(offsets)
        points.set_clip_on(False)
    summary.set_yticks([0, 50, 100])
    summary.set_xticks([0, 1], ["DMSO", "RARi"])
    summary.tick_params(labelsize=7, width=.6, length=2.5, pad=2)
    summary.yaxis.label.set_fontsize(7)
    for spine in summary.spines.values():
        spine.set_linewidth(.7)
    summary.plot([0, 0, 1, 1], [77, 81, 81, 77], color="black", lw=.75)
    summary.text(.5, 84, f'*  P = {statistics["p"]:.4f}', ha="center", va="bottom", fontsize=7)
    figure.text(settings["heading_pt"][0] / width, 1 - settings["heading_pt"][1] / height,
                settings["heading_text"], ha=settings["heading_align"], va="baseline",
                fontsize=8, fontweight="bold")
    for path in outputs[:2]:
        figure.savefig(path, dpi=600, transparent=True, bbox_inches=None, pad_inches=0,
                       metadata={"Creator": "Frozen IEL CCR9 figure renderer"})
    plt.close(figure)
    table.to_csv(outputs[2], index=False)
    outputs[4].write_text(json.dumps({"rendered_anchors": anchors}, indent=2) + "\n")
    inputs = [*lock["inputs"].values(), *renderer_sources,
              {"path": str(Path(lock_path).resolve()), "sha256": sha256(lock_path), "role": "IEL_CCR9_release_lock"},
              {"path": str(Path(layout_path).resolve()), "sha256": sha256(layout_path), "role": "final_page_layout"},
              {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__), "role": "render_script"},
              {"path": font, "sha256": sha256(font), "role": "Arial_font"}]
    outputs[3].write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "panel": "Extended Figure 4A",
        "command": [sys.executable, *sys.argv], "inputs": inputs,
        "outputs": [{"path": str(p), "sha256": sha256(p)} for p in outputs if p != outputs[3]],
        "representatives": plotted, "statistics_source": str(paths["statistics"]),
        "experimental_unit": "independent mouse", "matrix_revision": lock["matrix_revision"],
        "display_settings": settings,
        "gates_measurements_statistics_and_selection_unchanged": True,
        "python": platform.python_version(), "packages": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "model_fitting_run": False, "model_inference_run": False, "statistical_tests_recomputed": False,
        "random_seed": 7, "git_state": "No usable Git history in source workspace",
        "warnings": ["Python contour smoothing approximates FlowJo's proprietary renderer.", "Exploratory endpoint; exact rank sensitivity is recorded in the linked statistics."]}, indent=2) + "\n")
    return outputs[3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-lock", type=Path, required=True)
    parser.add_argument("--page-layout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--renderer-source-dir", type=Path,
                        help="Explicit archived source copies; each must match its existing release hash")
    args = parser.parse_args()
    render(args.release_lock, args.page_layout, args.output_dir, args.overwrite, args.renderer_source_dir)
