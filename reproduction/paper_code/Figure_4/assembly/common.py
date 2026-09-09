"""Final-page-sized plotting utilities; no fitting, inference, or metric creation."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from paper_paths import Path, lock_record, rendering_script

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle

from Figure_4.release import load_release, release_file
from Figure_4.spatial import GATE_COLORS, gate_background, frozen_density, frozen_gate_percentages
from Figure_4.imap_labels import REGISTRATION, draw_registered_gate_labels

HOME = Path(__file__).resolve().parents[2]
SATA = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")
REPAIR = SATA / "Figure_4/stored_outputs/scientific_repair_v1"
EVALUATION = REPAIR / "evaluation"
PAGE = (595.276, 841.89)
RELEASE_SHA = "f1b3d77147ad618ae3b02554a20ebbc8abefd161893e9f14966db67c3012d694"
PIXEL_UM = 0.16247698804244626


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure():
    for weight in ("normal", "bold"):
        findfont(FontProperties(family="Arial", weight=weight), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"],
                         "font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": .5,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})


class PageInputs:
    def __init__(self, output_dir, overwrite=False):
        configure()
        self.out = Path(output_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        self.inputs, self.elements, self.anchors, self.warnings = [], [], [], []
        self.details, self.outputs = {}, []
        self.read(EVALUATION / "release_manifest.json", RELEASE_SHA, "verified_IF_release")
        self.read(HOME / "frozen_preprocessing/frozen_model_registry.json", role="active_model_registry")
        for helper in ("Figure_4/release.py", "Figure_4/spatial.py", "Figure_4/common.py"):
            self.read(HOME / helper, role="rendering_dependency")
        self.read(REGISTRATION, role="original_IMAP_label_registration")
        self.read(HOME / "Figure_4/imap_labels.py", role="original_IMAP_label_importer")
        self.read(HOME / "figure_assembly/static_art.py", role="original_vector_path_parser")
        for weight in ("normal", "bold"):
            self.read(findfont(FontProperties(family="Arial", weight=weight), fallback_to_default=False), role="Arial_font_" + weight)
        self.release = load_release(EVALUATION / "release_manifest.json")
        for key in ("checkpoint", "lock", "xenium_initialization", "test_identity_manifest"):
            r = self.release[key]
            self.read(r["path"], r["sha256"], key)

    def read(self, path, expected=None, role="frozen_input"):
        path = Path(path).resolve()
        existing = next((r for r in self.inputs if r["path"] == str(path)), None)
        actual = existing["sha256"] if existing else sha256(path)
        if expected and actual != expected:
            raise RuntimeError(f"Input hash mismatch: {path}")
        if not existing:
            self.inputs.append({"path": str(path), "sha256": actual, "role": role})
        return path

    def frozen(self, relative):
        path = EVALUATION / relative
        release_file(path, self.release)
        return self.read(path, self.release["outputs"][relative]["sha256"], relative)

    def stage(self, relative):
        path = REPAIR / relative
        release_file(path, self.release)
        return self.read(path, role=relative)

    def figure(self):
        return plt.figure(figsize=(PAGE[0] / 72, PAGE[1] / 72), dpi=600, facecolor="none")

    def axis(self, figure, ident, rect, **kwargs):
        x, y, width, height = rect
        axis = figure.add_axes([x / PAGE[0], 1 - (y + height) / PAGE[1],
                                width / PAGE[0], height / PAGE[1]], **kwargs)
        axis._assembly_id = ident
        axis.tick_params(width=.5, length=2.5, pad=2)
        return axis

    def save(self, figure, name):
        path = self.out / f"{name}.pdf"
        if path.exists() and not self.overwrite:
            raise FileExistsError(path)
        figure.canvas.draw()
        for axis in figure.axes:
            if hasattr(axis, "_assembly_id"):
                p = axis.get_position()
                self.anchors.append({"id": axis._assembly_id,
                                     "rect_pt": [p.x0 * PAGE[0], (1 - p.y1) * PAGE[1],
                                                 p.width * PAGE[0], p.height * PAGE[1]]})
        # Every overlay has the exact page box. No tight bounding boxes, source
        # page copies, or flattened old science are used.
        figure.savefig(path, dpi=600, transparent=True, bbox_inches=None, pad_inches=0,
                       metadata={"Title": name, "Creator": "Frozen scientific assembly renderer"})
        plt.close(figure)
        panel = name.split("_")[1]
        if len(panel) != 1 or panel not in "ABCDEF":
            raise RuntimeError(f"Scientific layer has no explicit panel: {name}")
        self.elements.append({"id": name, "panel": panel, "kind": "pdf", "path": path.name,
                              "rect_pt": [0, 0, *PAGE], "fit": "exact"})
        self.outputs.append({"path": str(path), "sha256": sha256(path)})

    def text(self, text, x, y, size=8, bold=False, align="left", color="#000000", *, panel):
        self.elements.append({"id": f"text_{len(self.elements):03d}", "panel": panel, "kind": "text", "text": text, "x_pt": x, "y_pt": y,
                              "font_size_pt": size, "font": "Arial-Bold" if bold else "Arial",
                              "align": align, "color": color})

    def finish(self, layout, script):
        self.read(script, role="assembly_renderer")
        self.read(Path(__file__), role="assembly_helpers")
        self.read(layout, role="exact_layout_manifest")
        allowed_panels = set(json.loads(Path(layout).read_text())["panels"])
        if any(element.get("panel") not in allowed_panels for element in self.elements):
            raise RuntimeError("Every regenerated element must belong to one visible panel")
        data = {"elements": self.elements, "inputs": self.inputs,
                "commands": [[sys.executable, str(script), "--output-dir", str(self.out),
                              *(["--overwrite"] if self.overwrite else [])]],
                "warnings": self.warnings, "rendered_anchors": self.anchors,
                "intentional_differences": json.loads(Path(layout).read_text()).get("intentional_differences", []),
                "outputs": self.outputs, "details": self.details,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "packages": {"numpy": np.__version__,
                    "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "random_seed": None, "model_inference_run": False,
                "model_fitting_run": False, "metric_tables_created": False,
                "git_state": "No usable Git history in this workspace"}
        target = self.out / "assembly_inputs_provenance.json"
        if target.exists() and not self.overwrite:
            raise FileExistsError(target)
        target.write_text(json.dumps(data, indent=2) + "\n")
        return data


def top_text(figure, x, y, text, **kwargs):
    return figure.text(x / PAGE[0], 1 - y / PAGE[1], text, va="baseline", **kwargs)


def colorbar(figure, axis, artist, *, horizontal=False, ticks=None, label=None, vector=True):
    bar = figure.colorbar(artist, cax=axis, orientation="horizontal" if horizontal else "vertical", ticks=ticks)
    bar.outline.set_linewidth(.45)
    bar.ax.tick_params(labelsize=7, length=2, width=.45, pad=2)
    if label:
        bar.set_label(label, fontsize=7.5, labelpad=3)
    if vector and bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
    return bar


def imap(page, figure, condition, population, rect, *, title_y,
         show_xticks=True, xlabel=False, title_x=None, xlabel_anchor=None):
    suffix = "p14" if population == "P14" else "non_p14"
    key = f"{condition.lower()}_{suffix}"
    density_manifest = json.loads(page.frozen("imap/density_manifest.json").read_text())
    if not density_manifest["all_four_populations_share_normalization"]:
        raise RuntimeError("The four IMAP populations do not share density normalization")
    record = density_manifest["populations"][key]
    frame = pd.read_parquet(page.read(record["cells"]["path"], record["cells"]["sha256"], key + "_cells"))
    page.read(record["grid"]["path"], record["grid"]["sha256"], key + "_density_grid")
    density, gx, gy, grid = frozen_density(frame, density_manifest, key)
    table = pd.read_csv(page.frozen("imap/p14_composition_by_gate.tsv" if population == "P14" else
                                    "imap/non_p14_counts_and_fractions.tsv"), sep="\t")
    fractions = frozen_gate_percentages(table, condition)
    definitions = json.loads(page.frozen("imap/p14_gate_definitions.json").read_text())
    if definitions["checkpoint_sha256"] != page.release["checkpoint"]["sha256"]:
        raise RuntimeError("Gate geometry belongs to a different IF checkpoint")
    axis = page.axis(figure, ("F_" if population == "P14" else "D_") + condition, rect)
    gate_background(axis, definitions)
    order = np.argsort(density)
    artist = axis.scatter(frame.predicted_epithelial_distance_clipped_1p0.to_numpy()[order],
                          frame.predicted_axis_coordinate.to_numpy()[order], c=density[order],
                          cmap="viridis", norm=Normalize(*density_manifest["shared_color_limits"]),
                          s=2.5 if population == "P14" else .35, linewidths=0, alpha=.86,
                          rasterized=population != "P14")
    levels = [v for v in density_manifest["shared_contour_levels"] if grid.min() < v < grid.max()]
    if levels:
        axis.contour(gx, gy, grid, levels=levels, colors="#666666", linewidths=.55, alpha=.85, zorder=4)
    axis.set(xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
    axis.tick_params(labelbottom=show_xticks)
    axis.set_ylabel("Crypt-villus coordinate", labelpad=3)
    if xlabel and xlabel_anchor is None:
        axis.set_xlabel("Epithelial-distance coordinate", labelpad=17)
    if xlabel_anchor is not None:
        top_text(figure, *xlabel_anchor, "Epithelial-distance coordinate", fontsize=8)
    top_text(figure, rect[0] + rect[2] / 2 if title_x is None else title_x, title_y, f"{condition} {population} cells",
             ha="center", fontsize=8.5, fontweight="bold")
    registration = draw_registered_gate_labels(axis, fractions, population, condition, GATE_COLORS)
    page.read(registration["source"]["path"], registration["source"]["sha256"], "original_IMAP_label_artwork")
    for box in registration["boxes"]:
        ident = ("F_" if population == "P14" else "D_") + condition + "_label_" + box["gate"].replace(" ", "_")
        if not np.allclose(box["rect_pt"], box["expected_rect_pt"], rtol=0, atol=.001):
            raise RuntimeError(f"Original gate label moved: {ident}")
        page.anchors.append({"id": ident, "rect_pt": box["rect_pt"]})
    page.details[key] = {"cells": len(frame), "gate_percentages": fractions,
                         "gate_thresholds": definitions["thresholds"],
                         "frozen_density": True, "pdf_points_vector": population == "P14",
                         "density_limits": density_manifest["shared_color_limits"],
                         "density_levels": density_manifest["shared_contour_levels"],
                         "gate_label_registration": registration}
    return artist
