#!/usr/bin/env python3
"""Fixed-point export adapter for existing, provenance-checked panel renderers.

The renderer still reads its scientific inputs and performs its original checks.
Only the final Figure canvas/artist presentation is adapted. No existing artwork
or panel PDF is opened, copied, cropped, or used as an output background.
"""
from __future__ import annotations

import json
import os
from paper_paths import Path, lock_record, rendering_script
import runpy
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spatial_axes_assembly_mpl")
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.collections import PathCollection


def configure(fig, spec):
    width, height = spec["size_pt"]
    fig.set_layout_engine(None)
    fig.set_size_inches(width / 72, height / 72, forward=True)
    fig.patch.set_alpha(0)
    if len(fig.axes) != len(spec["axes"]):
        raise RuntimeError(f"Axes inventory changed: {len(fig.axes)} != {len(spec['axes'])}")
    for text in fig.findobj(Text):
        text.set_fontfamily("Arial")
        text.set_fontsize(spec.get("font_size_pt", 7))
    for text in fig.texts:
        text.set_visible(False)
    for legend in fig.legends:
        setting = spec.get("figure_legend")
        legend.set_visible(bool(setting))
        if setting:
            x, y = setting["anchor_pt"]
            legend.set_bbox_to_anchor((x / width, 1 - y / height), transform=fig.transFigure)
            legend.borderaxespad = 0
            legend.borderpad = 0
            for text in legend.get_texts():
                text.set_fontfamily("Arial")
                text.set_fontsize(setting.get("font_size_pt", 7))
    for i, (ax, setting) in enumerate(zip(fig.axes, spec["axes"], strict=True)):
        ax.set_axes_locator(None)
        if ax.get_label() == "<colorbar>":
            ax.set_box_aspect(None)
            ax.set_aspect("auto")
        if setting is None:
            ax.set_visible(False)
            continue
        x, y, w, h = setting["rect_pt"]
        ax.set_position([x / width, 1 - (y + h) / height, w / width, h / height])
        ax.patch.set_alpha(0)
        if setting.get('background_color'):
            ax.patch.set_facecolor(setting['background_color'])
            ax.patch.set_alpha(1)
        ax.title.set_fontsize(spec.get("title_size_pt", 7))
        ax.title.set_fontweight("normal")
        ax.title.set_y(1.0)
        ax.set_title(ax.get_title(), pad=3, fontsize=spec.get("title_size_pt", 7), fontweight="normal")
        ax.xaxis.label.set_fontsize(spec.get("label_size_pt", 7))
        ax.yaxis.label.set_fontsize(spec.get("label_size_pt", 7))
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(labelsize=spec.get("tick_size_pt", 6), width=.45, length=2, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(spec.get("spine_width_pt", .45))
        if "title" in setting:
            ax.set_title(setting["title"], fontsize=spec.get("title_size_pt", 7), fontweight="normal", pad=3)
        if "xlabel" in setting:
            ax.set_xlabel(setting["xlabel"])
        if "ylabel" in setting:
            ax.set_ylabel(setting["ylabel"])
        if 'xticklabels' in setting:
            ax.set_xticks(ax.get_xticks(), labels=setting['xticklabels'])
        if 'xtick_rotation' in setting:
            ax.tick_params(axis='x', labelrotation=setting['xtick_rotation'])
        if setting.get("label_left"):
            ax.yaxis.set_label_position("left")
        if "ticks" in setting:
            if setting.get('tick_axis','y')=='x':
                ax.set_xticks(setting['ticks'])
            else:
                ax.set_yticks(setting["ticks"])
        if setting.get("hide_text"):
            for text in ax.findobj(Text):
                text.set_visible(False)
        if "xlim" in setting:
            ax.set_xlim(*setting["xlim"])
        if "ylim" in setting:
            ax.set_ylim(*setting["ylim"])
        if setting.get("match_source_image_footprint"):
            # The reviewed Illustrator image CTM establishes this physical
            # footprint, including artist scaling. No points are discarded.
            inverted = ax.yaxis_inverted()
            ax.set_box_aspect(None)
            ax.set_aspect("auto")
            if not ax.images and not setting.get("preserve_limits"):
                ax.set_xlim(ax.dataLim.x0, ax.dataLim.x1)
                ax.set_ylim((ax.dataLim.y1,ax.dataLim.y0) if inverted else (ax.dataLim.y0,ax.dataLim.y1))
        if setting.get("title_anchor"):
            target = setting["title_anchor"]
            ax.set_title(target.get("text", ax.get_title()), y=1, fontsize=target["font_size_pt"])
            ax.title.set_transform(fig.transFigure)
            ax.title.set_position((target["x_pt"]/width, 1-target["y_pt"]/height))
            ax.title.set_ha(target.get("align", "left"))
            ax.title.set_multialignment(target.get("align", "left"))
            ax.title.set_va("baseline")
        for which in ("x", "y"):
            target = setting.get(which + "label_anchor")
            if target:
                obj = ax.xaxis if which == "x" else ax.yaxis
                obj.set_label_coords(target["x_pt"]/width, 1-target["y_pt"]/height, transform=fig.transFigure)
                obj.label.set_fontsize(target["font_size_pt"])
                obj.label.set_ha("left")
                obj.label.set_va("baseline")
        for collection_index in setting.get("hide_collections_indices", []):
            ax.collections[collection_index].set_visible(False)
        for text in ax.texts:
            text.set_fontsize(setting.get("annotation_size_pt", spec.get("annotation_size_pt", spec.get("font_size_pt", 7))))
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_sizes(collection.get_sizes() * setting.get("marker_area_scale",spec.get("marker_area_scale", 1)))
        for line in ax.lines:
            line.set_linewidth(line.get_linewidth() * spec.get("line_scale", .7))
            line.set_markersize(line.get_markersize() * spec.get("line_scale", .7))
        legend = ax.get_legend()
        if legend:
            if setting.get("hide_legend", spec.get("hide_legend", False)):
                legend.set_visible(False)
            else:
                for text in legend.get_texts():
                    text.set_fontsize(spec.get("legend_size_pt", 6))
    fig.canvas.draw()


def main():
    if len(sys.argv) < 3:
        raise SystemExit("fixed_render.py LAYOUT_JSON RENDERER.py [renderer arguments]")
    layout_path, script = map(Path, sys.argv[1:3])
    layout = json.loads(layout_path.read_text())
    original_save = Figure.savefig

    def fixed_save(fig, filename, *args, **kwargs):
        path = Path(filename)
        spec = layout["render_specs"].get(path.stem)
        if spec is None:
            raise RuntimeError(f"No reviewed fixed-layout export specification for {path.stem}")
        if not getattr(fig, "_fixed_assembly_configured", False):
            configure(fig, spec)
            fig._fixed_assembly_configured = True
        kwargs.update(bbox_inches=None, pad_inches=0, transparent=True, dpi=layout.get("dpi", 300))
        result = original_save(fig, filename, *args, **kwargs)
        if path.suffix == ".pdf":
            width, height = spec["size_pt"]
            dx, dy = spec["placement_pt"][:2]
            anchors = []
            for index, (ax, setting) in enumerate(zip(fig.axes, spec["axes"], strict=True)):
                if setting is None or not setting.get("anchor", True):
                    continue
                bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                anchors.append({"id": f"{path.stem}.axis{index}", "rect_pt": [
                    dx + bounds.x0 * 72, dy + height - bounds.y1 * 72,
                    bounds.width * 72, bounds.height * 72]})
            path.with_suffix(".geometry.json").write_text(json.dumps({
                "layout": str(layout_path.resolve()), "component": path.stem,
                "size_pt": [width, height], "rendered_anchors": anchors,
                "fixed_axes_not_tight_bbox": True, "font_family": "Arial",
            }, indent=2) + "\n")
        return result

    Figure.savefig = fixed_save
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
