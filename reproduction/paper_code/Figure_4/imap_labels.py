"""Import original IMAP label paths and positions; substitute frozen percentages."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from pypdf import PdfReader
from pypdf.generic import ContentStream

from figure_assembly.static_art import inspect_path_events, multiply, point, IDENTITY

REGISTRATION = Path(__file__).with_name("imap_label_registration.json")


@lru_cache(maxsize=2)
def registered_paths(population):
    """Read only the reviewed label-box Bézier paths from the hash-locked PDF."""
    config = json.loads(REGISTRATION.read_text())
    reference = config["references"][population]
    source = Path(reference["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != reference["sha256"]:
        raise RuntimeError(f"IMAP label artwork changed: {source}")
    labels = [label for row in config["populations"][population].values() for label in row["labels"]]
    selected = {label["path_index"]: label for label in labels}
    events = {event["index"]: event for event in inspect_path_events(source)
              if event["index"] in selected}
    if set(events) != set(selected):
        raise RuntimeError("Original IMAP label paths are missing")
    starts, ends = {}, {}
    for index, event in events.items():
        if event["form_path"] != "page" or event["op"] != "f":
            raise RuntimeError("Unexpected original label-box path structure")
        if not np.allclose(event["bbox_pt"], selected[index]["box_rect_pt"], atol=1e-4, rtol=0):
            raise RuntimeError("Original label-box registration does not match the PDF")
        starts[event["operation_index"] - event["path_operations"]] = index
        ends[event["operation_index"]] = index
    reader = PdfReader(source)
    matrix, stack, paths = IDENTITY, [], {}
    active, vertices, codes = None, [], []
    for number, (args, op) in enumerate(ContentStream(reader.pages[0].get_contents(), reader).operations):
        if op == b"q":
            stack.append(matrix)
        elif op == b"Q":
            matrix = stack.pop()
        elif op == b"cm":
            matrix = multiply(tuple(map(float, args)), matrix)
        if number in starts:
            active, vertices, codes = starts[number], [], []
        if active is not None:
            if op in (b"m", b"l"):
                vertices.append(point(matrix, *map(float, args)))
                codes.append(MplPath.MOVETO if op == b"m" else MplPath.LINETO)
            elif op == b"c":
                for offset in (0, 2, 4):
                    vertices.append(point(matrix, *map(float, args[offset:offset + 2])))
                    codes.append(MplPath.CURVE4)
            elif op == b"h":
                vertices.append(vertices[0]); codes.append(MplPath.CLOSEPOLY)
            elif op != b"f":
                raise RuntimeError(f"Unreviewed original label path operation: {op}")
        if number in ends:
            if active != ends[number]:
                raise RuntimeError("Original label path extraction became misaligned")
            vertices.append(vertices[0]); codes.append(MplPath.CLOSEPOLY)
            paths[active] = MplPath(vertices, codes)
            active = None
    return config, paths


def draw_registered_gate_labels(axis, percentages, population, condition, gate_colors):
    """Use original paths, baselines, and line breaks at the current axes scale.

    This deliberately restores user-selected in-plot labels. It does not move
    axes, change gates, or reuse historical numbers or plotted scientific data.
    """
    config, paths = registered_paths(population)
    record = config["populations"][population][condition]
    x, y, width, height = record["axis_rect_pt"]
    page_height = config["page_size_pt"][1]
    transform = Affine2D().translate(-x, -(page_height-y-height)).scale(1/width, 1/height) + axis.transAxes
    figure = axis.figure
    scale = axis.get_position().width * figure.get_figwidth() * 72 / width
    size = config["font_size_pt"] * scale
    artists, texts = [], []
    for label in record["labels"]:
        gate = label["gate"]
        original_path = paths[label["path_index"]]
        path = original_path
        if "display_box_rect_pt" in label:
            bx, by, bw, bh = label["box_rect_pt"]
            dx, dy, dw, dh = label["display_box_rect_pt"]
            vertices = original_path.vertices.copy()
            if label.get("display_fit") == "wrap_preserving_bottom_right":
                # A longer display name grows into the available space above
                # and to the left. The source box, corners and anchor remain
                # registered; only label geometry changes, never gate geometry.
                if label["gate"]!="Muscularis" or not np.allclose(
                        [bx+bw,by+bh],[dx+dw,dy+dh],atol=1e-7,rtol=0):
                    raise RuntimeError("Wrapped gate label moved its bottom-right anchor")
                if dw < bw or dh < bh:
                    raise RuntimeError("Wrapped gate label may only expand")
                vertices[:,0] += np.where(vertices[:,0] < bx+bw/2,dx-bx,0)
                vertices[:,1] += np.where(vertices[:,1] > page_height-by-bh/2,by-dy,0)
            else:
                if not np.allclose([bx+bw/2, by, bh], [dx+dw/2, dy, dh], atol=1e-7, rtol=0):
                    raise RuntimeError("Approved label width adjustment moved its center or height")
                # Widen straight spans only; retain imported corner curves exactly.
                vertices[:, 0] += np.where(vertices[:, 0] < bx+bw/2, -(dw-bw)/2, (dw-bw)/2)
            path = MplPath(vertices, original_path.codes)
        patch = PathPatch(path, transform=transform,
                          facecolor="white", edgecolor=gate_colors[gate], linewidth=.5*scale,
                          clip_on=False, zorder=8)
        axis.add_patch(patch)
        artists.append((label, patch))
        for line in label.get("display_lines",label["lines"]):
            text = line["text"].format(percentage=percentages[gate])
            artist = axis.text((line["x_pt"]+label.get("text_x_shift_pt", 0)-x)/width, 1-(line["y_pt"]-y)/height, text,
                              transform=axis.transAxes, ha=line.get("align","left"), va="baseline",
                              fontsize=size, fontfamily="Arial", fontweight="bold",
                              color="#151515", clip_on=False, zorder=9)
            texts.append((label, line, artist))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    actual_boxes, actual_baselines = [], []
    for label, artist in artists:
        box = artist.get_window_extent(renderer)
        rect = [box.x0/figure.dpi*72, figure.get_figheight()*72-box.y1/figure.dpi*72,
                box.width/figure.dpi*72, box.height/figure.dpi*72]
        actual_boxes.append({"gate": label["gate"], "rect_pt": rect,
                             "reference_rect_pt": label["box_rect_pt"],
                             "expected_rect_pt": label.get("display_box_rect_pt", label["box_rect_pt"]),
                             "display_fit": label.get("display_fit"),
                             "intentional_difference": label.get("intentional_difference")})
    for label, line, artist in texts:
        px, py = artist.get_transform().transform(artist.get_position())
        actual_baselines.append({"gate": label["gate"], "text": artist.get_text(),
                                 "x_pt": px/figure.dpi*72,
                                 "y_pt": figure.get_figheight()*72-py/figure.dpi*72,
                                 "reference_x_pt": line["x_pt"], "reference_y_pt": line["y_pt"],
                                 "expected_x_pt": line["x_pt"]+label.get("text_x_shift_pt", 0)})
    return {"source": config["references"][population], "boxes": actual_boxes,
            "text_baselines": actual_baselines, "font_size_pt": size,
            "original_bezier_paths_imported": True}
