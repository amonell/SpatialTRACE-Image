"""One canonical, registered PDF layer per panel, made from fresh render elements.

Panel PDFs retain the full artwork canvas so importing at (0, 0), without
scaling, preserves every position. They contain only their own panel's artwork,
text and newly rendered data. They are not crops of an old assembled PDF.
"""
from __future__ import annotations

import copy
import json
from paper_paths import Path, lock_record, rendering_script

from PIL import Image, ImageChops

from figure_assembly.compositor import compose_page, file_sha256
from figure_assembly.qa import font_check, private_content_check, render_png, text_bounds_check
from figure_assembly.static_art import extract_manual_art, inspect_path_events


def schematic_owner(package, x, y, text=None):
    """Reviewed Figure 1/3 panel ownership; geometry itself is never changed."""
    label = str(text or "").strip().lower()
    if label in {"a.", "b.", "c.", "d.", "e."}:
        return label[0].upper()
    if package == "Figure_1":
        if y < 163:
            return "A" if x < 280 else "B"
        return "C" if y < 383 else "D" if y < 607 else "E"
    if package == "Figure_3":
        if y < 380:
            return "A"
        if y < 538:
            return "B" if x < 234 else "C"
        return "D" if y < 676 else "E"
    raise ValueError(f"No reviewed implicit ownership rule for {package}")


def partition_elements(layout, prepared, component_root):
    """Expand the two schematic science layers and partition manual paths."""
    package = layout["package"]
    panels = layout["panels"]
    root = Path(component_root).resolve()
    grouped = {letter: [] for letter in panels}
    audits = []
    for source in prepared["elements"]:
        element = copy.deepcopy(source)
        letter = element.get("panel")
        if letter is not None:
            if letter not in grouped:
                raise ValueError(f"Unknown panel owner {letter}: {element['id']}")
            grouped[letter].append(element)
            continue
        if package not in {"Figure_1", "Figure_3"}:
            raise ValueError(f"Final element lacks explicit panel ownership: {package}/{element['id']}")
        if element["kind"] == "text":
            letter = schematic_owner(package, element["x_pt"], element["y_pt"], element["text"])
            element["panel"] = letter
            grouped[letter].append(element)
            continue
        path = (root / element["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Panel source must be freshly prepared: {path}")
        if path.name == "scientific_layers.pdf":
            layers = prepared["scientific_panel_layers"]
            if set(layers) != set(panels):
                raise ValueError(f"Scientific panel layer coverage mismatch: {package}")
            for letter in panels:
                split = {**element, "id": f"{element['id']}-{letter}",
                         "path": layers[letter], "panel": letter}
                split.pop("sha256", None)
                grouped[letter].append(split)
            continue
        if path.name not in {"manual_background.pdf", "manual_foreground.pdf"}:
            raise ValueError(f"Unregistered mixed-panel source: {path}")
        selections = {letter: [] for letter in panels}
        events = inspect_path_events(path)
        for event in events:
            x, y, width, height = event["bbox_pt"]
            if width * height > .7 * layout["page_size_pt"][0] * layout["page_size_pt"][1]:
                raise ValueError("A page-wide manual surface needs explicit panel ownership")
            letter = schematic_owner(package, x + width / 2, y + height / 2)
            selections[letter].append(event["index"])
        assert sum(map(len, selections.values())) == len(events)
        for letter, indices in selections.items():
            if not indices:
                continue
            relative = Path("panel_art") / f"{path.stem}_{letter}.pdf"
            audit = extract_manual_art(path, root / relative, source_sha256=file_sha256(path),
                                       keep_paint_indices=indices)
            if audit["image_paints_removed"] or audit["text_paints_removed"]:
                raise ValueError("A manual-only intermediate unexpectedly contained image/text paint")
            audit["panel"] = letter
            audits.append(audit)
            split = {**element, "id": f"{element['id']}-{letter}", "path": str(relative), "panel": letter}
            split.pop("sha256", None)
            grouped[letter].append(split)
    if any(not elements for elements in grouped.values()):
        raise ValueError(f"Missing final panel content: {package}")
    return grouped, audits


def build_panel_layers(layout, prepared, component_root):
    root = Path(component_root).resolve()
    groups, audits = partition_elements(layout, prepared, root)
    records, full_elements = [], []
    dpi = int(layout.get("png_dpi", 600 if layout["package"] in {"Figure_4", "Figure_4_Extended"} else 300))
    for letter, elements in groups.items():
        relative = Path("final_panels") / f"panel_{letter}_final.pdf"
        pdf = root / relative
        png = pdf.with_suffix(".png")
        composition = compose_page({**layout, "elements": elements}, root, pdf)
        render_png(pdf, png, dpi)
        checks = {"fonts": font_check(pdf), "private_content": private_content_check(pdf),
                  "text_bounds": text_bounds_check(pdf, layout["page_size_pt"])}
        errors = [error for check in checks.values() for error in check["errors"]]
        if errors:
            raise RuntimeError(f"{layout['package']} panel {letter}: {errors}")
        record = {"panel": letter, "pdf": str(pdf), "png": str(png),
                  "pdf_sha256": file_sha256(pdf), "png_sha256": file_sha256(png),
                  "canvas_pt": layout["page_size_pt"], "png_dpi": dpi,
                  "elements": elements, "composition": composition,
                  "qa": checks, "warnings": prepared.get("warnings", [])}
        provenance = pdf.with_name(pdf.stem + "_provenance.json")
        provenance.write_text(json.dumps(record, indent=2) + "\n")
        record["provenance"] = str(provenance)
        records.append(record)
        full_elements.append({"id": f"final-panel-{letter}", "panel": letter,
                              "kind": "pdf", "path": str(relative),
                              "rect_pt": [0, 0, *layout["page_size_pt"]], "sha256": record["pdf_sha256"]})
    return {"panels": records, "elements": full_elements, "manual_partition_audits": audits}


def verify_recomposition(direct_pdf, layered_pdf, output_dir):
    """Require literal pixel equality between direct and panel-layer assembly."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    direct = directory / "direct.png"
    layered = directory / "panel_layers.png"
    render_png(Path(direct_pdf), direct, 150)
    render_png(Path(layered_pdf), layered, 150)
    with Image.open(direct) as a, Image.open(layered) as b:
        if a.size != b.size:
            raise ValueError("Panel recomposition changed page dimensions")
        difference = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
        bbox = difference.getbbox()
        if bbox is not None:
            difference.save(directory / "recomposition_difference.png")
            raise ValueError(f"Panel-layer assembly differs from direct rendering; pixel bounds {bbox}")
    return {"status": "PIXEL_IDENTICAL", "dpi": 150,
            "direct_render_sha256": file_sha256(direct), "panel_render_sha256": file_sha256(layered)}
