"""Regenerate fixed Figure 2 components and retained manual path artwork."""
from __future__ import annotations

import json
from paper_paths import Path, lock_record, rendering_script

from Figure_2.assembly.prepare_common import prepare_package, record, REPO


def manual_text_element(item, number, corrections):
    element = {"id": f"panel_A_manual_text_{number}", "panel": "A", "kind": "text", "text": item["text"],
        "x_pt": item["x_pt"], "y_pt": item["y_pt"],
        "font_size_pt": 7.5 if item["text"].startswith("GAT Convolutional Layer") else item["font_size_pt"],
        "font": "Arial-Bold" if item["paint_count"] > 1 else "Arial",
        "rotation_degrees": item["rotation_degrees"], "color": "#222222"}
    for correction in corrections:
        if item["text"] == correction["source_text"]:
            element["text"] = correction["replacement_text"]
            # Explicit display adjustments affect only the matched label.
            for key in ("x_pt", "y_pt", "align", "font_size_pt"):
                if key in correction:
                    element[key] = correction[key]
            break
    return element


def prepare(output_dir, overwrite=False):
    from figure_assembly.static_art import extract_manual_art, inspect_text
    output_dir = Path(output_dir).resolve()
    result = prepare_package("Figure_2", output_dir, overwrite)
    layout = json.loads((REPO / "Figure_2/assembly/layout.json").read_text())
    elements = []
    # Keep original paint operations intact, but give each panel its own asset.
    for panel, indices in layout["static_groups"]["by_panel"].items():
        static_path = output_dir / f"panel_{panel}_manual_vectors.pdf"
        audit = extract_manual_art(layout["reference_pdf"], static_path,
            source_sha256=layout["reference_sha256"], keep_paint_indices=indices, overwrite=overwrite)
        audit_path = static_path.with_name(static_path.stem + "_audit.json")
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        result["inputs"].append(record(audit_path, "reviewed_static_path_extraction"))
        elements.append({"id": f"panel_{panel}_manual_art", "panel": panel, "kind": "pdf", "path": static_path.name,
                         "rect_pt": [0, 0, 595.276, 841.89], "fit": "exact"})
    elements.extend(result["elements"])
    # Labels remain original artwork in Arial; numerical insets now use the
    # same panel_A renderer as independently regenerated panel components.
    accepted = []
    for item in inspect_text(layout["reference_pdf"]):
        if not 0 < item["y_pt"] < 330 or not item["text"].strip():
            continue
        if any(item["text"] == old["text"] and abs(item["x_pt"]-old["x_pt"]) < .03
               and abs(item["y_pt"]-old["y_pt"]) < .03 for old in accepted):
            continue
        accepted.append(item)
        elements.append(manual_text_element(item, len(accepted), layout.get("manual_text_replacements", [])))
    result["elements"] = elements
    return result
