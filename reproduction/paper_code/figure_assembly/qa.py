"""Geometry, font, content and visual diagnostics for complete figure pages."""
from __future__ import annotations

import json
import re
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from paper_paths import Path, lock_record, rendering_script

import numpy as np
from PIL import Image, ImageChops
from pypdf import PdfReader


def render_png(pdf: Path, output: Path, dpi: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    if dpi < 72:
        raise ValueError("Page preview DPI must be at least 72")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-r", str(dpi), "-png",
                    str(pdf), str(output.with_suffix(""))], check=True, capture_output=True)


def font_check(pdf: Path) -> dict:
    result = subprocess.run(["pdffonts", str(pdf)], check=True, capture_output=True, text=True)
    fonts, errors = [], []
    for line in result.stdout.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 8:
            continue
        name = fields[0]
        # Last five fields: embedded, subset, Unicode, object number, generation.
        embedded, subset, unicode = fields[-5:-2]
        fonts.append({"name": name, "embedded": embedded, "subset": subset, "unicode": unicode})
        if "arial" not in name.lower():
            errors.append(f"Non-Arial font resource: {name}")
        if embedded != "yes":
            errors.append(f"Font is not embedded: {name}")
    return {"fonts": fonts, "errors": errors, "raw": result.stdout}


def geometry_check(layout: dict, assembly: dict, prepared: dict, tolerance_pt=0.1) -> dict:
    errors, measurements = [], []
    expected = {a["id"]: a["rect_pt"] for a in layout.get("anchors", [])}
    actual = {a["id"]: a["rect_pt"] for a in prepared.get("rendered_anchors", [])}
    for name, rectangle in expected.items():
        if name not in actual:
            errors.append(f"Missing measured rendered anchor: {name}")
            continue
        delta = float(np.max(np.abs(np.asarray(rectangle, float) - np.asarray(actual[name], float))))
        measurements.append({"id": name, "expected_rect_pt": rectangle,
                             "actual_rect_pt": actual[name], "max_error_pt": delta})
        if delta > tolerance_pt:
            errors.append(f"Anchor {name} moved by {delta:.4f} pt (limit {tolerance_pt})")
    for element in assembly["elements"]:
        if element.get("nonuniform_scaling"):
            errors.append(f"Non-uniform PDF scaling: {element['id']}; render at its final physical size")
    if not expected:
        errors.append("Layout has no registered geometry anchors")
    return {"tolerance_pt": tolerance_pt, "anchors": measurements, "errors": errors}


def _compact(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text.replace("−", "-").replace("–", "-")).casefold()


def text_check(pdf: Path, layout: dict, prepared: dict) -> dict:
    result = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True)
    # Layout extraction interleaves adjacent multiline category labels. Check
    # both visible layout order and PDF text-object order, retaining exact
    # phrase checks rather than accepting scattered individual words.
    content = [_compact(result.stdout), _compact(PdfReader(pdf).pages[0].extract_text())]
    missing = [t for t in layout.get("expected_text", []) + prepared.get("expected_text", [])
               if not any(_compact(t) in stream for stream in content)]
    obsolete = [t for t in layout.get("forbidden_text", []) + prepared.get("forbidden_text", [])
                if any(_compact(t) in stream for stream in content)]
    return {"missing_expected_text": missing, "forbidden_text_found": obsolete,
            "errors": [*(f"Missing expected text: {t}" for t in missing),
                       *(f"Obsolete/forbidden text remains: {t}" for t in obsolete)], "text": result.stdout}


def text_bounds_check(pdf: Path, page_size) -> dict:
    """Check page-edge clipping; interior typography still needs visual review."""
    result = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], check=True, capture_output=True, text=True)
    document = ET.fromstring(result.stdout)
    outside, count = [], 0
    for word in document.iter():
        if word.tag.rsplit("}", 1)[-1] != "word":
            continue
        count += 1
        left, top, right, bottom = [float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax")]
        if left < -0.5 or top < -0.5 or right > page_size[0] + 0.5 or bottom > page_size[1] + 0.5:
            outside.append({"text": word.text, "bbox_pt": [left, top, right, bottom]})
    return {"word_count": count, "outside_page": outside,
            "errors": [f"Text extends beyond page: {item['text']} at {item['bbox_pt']}" for item in outside]}


def panel_letter_check(pdf: Path, layout: dict) -> dict:
    """Inspect actual PDF words so dotted labels cannot slip into final pages."""
    if layout.get('panel_letter_punctuation') != 'none':
        return {'status': 'NOT_REQUESTED', 'errors': []}
    result = subprocess.run(['pdftotext', '-bbox', str(pdf), '-'], check=True,
                            capture_output=True, text=True)
    words = [node.text or '' for node in ET.fromstring(result.stdout).iter()
             if node.tag.rsplit('}', 1)[-1] == 'word']
    expected = set(layout['panels'])
    labels = sorted({w for w in words if len(w) == 1 and w.upper() in expected})
    dotted = [w for w in words if re.fullmatch(r'[A-Za-z]\.', w) and w[0].upper() in expected]
    missing = sorted(expected - {w.upper() for w in labels})
    errors = ([f'Dotted panel letter remains: {w}' for w in dotted] +
              [f'Missing unpunctuated panel letter: {w}' for w in missing])
    return {'labels': labels, 'dotted_labels': dotted, 'missing': missing, 'errors': errors}


def private_content_check(pdf: Path) -> dict:
    reader = PdfReader(pdf)
    failures = []
    forbidden = {"/PieceInfo", "/Thumb", "/Metadata", "/EmbeddedFiles"}
    seen = set()
    def visit(obj, trail):
        if hasattr(obj, "get_object"):
            obj = obj.get_object()
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key) in forbidden or str(key).startswith("/AIPDFPrivateData"):
                    failures.append(f"Unexpected private/source content at {trail}/{key}")
                if str(key) not in {"/Parent", "/Prev"}:
                    visit(value, f"{trail}/{key}")
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                visit(value, f"{trail}[{index}]")
    visit(reader.trailer["/Root"], "root")
    return {"errors": failures}


def raster_contract_check(layout: dict, assembly: dict) -> dict:
    """Enforce the explicitly registered vector/600-dpi component contracts."""
    errors, records = [], []
    for contract in layout.get("raster_contracts", []):
        paths = [Path(e["path"]) for e in assembly["elements"]
                 if e.get("path") and Path(e["path"]).name == contract["component"]]
        if len(paths) != 1:
            errors.append(f"Raster contract has no unique component: {contract['component']}")
            continue
        result = subprocess.run(["pdfimages", "-list", str(paths[0])], check=True, capture_output=True, text=True)
        images = []
        for line in result.stdout.splitlines()[2:]:
            columns = line.split()
            if len(columns) >= 14 and columns[2] == "image":
                images.append({"width": int(columns[3]), "height": int(columns[4]),
                               "x_ppi": float(columns[12]), "y_ppi": float(columns[13])})
        if contract.get("vector_only") and images:
            errors.append(f"Required vector-only component contains raster images: {paths[0].name}")
        if "image_count" in contract and len(images) != contract["image_count"]:
            errors.append(f"Unexpected raster layer count in {paths[0].name}: {len(images)}")
        minimum = contract.get("minimum_ppi")
        if minimum is not None:
            for item in images:
                if min(item["x_ppi"], item["y_ppi"]) < minimum - 1:
                    errors.append(f"Raster resolution below {minimum} ppi in {paths[0].name}: {item}")
        records.append({"component": str(paths[0]), "contract": contract, "raster_images": images})
    return {"components": records, "errors": errors}


def visual_comparison(reference: Path, output_pdf: Path, directory: Path, dpi=150) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    reference_png = directory / "reference.png"
    result_png = directory / "regenerated.png"
    render_png(reference, reference_png, dpi)
    render_png(output_pdf, result_png, dpi)
    with Image.open(reference_png) as image:
        original = image.convert("RGB")
    with Image.open(result_png) as image:
        result = image.convert("RGB")
    if original.size != result.size:
        raise ValueError(f"Reference and assembled canvases differ: {original.size} vs {result.size}")
    overlay = Image.blend(original, result, 0.5)
    overlay.save(directory / "overlay.png")
    difference = ImageChops.difference(original, result)
    # Fourfold contrast makes small geometric displacements easier to inspect.
    difference.point(lambda value: min(255, value * 4)).save(directory / "difference_4x.png")
    values = np.asarray(difference, dtype=np.float32)
    stats = {"dpi": dpi, "pixel_size": list(original.size),
             "mean_absolute_rgb_difference": float(values.mean()),
             "fraction_pixels_changed_over_16": float((values.max(axis=2) > 16).mean()),
             "interpretation": "Diagnostic only. Updated science and Arial substitution intentionally change pixels; geometry is checked independently."}
    return stats


def check_page(pdf: Path, reference: Path, layout: dict, assembly: dict, prepared: dict, qa_dir: Path) -> dict:
    qa_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(pdf)
    errors = []
    if len(reader.pages) != 1:
        errors.append("Complete figure must have exactly one page")
    page = reader.pages[0]
    size = [float(page.mediabox.width), float(page.mediabox.height)]
    if np.max(np.abs(np.asarray(size) - np.asarray(layout["page_size_pt"]))) > 0.001:
        errors.append(f"Incorrect page size: {size}")
    fonts = font_check(pdf)
    geometry = geometry_check(layout, assembly, prepared)
    text = text_check(pdf, layout, prepared)
    private = private_content_check(pdf)
    bounds = text_bounds_check(pdf, size)
    raster = raster_contract_check(layout, assembly)
    letters = panel_letter_check(pdf, layout)
    for group in (fonts, geometry, text, private, bounds, raster, letters):
        errors.extend(group["errors"])
    (qa_dir / "text.txt").write_text(text.pop("text"))
    (qa_dir / "fonts.txt").write_text(fonts.pop("raw"))
    comparison = visual_comparison(reference, pdf, qa_dir)
    report = {"status": "FAILED" if errors else "AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED",
              "errors": errors, "page_size_pt": size, "fonts": fonts, "geometry": geometry,
              "text": text, "text_bounds": bounds, "raster_contracts": raster, "panel_letters": letters,
              "private_content": private, "visual_comparison": comparison,
              "intentional_differences": list(dict.fromkeys(
                  layout.get("intentional_differences", []) + prepared.get("intentional_differences", [])))}
    (qa_dir / "qa.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def write_review_index(run_dir: Path, entries: list[dict]) -> None:
    import html
    rows = []
    for item in entries:
        name = item["package"]
        panel_links = " · ".join(
            f'<a href="panels/{html.escape(name)}/{Path(panel["pdf"]).name}">Panel {html.escape(panel["panel"])}</a>'
            for panel in item.get("final_panels", []))
        rows.append(f'<section><h2>{html.escape(item["label"])}</h2>'
                    f'<p>{html.escape(item["qa_status"])}</p>'
                    f'<a href="{html.escape(item["pdf"])}">Vector PDF</a>'
                    f'<p>Registered final panels: {panel_links}</p>'
                    f'<div class="pair"><img src="QA/{name}/reference.png" title="Reference">'
                    f'<img src="QA/{name}/regenerated.png" title="Regenerated"></div>'
                    f'<p><a href="QA/{name}/overlay.png">50% overlay</a> · '
                    f'<a href="QA/{name}/difference_4x.png">Difference ×4</a> · '
                    f'<a href="QA/{name}/qa.json">QA report</a></p></section>')
    page = ('<!doctype html><html><head><meta charset="utf-8"><title>Complete figure review</title>'
            '<style>body{font:15px Arial,sans-serif;margin:24px;background:#eee}'
            'section{background:white;margin:20px 0;padding:20px}.pair{display:flex;gap:16px}'
            '.pair img{width:48%;height:auto;object-fit:contain;align-self:start}</style></head><body>'
            '<h1>Complete figure review</h1><p>Reference left; regenerated right. '
            'Current verified science and Arial replace historical content and fonts. '
            'Panel PDFs use the same artwork-sized transparent canvas; import at the origin without scaling.</p>'
            + "".join(rows) + '</body></html>')
    (run_dir / "review.html").write_text(page)
