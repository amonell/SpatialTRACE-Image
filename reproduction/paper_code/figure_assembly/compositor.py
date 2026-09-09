"""Compose a fresh PDF page without cloning the assembled artwork reference.

All manifest geometry uses points measured from the top-left of the page.
Text y coordinates identify baselines, not bounding-box tops. Scientific
components should be exported at their final physical size to avoid scaling
fonts, markers and strokes. A page-sized transparent component is supported.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import re
from paper_paths import Path, lock_record, rendering_script

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

FONT_FILES = {
    "Arial": Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    "Arial-Bold": Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
    "Arial-Italic": Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"),
    "Arial-BoldItalic": Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_fonts() -> list[dict]:
    records = []
    for name, path in FONT_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required font unavailable; substitution is forbidden: {path}")
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        records.append({"font": name, "path": str(path), "sha256": file_sha256(path)})
    return records


def rectangle(value, *, page_size=None, name="rectangle") -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{name} must be [x, y, width, height] in points")
    x, y, width, height = map(float, value)
    if not all(math.isfinite(v) for v in (x, y, width, height)) or min(width, height) <= 0:
        raise ValueError(f"Invalid {name}: {value}")
    if page_size is not None:
        pw, ph = page_size
        if x < -0.1 or y < -0.1 or x + width > pw + 0.1 or y + height > ph + 0.1:
            raise ValueError(f"{name} is outside the page: {value}, page {page_size}")
    return x, y, width, height


def pdf_placement(source_size, rect_pt, source_rect_pt=None, fit="exact", alignment="center"):
    """Return the PDF affine matrix and actual top-left destination rectangle."""
    source_width, source_height = map(float, source_size)
    crop = rectangle(source_rect_pt or (0, 0, source_width, source_height),
                     page_size=source_size, name="source rectangle")
    left, top, crop_width, crop_height = crop
    x, y, width, height = rectangle(rect_pt)
    sx, sy = width / crop_width, height / crop_height
    if fit == "contain":
        sx = sy = min(sx, sy)
        dx, dy = width - crop_width * sx, height - crop_height * sy
        if alignment == "center":
            x, y = x + dx / 2, y + dy / 2
        elif alignment != "top-left":
            raise ValueError(f"Unknown containment alignment: {alignment}")
    elif fit != "exact":
        raise ValueError(f"Unknown placement fit: {fit}")
    actual = (x, y, crop_width * sx, crop_height * sy)
    # The y translation is completed by compose_page with the destination height.
    return (sx, 0.0, 0.0, sy, x - left * sx,
            -(y + actual[3]) - (source_height - top - crop_height) * sy), actual, crop


def _color(value):
    if isinstance(value, str):
        text = value.removeprefix("#")
        if len(text) != 6:
            raise ValueError(f"Expected six-digit RGB color: {value}")
        return tuple(int(text[i:i + 2], 16) / 255 for i in (0, 2, 4))
    values = tuple(map(float, value))
    if len(values) != 3 or any(v < 0 or v > 1 for v in values):
        raise ValueError(f"Expected RGB values in [0,1]: {value}")
    return values


def _text_page(element, page_size):
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=tuple(page_size), invariant=1, pageCompression=1,
                    initialFontName="Arial")
    # ReportLab's default unused Helvetica resource is avoided on the page.
    font = element.get("font", "Arial")
    if font not in FONT_FILES:
        raise ValueError(f"Only the installed Arial family is permitted: {font}")
    size = float(element["font_size_pt"])
    if not math.isfinite(size) or size <= 0:
        raise ValueError("Text font size must be positive and finite")
    canvas.setFont(font, size)
    canvas.setFillColorRGB(*_color(element.get("color", "#000000")))
    canvas.translate(float(element["x_pt"]), page_size[1] - float(element["y_pt"]))
    canvas.rotate(float(element.get("rotation_degrees", 0)))
    alignment = element.get("align", "left")
    draw = {"left": canvas.drawString, "center": canvas.drawCentredString,
            "right": canvas.drawRightString}.get(alignment)
    if draw is None:
        raise ValueError(f"Unknown text alignment: {alignment}")
    for index, line in enumerate(str(element["text"]).split("\n")):
        draw(0, -index * float(element.get("line_height_pt", size * 1.2)), line)
    canvas.showPage()
    canvas.save()
    reader = PdfReader(buffer)
    return reader.pages[0]


def _prune_unused_fonts(page):
    """Remove unused default fonts; retain actual content-selected font resources."""
    from pypdf.generic import ContentStream
    seen = set()
    def visit(owner):
        identity = id(owner)
        if identity in seen:
            return
        seen.add(identity)
        resources = owner.get("/Resources", {})
        resources = resources.get_object() if hasattr(resources, "get_object") else resources
        if not resources:
            return
        content = owner.get_contents() if hasattr(owner, "get_contents") else ContentStream(owner, page.pdf)
        # ContentStream is a dictionary whose truth value can be False despite
        # nonempty operations. Track fonts that actually paint glyphs, not the
        # unused default font selected by ReportLab's initial BT/ET block.
        used, current_font, stack = set(), None, []
        if content is not None:
            for args, op in content.operations:
                if op == b"q":
                    stack.append(current_font)
                elif op == b"Q" and stack:
                    current_font = stack.pop()
                elif op == b"Tf":
                    current_font = str(args[0])
                elif op in {b"Tj", b"TJ", b"'", b'"'} and current_font:
                    used.add(current_font)
            content.operations = [(args, op) for args, op in content.operations
                                  if op != b"Tf" or str(args[0]) in used]
            if hasattr(owner, "replace_contents"):
                owner.replace_contents(content)
            else:
                owner.set_data(content.get_data())
        fonts = resources.get("/Font", {})
        fonts = fonts.get_object() if hasattr(fonts, "get_object") else fonts
        for key in list(fonts):
            if str(key) not in used:
                del fonts[key]
        xobjects = resources.get("/XObject", {})
        xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
        for reference in xobjects.values():
            obj = reference.get_object()
            if obj.get("/Subtype") == "/Form":
                visit(obj)
    visit(page)


def format_panel_letter(element: dict, layout: dict) -> dict:
    """Apply the approved punctuation style only to standalone panel labels.

    Original artwork/source text and its anchor remain unchanged in the input
    records. The composition records carry the actual newly rendered text.
    Decimal points, abbreviations, punctuation in titles and metric labels
    are not affected.
    """
    if layout.get('panel_letter_punctuation') != 'none' or element.get('kind') != 'text':
        return element
    label = str(element.get('text', '')).strip()
    if re.fullmatch(r'[A-Za-z]\.', label) and label[0].upper() in layout.get('panels', []):
        return {**element, 'text': label[0]}
    return element


def compose_page(layout: dict, component_root: Path, output_pdf: Path) -> dict:
    """Write one complete page from newly prepared, explicit component paths."""
    fonts = register_fonts()
    page_size = tuple(map(float, layout["page_size_pt"]))
    if len(page_size) != 2 or min(page_size) <= 0:
        raise ValueError("page_size_pt must contain two positive numbers")
    output_pdf = Path(output_pdf).resolve()
    if output_pdf.exists():
        raise FileExistsError(output_pdf)
    writer = PdfWriter()
    page = writer.add_blank_page(width=page_size[0], height=page_size[1])
    placements, readers = [], []
    ids = set()
    elements = sorted(enumerate(layout["elements"]), key=lambda pair: (pair[1].get("z", 0), pair[0]))
    for _, element in elements:
        element = format_panel_letter(element, layout)
        identifier = element["id"]
        if identifier in ids:
            raise ValueError(f"Duplicate element id: {identifier}")
        ids.add(identifier)
        kind = element["kind"]
        if kind == "text":
            page.merge_page(_text_page(element, page_size))
            placements.append({"id": identifier, "kind": kind, "text": element["text"],
                               "anchor_pt": [element["x_pt"], element["y_pt"]],
                               "font": element.get("font", "Arial"), "font_size_pt": element["font_size_pt"]})
            continue
        if kind not in {"pdf", "static"}:
            raise ValueError(f"Unsupported element kind: {kind}")
        destination = rectangle(element["rect_pt"], page_size=page_size, name=identifier)
        path = Path(element["path"])
        path = (component_root / path).resolve() if not path.is_absolute() else path.resolve()
        # Fresh scientific overlays and structurally filtered artwork assets are
        # expected under the preparation directory, never the original PDF.
        if not path.is_relative_to(component_root.resolve()):
            raise ValueError(f"Component must be freshly prepared inside {component_root}: {path}")
        digest = file_sha256(path)
        if element.get("sha256") and digest != element["sha256"]:
            raise ValueError(f"Component hash mismatch: {path}")
        reader = PdfReader(path)
        readers.append(reader)  # Keep source objects alive until serialization.
        source = copy.copy(reader.pages[int(element.get("page", 0))])
        if int(source.get("/Rotate", 0)) % 360:
            raise ValueError(f"Component rotation must be normalized by its renderer: {path}")
        source_size = (float(source.mediabox.width), float(source.mediabox.height))
        if abs(float(source.mediabox.left)) > 1e-6 or abs(float(source.mediabox.bottom)) > 1e-6:
            raise ValueError(f"Component must use zero-origin page geometry: {path}")
        matrix, actual, crop = pdf_placement(source_size, destination,
                                            element.get("source_rect_pt"), element.get("fit", "exact"),
                                            element.get("alignment", "center"))
        matrix = (*matrix[:5], matrix[5] + page_size[1])
        left, top, width, height = crop
        source.cropbox = RectangleObject((left, source_size[1] - top - height,
                                          left + width, source_size[1] - top))
        page.merge_transformed_page(source, Transformation(matrix), expand=False)
        placements.append({"id": identifier, "kind": kind, "path": str(path), "sha256": digest,
                           "requested_rect_pt": list(destination), "actual_rect_pt": list(actual),
                           "source_rect_pt": list(crop), "matrix": list(matrix),
                           "nonuniform_scaling": abs(matrix[0] - matrix[3]) > 1e-8})
    _prune_unused_fonts(page)
    writer.add_metadata({"/Title": layout.get("title", layout.get("package", "Manuscript figure")),
                         "/Creator": "spatial_axes_manuscript fixed-coordinate figure assembly",
                         "/Producer": "pypdf / ReportLab; frozen scientific inputs"})
    writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as handle:
        writer.write(handle)
    return {"page_size_pt": list(page_size), "elements": placements, "fonts": fonts,
            "output": {"path": str(output_pdf), "sha256": file_sha256(output_pdf)}}
