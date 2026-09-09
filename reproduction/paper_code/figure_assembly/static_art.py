"""Extract reviewed manual PDF paths, never a flattened reference-page background.

All rectangles use top-left PDF points: [x, y, width, height]. Path indices are
stable depth-first paint-event indices for the hash-locked source page. Images,
text, marked-content metadata, Illustrator private streams, and unselected path
paint events are excluded. Clipping and graphics state are preserved.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from paper_paths import Path, lock_record, rendering_script
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, ContentStream, DecodedStreamObject, DictionaryObject,
    FloatObject, NameObject,
)
import pypdf.filters

IDENTITY = (1., 0., 0., 1., 0., 0.)
PAINT = {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}
PATH = {b"m", b"l", b"c", b"v", b"y", b"re", b"h"}
TEXT = {b"BT", b"ET", b"Tc", b"Tw", b"Tz", b"TL", b"Tf", b"Tr", b"Ts",
        b"Td", b"TD", b"Tm", b"T*", b"Tj", b"TJ", b"'", b'"'}
SKIP = {b"BDC", b"BMC", b"EMC", b"MP", b"DP", b"BX", b"EX"}


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reader(path: Path | str) -> PdfReader:
    # The verified Figure 1 contains one 82.8 MB compressed image. This bounded
    # parser allowance permits reading its dictionary; images are never decoded
    # or copied into the extracted static asset.
    pypdf.filters.MAX_DECLARED_STREAM_LENGTH = max(
        pypdf.filters.MAX_DECLARED_STREAM_LENGTH, 100_000_000)
    return PdfReader(str(Path(path)))


def multiply(a, b):
    """Composition b(a(point)); PDF cm concatenation in row-vector notation."""
    return (a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
            a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3],
            a[4]*b[0]+a[5]*b[2]+b[4], a[4]*b[1]+a[5]*b[3]+b[5])


def point(matrix, x, y):
    return (matrix[0]*x + matrix[2]*y + matrix[4],
            matrix[1]*x + matrix[3]*y + matrix[5])


def contains(rect, bbox, tolerance=.02):
    x, y, w, h = map(float, rect)
    bx, by, bw, bh = bbox
    return bx >= x-tolerance and by >= y-tolerance and bx+bw <= x+w+tolerance and by+bh <= y+h+tolerance


def intersects(rect, bbox, tolerance=.001):
    x, y, w, h = map(float, rect); bx, by, bw, bh = bbox
    return min(x+w, bx+bw)-max(x,bx) > tolerance and min(y+h,by+bh)-max(y,by) > tolerance


def _numbers(values):
    return [FloatObject(float(v)) for v in values]


class _Walker:
    def __init__(self, reader, page_number=0, select=None):
        self.reader = reader
        self.page = reader.pages[page_number]
        self.height = float(self.page.mediabox.height)
        self.select = select
        self.events = []
        self.kept = []
        self.images_removed = 0
        self.text_paint_removed = 0
        self.resources = DictionaryObject()
        self.warnings = []

    def form_has_paths(self, obj, resources, depth=0):
        if depth > 12:raise RuntimeError("Excessive Form nesting")
        for operands,op in ContentStream(obj,self.reader).operations:
            if op in PAINT:return True
            if op==b"Do":
                child=resources["/XObject"][operands[0]].get_object()
                if child.get("/Subtype")=="/Form" and self.form_has_paths(child,child.get("/Resources",resources),depth+1):
                    return True
        return False

    def count_discarded_form(self, obj, resources, depth=0):
        if depth > 12:raise RuntimeError("Excessive Form nesting")
        for operands,op in ContentStream(obj,self.reader).operations:
            self.text_paint_removed+=int(op in (b"Tj",b"TJ",b"'",b'"'))
            if op==b"Do":
                child=resources["/XObject"][operands[0]].get_object()
                if child.get("/Subtype")=="/Image":self.images_removed+=1
                elif child.get("/Subtype")=="/Form":self.count_discarded_form(child,child.get("/Resources",resources),depth+1)

    def resource(self, resources, category, name, context):
        original = resources.get(category, {}).get(name)
        if original is None:
            raise RuntimeError(f"Missing {category} {name} in {context}")
        obj = original.get_object()
        if category == "/ExtGState" and obj.get("/SMask") not in (None, "/None"):
            raise RuntimeError(f"Manual path depends on an unreviewed soft-mask group: {context}/{name}")
        bucket = self.resources.setdefault(NameObject(category), DictionaryObject())
        new = NameObject(f"/R{len(bucket)}")
        # Clone only resources required by retained state operators. Never clone
        # page /Resources wholesale, /XObject, /Font, or /PieceInfo.
        bucket[new] = original
        return new

    def walk(self, stream, resources, matrix=IDENTITY, context="page", depth=0):
        if depth > 12:
            raise RuntimeError("Excessive nested Form depth")
        output = []
        current = tuple(matrix)
        stack = []
        pending = []
        points = []
        clip = False
        for op_index, (operands, op) in enumerate(ContentStream(stream, self.reader).operations):
            operands = list(operands)
            if op in TEXT:
                self.text_paint_removed += int(op in (b"Tj", b"TJ", b"'", b'"'))
                continue
            if op in SKIP:
                continue
            if op == b"q":
                stack.append(current); output.append((operands, op)); continue
            if op == b"Q":
                if not stack:
                    raise RuntimeError(f"Unbalanced graphics stack in {context}")
                current = stack.pop(); output.append((operands, op)); continue
            if op == b"cm":
                current = multiply(tuple(map(float, operands)), current)
                output.append((operands, op)); continue
            if op in PATH:
                pending.append((operands, op))
                values = list(map(float, operands))
                if op == b"re":
                    x,y,w,h=values
                    points.extend(point(current,a,b) for a,b in ((x,y),(x+w,y),(x+w,y+h),(x,y+h)))
                elif op != b"h":
                    points.extend(point(current, values[i], values[i+1]) for i in range(0,len(values),2))
                continue
            if op in (b"W", b"W*"):
                clip=True; pending.append((operands,op)); continue
            if op in PAINT or op == b"n":
                keep=False
                if op in PAINT:
                    if not points:
                        raise RuntimeError(f"Paint operation has no measurable path in {context}")
                    xs,ys=zip(*points)
                    bbox=[min(xs),self.height-max(ys),max(xs)-min(xs),max(ys)-min(ys)]
                    event={"index":len(self.events),"bbox_pt":[round(v,6) for v in bbox],
                           "op":op.decode(),"form_path":context,"operation_index":op_index,
                           "path_operations":len(pending)}
                    self.events.append(event)
                    keep=bool(self.select(event)) if self.select else False
                    if keep:self.kept.append(event["index"])
                if keep:
                    output.extend(pending); output.append((operands,op))
                elif clip:
                    output.extend(pending); output.append(([],b"n"))
                pending=[];points=[];clip=False
                continue
            if op == b"Do":
                key=operands[0]; obj=resources["/XObject"][key].get_object()
                if obj.get("/Subtype")=="/Image":
                    self.images_removed+=1;continue
                if obj.get("/Subtype")!="/Form":
                    raise RuntimeError(f"Unknown XObject type in {context}/{key}")
                child_resources=obj.get("/Resources",resources)
                if not self.form_has_paths(obj,child_resources):
                    self.count_discarded_form(obj,child_resources)
                    continue
                form_matrix=tuple(map(float,obj.get("/Matrix",IDENTITY)))
                children=self.walk(obj,child_resources,multiply(form_matrix,current),
                                   f"{context}/{key}@{op_index}",depth+1)
                output.append(([],b"q"));output.append((_numbers(form_matrix),b"cm"))
                if "/BBox" in obj:
                    x0,y0,x1,y1=map(float,obj["/BBox"])
                    output.append((_numbers((min(x0,x1),min(y0,y1),abs(x1-x0),abs(y1-y0))),b"re"))
                    output.extend([([],b"W"),([],b"n")])
                output.extend(children);output.append(([],b"Q"));continue
            if op == b"INLINE IMAGE":
                self.images_removed+=1;continue
            if op == b"gs":
                operands[0]=self.resource(resources,"/ExtGState",operands[0],context)
            elif op in (b"CS",b"cs") and operands[0] not in ("/DeviceGray","/DeviceRGB","/DeviceCMYK","/Pattern"):
                operands[0]=self.resource(resources,"/ColorSpace",operands[0],context)
            elif op in (b"sh",b"SCN",b"scn"):
                raise RuntimeError(f"Unreviewed shading or pattern paint {op!r} in {context}")
            output.append((operands,op))
        if stack or pending:
            raise RuntimeError(f"Unbalanced graphics/path state in {context}")
        return output


def inspect_path_events(reference: Path | str, page_number=0) -> list[dict[str,Any]]:
    reader=_reader(reference);walker=_Walker(reader,page_number)
    walker.walk(walker.page["/Contents"],walker.page["/Resources"])
    return walker.events


def extract_manual_art(reference: Path | str, output_pdf: Path | str, *,
                       source_sha256: str,
                       keep_paint_indices=None, keep_regions_pt=(), drop_regions_pt=(),
                       drop_form_names=(), page_number=0, overwrite=False):
    reference=Path(reference); output_pdf=Path(output_pdf)
    actual=sha256(reference)
    if actual!=source_sha256:raise RuntimeError(f"Artwork hash mismatch: {reference}")
    if output_pdf.exists() and not overwrite:raise FileExistsError(output_pdf)
    if keep_paint_indices is None and not keep_regions_pt:
        raise ValueError("An explicit manual-path allowlist or reviewed keep region is required")
    selected=set(keep_paint_indices) if keep_paint_indices is not None else None
    def keep(event):
        chosen=event["index"] in selected if selected is not None else any(contains(r,event["bbox_pt"]) for r in keep_regions_pt)
        return chosen and not any(intersects(r,event["bbox_pt"]) for r in drop_regions_pt) and not any(n in event["form_path"] for n in drop_form_names)
    reader=_reader(reference);walker=_Walker(reader,page_number,keep)
    operations=walker.walk(walker.page["/Contents"],walker.page["/Resources"])
    if selected is not None and selected-set(e["index"] for e in walker.events):
        raise RuntimeError("Path allowlist contains indices missing from the locked source")
    if not walker.kept:raise RuntimeError("Static extraction selected no painted paths")
    writer=PdfWriter();page=writer.add_blank_page(float(walker.page.mediabox.width),float(walker.page.mediabox.height))
    stream=ContentStream(None,reader);stream.operations=operations
    clean=DecodedStreamObject();clean.set_data(stream.get_data())
    page[NameObject("/Contents")]=writer._add_object(clean)
    # Clone a tiny explicit set of path resources, rather than any source page.
    page[NameObject("/Resources")]=walker.resources.clone(writer)
    if "/Group" in walker.page:
        group=walker.page["/Group"]
        page[NameObject("/Group")]=group.clone(writer)
    writer.add_metadata({"/Title":"Reviewed manual vector artwork", "/Producer":"spatial_axes_manuscript static_art"})
    output_pdf.parent.mkdir(parents=True,exist_ok=True)
    with output_pdf.open("wb") as handle:writer.write(handle)
    result={"source":str(reference),"source_sha256":actual,"output":str(output_pdf),
            "output_sha256":sha256(output_pdf),"path_events_total":len(walker.events),
            "kept_paint_indices":walker.kept,"image_paints_removed":walker.images_removed,
            "text_paints_removed":walker.text_paint_removed,"keep_regions_pt":list(keep_regions_pt),
            "drop_regions_pt":list(drop_regions_pt),"no_reference_background":True,
            "private_data_retained":False,"fonts_retained":False}
    return result


def inspect_text(reference: Path | str, page_number=0) -> list[dict[str,Any]]:
    """Decode individual text paint events at their actual PDF baselines.

    We deliberately do not use visitor_text: it merges adjacent unrelated runs
    and emits extra nested-Form callbacks with incorrect inherited transforms.
    """
    from pypdf._cmap import get_encoding
    reader=_reader(reference);page=reader.pages[page_number];height=float(page.mediabox.height)
    collected=[]
    def visit(stream,resources,ctm=IDENTITY,depth=0):
        if depth>12:raise RuntimeError("Excessive text Form nesting")
        state={"ctm":ctm,"font":None,"size":1.,"tc":0.,"tw":0.,"tz":1.,
               "leading":0.,"rise":0.,"mode":0,"color":[.14,.12,.13]}
        stack=[];tm=IDENTITY;lm=IDENTITY
        def translated(matrix,dx,dy):return multiply((1.,0.,0.,1.,dx,dy),matrix)
        def show(parts):
            nonlocal tm
            font=state["font"]
            if font is None:raise RuntimeError("Text paint has no current font")
            encoding,mapping=get_encoding(font)
            widths=font.get("/Widths",[]);first=int(font.get("/FirstChar",0))
            advance=0.;strings=[]
            for part in parts:
                if isinstance(part,(int,float,FloatObject)):
                    advance-=float(part)/1000*state["size"]*state["tz"];continue
                try:raw=part.original_bytes
                except AttributeError:raw=bytes(part) if isinstance(part,bytes) else str(part).encode("latin1")
                decoded="".join(encoding.get(c,chr(c)) for c in raw) if isinstance(encoding,dict) else raw.decode(encoding,errors="replace")
                decoded="".join(mapping.get(c,c) for c in decoded)
                strings.append(decoded)
                for code in raw:
                    width=float(widths[code-first]) if 0<=code-first<len(widths) else 500.
                    advance+=(width/1000*state["size"]+state["tc"]+(state["tw"] if code==32 else 0))*state["tz"]
            text="".join(strings)
            text=re.sub(r"/uni([0-9A-Fa-f]{4})",lambda match:chr(int(match.group(1),16)),text)
            text=text.replace("/f_i","fi").replace("/f_l","fl")
            matrix=multiply(tm,state["ctm"]);x,y=point(matrix,0,state["rise"])
            size=state["size"]*math.hypot(matrix[2],matrix[3])
            if text.strip():
                collected.append({"text":text,"x_pt":x,"y_pt":height-y,"font_size_pt":size,
                                  "source_font":str(font.get("/BaseFont","")),"rotation_degrees":math.degrees(math.atan2(matrix[1],matrix[0])),
                                  "matrix":list(matrix),"text_rendering_mode":state["mode"],"color":state["color"][:],
                                  "advance_pt":abs(advance)*math.hypot(matrix[0],matrix[1])})
            tm=translated(tm,advance,0)
        for a,op in ContentStream(stream,reader).operations:
            if op==b"q":stack.append(copy.copy(state))
            elif op==b"Q":state=stack.pop()
            elif op==b"cm":state["ctm"]=multiply(tuple(map(float,a)),state["ctm"])
            elif op==b"BT":tm=IDENTITY;lm=IDENTITY
            elif op==b"Tf":state["font"]=resources["/Font"][a[0]].get_object();state["size"]=float(a[1])
            elif op==b"Tc":state["tc"]=float(a[0])
            elif op==b"Tw":state["tw"]=float(a[0])
            elif op==b"Tz":state["tz"]=float(a[0])/100
            elif op==b"TL":state["leading"]=float(a[0])
            elif op==b"Ts":state["rise"]=float(a[0])
            elif op==b"Tr":state["mode"]=int(a[0])
            elif op==b"Tm":tm=tuple(map(float,a));lm=tm
            elif op in (b"Td",b"TD"):
                dx,dy=map(float,a)
                if op==b"TD":state["leading"]=-dy
                lm=translated(lm,dx,dy);tm=lm
            elif op==b"T*":lm=translated(lm,0,-state["leading"]);tm=lm
            elif op in (b"Tj",b"TJ"):show(a if op==b"Tj" else a[0])
            elif op in (b"'",b'"'):
                if op==b'"':state["tw"]=float(a[0]);state["tc"]=float(a[1])
                lm=translated(lm,0,-state["leading"]);tm=lm;show([a[-1]])
            elif op==b"rg":state["color"]=list(map(float,a))
            elif op==b"g":state["color"]=[float(a[0])]*3
            elif op==b"k":
                c,m,y,k=map(float,a);state["color"]=[1-min(1,v+k) for v in (c,m,y)]
            elif op==b"Do":
                obj=resources["/XObject"][a[0]].get_object()
                if obj.get("/Subtype")=="/Form":visit(obj,obj.get("/Resources",resources),multiply(tuple(map(float,obj.get("/Matrix",IDENTITY))),state["ctm"]),depth+1)
    visit(page["/Contents"],page["/Resources"])
    unique={}
    for item in collected:
        key=(item["text"],round(item["x_pt"],4),round(item["y_pt"],4),round(item["font_size_pt"],4))
        if key in unique:
            unique[key]["paint_count"]+=1
            unique[key]["text_rendering_mode"]=max(unique[key]["text_rendering_mode"],item["text_rendering_mode"])
        else:unique[key]={**item,"paint_count":1}
    return list(unique.values())


def inspect_image_events(reference: Path | str, page_number=0):
    """List image invocation transforms for registering fresh data components."""
    reader=_reader(reference);page=reader.pages[page_number];height=float(page.mediabox.height)
    result=[]
    def visit(stream,resources,ctm=IDENTITY,context="page",depth=0,clip_rect=None):
        if depth>12:raise RuntimeError("Excessive Form nesting")
        stack=[];pending=[];clip_pending=False
        if clip_rect is None:clip_rect=[0.,0.,float(page.mediabox.width),height]
        for index,(operands,op) in enumerate(ContentStream(stream,reader).operations):
            if op==b"q":stack.append((ctm,clip_rect[:]))
            elif op==b"Q":ctm,clip_rect=stack.pop()
            elif op==b"cm":ctm=multiply(tuple(map(float,operands)),ctm)
            elif op in PATH:
                v=list(map(float,operands))
                if op==b"re":
                    x,y,w,h=v;pending.extend(point(ctm,a,b) for a,b in ((x,y),(x+w,y),(x+w,y+h),(x,y+h)))
                elif op!=b"h":pending.extend(point(ctm,v[j],v[j+1]) for j in range(0,len(v),2))
            elif op in (b"W",b"W*"):clip_pending=True
            elif op in PAINT or op==b"n":
                if clip_pending and pending:
                    xs,ys=zip(*pending);rect=[min(xs),height-max(ys),max(xs)-min(xs),max(ys)-min(ys)]
                    left=max(rect[0],clip_rect[0]);top=max(rect[1],clip_rect[1])
                    right=min(rect[0]+rect[2],clip_rect[0]+clip_rect[2]);bottom=min(rect[1]+rect[3],clip_rect[1]+clip_rect[3])
                    clip_rect=[left,top,max(0,right-left),max(0,bottom-top)]
                pending=[];clip_pending=False
            elif op==b"Do":
                key=operands[0];ref=resources["/XObject"].raw_get(key);obj=ref.get_object()
                if obj.get("/Subtype")=="/Form":
                    visit(obj,obj.get("/Resources",resources),multiply(tuple(map(float,obj.get("/Matrix",IDENTITY))),ctm),f"{context}/{key}@{index}",depth+1,clip_rect[:])
                elif obj.get("/Subtype")=="/Image":
                    corners=[point(ctm,x,y) for x,y in ((0,0),(1,0),(1,1),(0,1))]
                    xs,ys=zip(*corners)
                    raw=[min(xs),height-max(ys),max(xs)-min(xs),max(ys)-min(ys)]
                    left=max(raw[0],clip_rect[0]);top=max(raw[1],clip_rect[1]);right=min(raw[0]+raw[2],clip_rect[0]+clip_rect[2]);bottom=min(raw[1]+raw[3],clip_rect[1]+clip_rect[3])
                    result.append({"index":len(result),"name":str(key),"object_id":ref.idnum,
                                   "form_path":context,"operation_index":index,
                                   "rect_pt":raw,"clip_rect_pt":clip_rect[:],
                                   "visible_rect_pt":[left,top,max(0,right-left),max(0,bottom-top)],
                                   "matrix":list(ctm),"width_px":int(obj["/Width"]),"height_px":int(obj["/Height"])})
    visit(page["/Contents"],page["/Resources"])
    return result
