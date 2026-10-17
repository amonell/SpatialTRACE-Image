"""Measure image CTMs, path spines and text baselines in locked source PDFs."""
from paper_paths import Path, lock_record, rendering_script
from pypdf import PdfReader
from pypdf.generic import ContentStream
from figure_assembly.static_art import multiply, point, inspect_path_events, inspect_text, sha256


def measure(reference):
    reader = PdfReader(reference)
    page = reader.pages[0]
    height = float(page.mediabox.height)
    images = []
    def walk(stream, resources, ctm=(1,0,0,1,0,0), context="page"):
        stack = []
        for index, (values, op) in enumerate(ContentStream(stream, reader).operations):
            if op == b"q": stack.append(ctm)
            elif op == b"Q": ctm = stack.pop()
            elif op == b"cm": ctm = multiply(tuple(map(float,values)),ctm)
            elif op == b"Do":
                obj = resources["/XObject"][values[0]].get_object()
                path = context + "/" + str(values[0])
                if obj.get("/Subtype") == "/Image":
                    coords = [point(ctm,x,y) for x,y in [(0,0),(0,1),(1,0),(1,1)]]
                    xs,ys = zip(*coords)
                    images.append({"id":path,"rect_pt":[min(xs),height-max(ys),max(xs)-min(xs),max(ys)-min(ys)],
                                   "pixels":[int(obj["/Width"]),int(obj["/Height"])]})
                elif obj.get("/Subtype") == "/Form":
                    walk(obj,obj.get("/Resources",resources),multiply(tuple(map(float,obj.get("/Matrix",[1,0,0,1,0,0]))),ctm),path)
    walk(page["/Contents"],page["/Resources"])
    return {"reference_pdf":str(Path(reference).resolve()),"reference_sha256":sha256(reference),
            "method":"Original image unit-square transformed by recursive PDF CTMs; original path paint bounds; direct text-paint baselines.",
            "images":images,"paths":inspect_path_events(reference),"texts":inspect_text(reference)}
