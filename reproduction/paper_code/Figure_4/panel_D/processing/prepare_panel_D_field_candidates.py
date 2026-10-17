#!/usr/bin/env python3
"""Prepare microscopy for an author-requested qualitative field review.

This is not an accuracy evaluation. It neither runs inference nor changes gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import numpy as np

LOCK_SHA = "f3a67a9481d9969126e2df03d2e7e598158bcd286e92295d717779c7c8817d5b"


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def verify(rec):
    actual = record(rec["path"])
    if actual["sha256"] != rec["sha256"]:
        raise RuntimeError(f"Changed frozen input: {rec['path']}")
    return Path(rec["path"])


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def read_tiles(reader, channel, bounds):
    x0, x1, y0, y1 = bounds
    dtype = np.dtype("<u2" if reader.isLittleEndian() else ">u2")
    result = np.empty((y1-y0, x1-x0), dtype=np.uint16)
    plane = int(reader.getIndex(0, channel, 0))
    for y in range(y0, y1, 1024):
        for x in range(x0, x1, 1024):
            w, h = min(1024, x1-x), min(1024, y1-y)
            result[y-y0:y-y0+h, x-x0:x-x0+w] = np.frombuffer(
                bytes(reader.openBytes(plane, x, y, w, h)), dtype=dtype).reshape(h, w)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--windows", type=Path, help="Explicit native-pixel candidate bounds; omit for overview")
    p.add_argument("--reuse-manifest", type=Path, help="Reuse verified native crops only when identity and bounds match exactly")
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    verify({"path": str(a.source_manifest), "sha256": LOCK_SHA})
    lock = json.loads(a.source_manifest.read_text())
    inputs = [record(a.source_manifest)]
    for key in ("release", "channel_manifest", "channel_key", "gates", "checkpoint"):
        verify(lock[key]); inputs.append(lock[key])
    channels = json.loads(Path(lock["channel_manifest"]["path"]).read_text())
    for key in ("source", "source_tiles"):
        verify(channels[key]); inputs.append(channels[key])
    key = json.loads(Path(lock["channel_key"]["path"]).read_text())
    if key["status"] != "CONFIRMED" or key["E_cadherin_channel_index"] != 2:
        raise RuntimeError("Expected confirmed Cy5 E-cadherin channel")
    for window in lock["windows"].values():
        verify(window["source_predictions"]); inputs.append(window["source_predictions"])
    reusable = {}
    if a.reuse_manifest:
        parent = json.loads(a.reuse_manifest.read_text())
        verify(parent["source_manifest"])
        if parent["source_manifest"]["sha256"] != LOCK_SHA or parent["display"] != lock["display"]:
            raise RuntimeError("Reuse manifest belongs to another lineage")
        inputs.append(record(a.reuse_manifest))
        reusable = {(w["id"], w["condition"], tuple(w["bounds_xxyy_px"])): w for w in parent["windows"]}
    os.environ["BIOFORMATS_VERSION"] = "8.5.0"
    os.environ["BIOFORMATS_LOG_LEVEL"] = "ERROR"
    from bioformats_jar import get_loci
    loci = get_loci()
    reader = loci.formats.ImageReader()
    reader.setFlattenedResolutions(False)
    meta = loci.formats.MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(meta)
    a.output_dir.mkdir(parents=True)
    result = {"status": "QUALITATIVE_FIELD_REVIEW_NOT_SELECTED_FOR_PUBLICATION",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "command": [sys.executable, *sys.argv], "python": platform.python_version(),
              "packages": {n: importlib.metadata.version(n) for n in ("numpy", "bioformats-jar", "jpype1")},
              "git_state": "No Git repository", "random_seed": None,
              "code": record(__file__), "source_manifest": inputs[0], "inputs": inputs,
              "pixel_size_um": lock["pixel_size_um"], "display": lock["display"],
              "predictions": {c: w["source_predictions"] for c, w in lock["windows"].items()},
              "current_bounds": {c: w["bounds_xxyy_px"] for c, w in lock["windows"].items()},
              "gate_assignments_changed": False, "inference_run": False,
              "warnings": ["Candidates are reviewed using morphology and visible predictions, at the author's request.",
                           "No independent muscularis labels: visual agreement is not a measured accuracy estimate.",
                           "Original two IF specimens were used during model development; no cohort statistics change."]}
    try:
        reader.setId(channels["source"]["path"])
        reader.setSeries(0); reader.setResolution(0)
        shape = (int(reader.getSizeY()), int(reader.getSizeX()))
        result["native_shape_yx"] = list(shape)
        if reader.getPixelType() != 3 or str(meta.getChannelName(0, 2)) != "Cy5":
            raise RuntimeError("Unexpected microscopy type/channel")
        if a.windows:
            plan = json.loads(a.windows.read_text())
            inputs.append(record(a.windows))
            result["selection_plan"] = plan
            result["windows"] = []
            for window in plan["windows"]:
                ident = window["id"]
                if not ident.replace("_", "").isalnum():
                    raise ValueError("Invalid candidate ID")
                bounds = window["bounds_xxyy_px"]
                x0, x1, y0, y1 = bounds
                if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
                    raise ValueError(bounds)
                previous = reusable.get((ident, window["condition"], tuple(bounds)))
                if previous:
                    verify(previous["microscopy"])
                    result["windows"].append({**window, "microscopy": previous["microscopy"]})
                    print(f"Reuse exact native crop {ident}", flush=True)
                    continue
                arrays = {}
                for name, ch in (("dapi", 0), ("ecad", 2)):
                    print(f"Extract {ident}: {name} {bounds}", flush=True)
                    arrays[name] = read_tiles(reader, ch, bounds)
                path = a.output_dir / f"{ident}.npz"
                np.savez_compressed(path, **arrays)
                result["windows"].append({**window, "microscopy": record(path)})
        else:
            options = []
            for resolution in range(int(reader.getResolutionCount())):
                reader.setResolution(resolution)
                options.append((abs(shape[1]/int(reader.getSizeX()) - 16), resolution))
            reader.setResolution(min(options)[1])
            h, w = int(reader.getSizeY()), int(reader.getSizeX())
            print(f"Overview: {w} × {h}, resolution {reader.getResolution()}", flush=True)
            arrays = {name: read_tiles(reader, ch, [0, w, 0, h]) for name, ch in (("dapi", 0), ("ecad", 2))}
            path = a.output_dir / "overview.npz"
            np.savez_compressed(path, **arrays)
            result["overview"] = record(path)
            result["overview_resolution"] = int(reader.getResolution())
            result["overview_shape_yx"] = [h, w]
            result["overview_scale_xy"] = [shape[1]/w, shape[0]/h]
        # Exact registration check against the existing, immutable Figure 4 field.
        old = lock["windows"]["DMSO"]
        x0, _, y0, _ = old["bounds_xxyy_px"]
        reader.setResolution(0)
        for name, ch in (("DAPI", 0), ("E_cadherin", 2)):
            raw = np.load(verify(old[name]), mmap_mode="r")
            np.testing.assert_array_equal(raw[:256, :256], read_tiles(reader, ch, [x0,x0+256,y0,y0+256]))
        result["registration_check"] = "DAPI and Cy5 native 256×256 tiles exactly match existing frozen DMSO field"
    finally:
        reader.close()
    save_json(a.output_dir / "manifest.json", result)
    print(a.output_dir / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
