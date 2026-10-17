#!/usr/bin/env python3
"""Extract the prespecified RARi IF crop directly from the raw VSI file.

Run this script with the environment that provides ``bffile``. It writes a
display-ready RGB array but does not assemble or plot Figure 1 artwork.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from paper_paths import Path, lock_record, rendering_script

from bffile import BioFile
import numpy as np

COLORS = ((0.20, 0.34, 1.00), (0.00, 0.85, 0.42), (0.78, 0.24, 1.00), (1.00, 0.48, 0.10))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(image, dtype=np.float32)
    nonzero = values[values > 0]
    source = nonzero if nonzero.size >= 100 else values.reshape(-1)
    low, high = (float(v) for v in np.percentile(source, [0.5, 99.85]))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0, 1)
    scaled[scaled < 0.015] = 0
    return np.power(scaled, 0.78), {"display_low": low, "display_high": high}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--series", type=int, default=0)
    parser.add_argument("--x-min", type=int, required=True)
    parser.add_argument("--x-max", type=int, required=True)
    parser.add_argument("--y-min", type=int, required=True)
    parser.add_argument("--y-max", type=int, required=True)
    parser.add_argument("--pixel-size-um", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    image_path = args.image.expanduser().resolve()
    companion_path = args.companion.expanduser().resolve()
    output = args.output_npy.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not companion_path.is_file():
        raise FileNotFoundError(companion_path)
    if (output.exists() or metadata.exists()) and not args.overwrite:
        raise FileExistsError("IF crop output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)

    crops = []
    summaries = []
    with BioFile(image_path, meta=False) as bf:
        core = bf.core_metadata(args.series)
        names = list(core.series_metadata.get("Channel name", []))
        channel_count = int(core.shape.c)
        if len(names) != channel_count:
            names = [f"Channel {index}" for index in range(channel_count)]
        lazy = bf.as_array(series=args.series, resolution=0)
        for channel in range(channel_count):
            raw = np.asarray(lazy[0, channel, 0, args.y_min:args.y_max, args.x_min:args.x_max])
            scaled, summary = normalize(raw)
            crops.append(scaled)
            summaries.append({"channel": channel, "name": names[channel], **summary})
    rgb = np.zeros((*crops[0].shape, 3), dtype=np.float32)
    for channel, crop in enumerate(crops):
        color = np.asarray(COLORS[channel % len(COLORS)], dtype=np.float32)
        layer = crop[:, :, None] * color
        rgb = 1 - (1 - rgb) * (1 - layer)
        summaries[channel]["rgb"] = color.tolist()
    rgb_u8 = np.asarray(np.clip(rgb, 0, 1) * 255 + 0.5, dtype=np.uint8)
    np.save(output, rgb_u8)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "direct multichannel crop extraction from raw VSI",
        "image_path": str(image_path),
        "image_sha256": sha256(image_path),
        "image_bytes": image_path.stat().st_size,
        "image_mtime_ns": image_path.stat().st_mtime_ns,
        "companion_path": str(companion_path),
        "companion_sha256": sha256(companion_path),
        "companion_bytes": companion_path.stat().st_size,
        "companion_mtime_ns": companion_path.stat().st_mtime_ns,
        "series": args.series,
        "bounds_fullres_px": {"x_min": args.x_min, "x_max": args.x_max, "y_min": args.y_min, "y_max": args.y_max},
        "pixel_size_um": args.pixel_size_um,
        "channels": summaries,
        "output": {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size},
    }
    metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
