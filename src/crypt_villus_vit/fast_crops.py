"""Bounded decoded-tile reuse for raw-image inference.

No prepared crops, downsampled substitutes, or changes to crop geometry. The
reference TIFF reader still handles channels, Z projection and pyramid levels.
"""
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as F

from .production_crops import ImagePyramid, ProductionCropExtractor


def rust_normalizer():
    try:
        from spatialtrace_image_rust import normalize
    except ImportError as exc:
        raise RuntimeError(
            "Rust preprocessing is not installed. Run uv sync --extra rust "
            "with your cpu or cu126 extra, or use --crop-backend cached."
        ) from exc
    return normalize


class CachedImagePyramid(ImagePyramid):
    def __init__(self, source, *, cache_bytes, backend="cached"):
        super().__init__(source)
        self.cache_bytes = int(cache_bytes)
        if self.cache_bytes < 0:
            self.close()
            raise ValueError("Tile cache size must be nonnegative")
        self.tiles = OrderedDict()
        self.cached_bytes = 0
        self.hits = 0
        self.misses = 0
        self.normalizer = rust_normalizer() if backend == "rust" else None
        self.tile_shapes = {}
        if self.tiff is not None:
            for i, level in enumerate(self.tiff.series[0].levels):
                page = level.pages[0]
                self.tile_shapes[i] = (
                    int(page.tilelength or 512), int(page.tilewidth or 512)
                )
        # Per-instance cache; does not hold previously closed slides alive.
        self.choose_level = lru_cache(maxsize=16)(self.choose_level)

    def read(self, level, top, bottom, left, right):
        if self.tiff is None or self.cache_bytes == 0:
            return super().read(level, top, bottom, left, right)
        th, tw = self.tile_shapes[level]
        height, width = self.level_shapes[level]
        result = None
        for y in range(top // th * th, bottom, th):
            for x in range(left // tw * tw, right, tw):
                key = (level, y, x)
                tile = self.tiles.get(key)
                if tile is None:
                    self.misses += 1
                    tile = super().read(level, y, min(y + th, height), x, min(x + tw, width))
                    # Own the backing storage so the memory bound is meaningful.
                    tile = np.array(tile, copy=True, order="C")
                    if tile.nbytes <= self.cache_bytes:
                        while self.cached_bytes + tile.nbytes > self.cache_bytes:
                            _, old = self.tiles.popitem(last=False)
                            self.cached_bytes -= old.nbytes
                        self.tiles[key] = tile
                        self.cached_bytes += tile.nbytes
                else:
                    self.hits += 1
                    self.tiles.move_to_end(key)
                if result is None:
                    result = np.empty((bottom - top, right - left), dtype=tile.dtype)
                a, b = max(top, y), min(bottom, y + tile.shape[0])
                c, d = max(left, x), min(right, x + tile.shape[1])
                result[a - top:b - top, c - left:d - left] = tile[a - y:b - y, c - x:d - x]
        return result

    def normalize_crop(self, crop, output):
        if self.normalizer is None:
            return super().normalize_crop(crop, output)
        values = self.normalizer(np.ascontiguousarray(crop, dtype=np.float32))
        # Retain the exact PyTorch resize and NumPy quantization used in training.
        resized = F.interpolate(torch.from_numpy(values)[None, None],
                                size=(output, output), mode="bilinear", align_corners=False)
        u8 = np.rint(np.clip(resized[0, 0].numpy(), 0., 1.) * 255.).astype(np.uint8)
        return u8.astype(np.float32) / 255.

    def close(self):
        if hasattr(self, "tiles"):
            self.tiles.clear()
            self.cached_bytes = 0
        if hasattr(self.choose_level, "cache_clear"):
            self.choose_level.cache_clear()
        super().close()


class CachedCropExtractor(ProductionCropExtractor):
    def __init__(self, sources, *, backend, cache_bytes):
        super().__init__(sources)
        self.backend = backend
        self.cache_bytes = cache_bytes
        if backend == "rust":
            rust_normalizer()  # Fail before starting worker processes.

    def crop_for_source(self, source_id, *, center_x, center_y, crop_size_px, output_size_px):
        if source_id not in self.backends:
            for backend in self.backends.values():
                backend.close()
            self.backends = {source_id: CachedImagePyramid(
                self.sources[source_id], cache_bytes=self.cache_bytes, backend=self.backend
            )}
        return self.backends[source_id].crop(center_x, center_y, crop_size_px, output_size_px)
