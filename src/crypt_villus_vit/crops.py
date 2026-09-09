from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from crypt_villus_vit.sources import SourceSpec
from crypt_villus_vit.sources import load_grayscale_image


@lru_cache(maxsize=2)
def _cached_image(path: str, channel_index: int) -> np.ndarray:
    return load_grayscale_image(Path(path), channel_index=int(channel_index), normalize=False)


def native_crop_size_px(
    reference_crop_px: int,
    *,
    reference_pixel_size_um: float,
    source_pixel_size_um: float,
) -> int:
    if int(reference_crop_px) <= 0:
        raise ValueError("Reference crop size must be positive.")
    if float(reference_pixel_size_um) <= 0 or float(source_pixel_size_um) <= 0:
        raise ValueError("Pixel sizes must be positive.")
    return max(
        1,
        int(round(float(reference_crop_px) * float(reference_pixel_size_um) / float(source_pixel_size_um))),
    )


def normalize_grayscale_crop(
    image: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
    percentile_max_pixels: int = 262_144,
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    max_pixels = int(percentile_max_pixels)
    percentile_values = values
    if max_pixels > 0 and values.size > max_pixels:
        step = max(1, int(np.ceil(np.sqrt(values.size / max_pixels))))
        percentile_values = values[::step, ::step]
    finite = percentile_values[np.isfinite(percentile_values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(finite, [float(low_percentile), float(high_percentile)])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        high = low + 1.0
    normalized = (values - float(low)) / max(float(high - low), 1e-6)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def extract_crop(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    crop_size_px: int,
) -> np.ndarray:
    crop_size = int(crop_size_px)
    if crop_size <= 0:
        raise ValueError("Crop size must be positive.")
    half = crop_size / 2.0
    x0 = int(np.floor(float(center_x) - half))
    y0 = int(np.floor(float(center_y) - half))
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    image_height, image_width = image.shape[:2]
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image_width, x1)
    src_y1 = min(image_height, y1)
    crop = image[src_y0:src_y1, src_x0:src_x1]
    pad_left = src_x0 - x0
    pad_top = src_y0 - y0
    pad_right = x1 - src_x1
    pad_bottom = y1 - src_y1
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
    return np.asarray(crop, dtype=np.float32)


def crop_resize(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    crop_size_px: int,
    output_size_px: int,
) -> np.ndarray:
    output_size = int(output_size_px)
    if output_size <= 0:
        raise ValueError("Output size must be positive.")
    crop = extract_crop(
        image,
        center_x=center_x,
        center_y=center_y,
        crop_size_px=crop_size_px,
    )
    tensor = torch.from_numpy(crop.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy().astype(np.float32)


def crop_normalize_resize(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    crop_size_px: int,
    output_size_px: int,
) -> np.ndarray:
    crop = extract_crop(
        image,
        center_x=center_x,
        center_y=center_y,
        crop_size_px=crop_size_px,
    )
    normalized = normalize_grayscale_crop(crop)
    tensor = torch.from_numpy(normalized)[None, None]
    resized = F.interpolate(
        tensor,
        size=(int(output_size_px), int(output_size_px)),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy().astype(np.float32)


class CropExtractor:
    def __init__(self, sources: dict[str, SourceSpec]):
        self.sources = dict(sources)

    def image_for_source(self, source_id: str) -> np.ndarray:
        source = self.sources[str(source_id)]
        return _cached_image(str(source.image_path), int(source.channel_index))

    def crop_for_source(
        self,
        source_id: str,
        *,
        center_x: float,
        center_y: float,
        crop_size_px: int,
        output_size_px: int,
    ) -> np.ndarray:
        image = self.image_for_source(source_id)
        return crop_normalize_resize(
            image,
            center_x=center_x,
            center_y=center_y,
            crop_size_px=crop_size_px,
            output_size_px=output_size_px,
        )
