import numpy as np
import torch
import torch.nn.functional as F

from crypt_villus_vit.crops import crop_normalize_resize
from crypt_villus_vit.crops import crop_resize
from crypt_villus_vit.crops import native_crop_size_px


def _reference_crop_resize(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    crop_size_px: int,
    output_size_px: int,
) -> np.ndarray:
    crop_size = int(crop_size_px)
    output_size = int(output_size_px)
    half = crop_size / 2.0
    x0 = int(np.floor(float(center_x) - half))
    y0 = int(np.floor(float(center_y) - half))
    x1 = x0 + crop_size
    y1 = y0 + crop_size
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - image.shape[1])
    pad_bottom = max(0, y1 - image.shape[0])
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )
    crop = padded[y0 + pad_top : y1 + pad_top, x0 + pad_left : x1 + pad_left]
    tensor = torch.from_numpy(crop.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(output_size, output_size), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy().astype(np.float32)


def test_crop_resize_matches_reference_at_image_edges():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    cases = [
        (5, 5, 4, 4),
        (1, 1, 6, 4),
        (9, 9, 6, 4),
        (-1, 5, 6, 4),
        (5, 11, 6, 4),
    ]
    for center_x, center_y, crop_size, output_size in cases:
        expected = _reference_crop_resize(
            image,
            center_x=center_x,
            center_y=center_y,
            crop_size_px=crop_size,
            output_size_px=output_size,
        )
        observed = crop_resize(
            image,
            center_x=center_x,
            center_y=center_y,
            crop_size_px=crop_size,
            output_size_px=output_size,
        )
        np.testing.assert_allclose(observed, expected)


def test_native_crop_size_preserves_physical_width():
    assert native_crop_size_px(
        512,
        reference_pixel_size_um=0.325,
        source_pixel_size_um=0.2125,
    ) == 783
    assert native_crop_size_px(
        2048,
        reference_pixel_size_um=0.325,
        source_pixel_size_um=0.2125,
    ) == 3132
    assert native_crop_size_px(
        128,
        reference_pixel_size_um=0.325,
        source_pixel_size_um=0.2125,
    ) == 196


def test_crop_normalize_resize_normalizes_before_resizing():
    image = np.zeros((12, 12), dtype=np.float32)
    image[3:9, 3:9] = np.arange(36, dtype=np.float32).reshape(6, 6) + 100
    observed = crop_normalize_resize(
        image,
        center_x=6,
        center_y=6,
        crop_size_px=6,
        output_size_px=6,
    )
    expected_crop = image[3:9, 3:9]
    low, high = np.percentile(expected_crop, [1.0, 99.8])
    expected = np.clip((expected_crop - low) / (high - low), 0.0, 1.0)
    np.testing.assert_allclose(observed, expected, atol=1e-6)
