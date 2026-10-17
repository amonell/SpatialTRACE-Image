"""Match the released supervised uint8 crop contract after direct resizing.

The source backend chooses the pyramid for the final 256/128 input, masks
excluded tissue, normalizes that crop, and resizes. Coordinate SFT quantized
this result to uint8. Its pretraining-only 512-intermediate recipe is distinct.
"""
from __future__ import annotations

import numpy as np

PROTOCOL = 'direct_pyramid_crop_normalize_resize_uint8_v1'


def quantize_sft_input(normalized: np.ndarray) -> np.ndarray:
    """Return the exact stored SFT bytes; no image resizing occurs here."""
    array=np.asarray(normalized,dtype=np.float32)
    if array.ndim!=3 or array.shape[0]!=1 or array.shape[1:] not in ((256,256),(128,128)):
        raise ValueError(f'Unexpected supervised input shape: {array.shape}')
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError('Supervised normalized crop must be finite and within [0,1]')
    return np.rint(np.clip(array,0,1)*255).astype(np.uint8)


def dequantize_sft_input(array: np.ndarray) -> np.ndarray:
    if np.asarray(array).dtype!=np.uint8:
        raise ValueError('Supervised stored crop must be uint8')
    return np.asarray(array,dtype=np.float32)/255.0
