"""Production-supervised image preparation without workstation dependencies.

Preserves source pyramids, round-centered crops, per-crop percentile bounds,
direct bilinear resize and the original uint8 roundtrip. TIFF reads are bounded
to the requested region using tifffile/Zarr rather than loading a whole slide.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
import torch
from torch.nn import functional as F

PROTOCOL = "direct_pyramid_crop_normalize_resize_uint8_v1"


def crop_size(native, full, level):
    return max(4, int(round(float(native) * float(level) / float(full))))


def normalize_resize_quantize(crop, output_size):
    values = np.asarray(crop, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not len(finite):
        low, high = 0., 1.
    else:
        low, high = np.percentile(finite, [1., 99.8])
        if not np.isfinite(low + high) or high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high <= low:
            high = low + 1.
    # The training pipeline explicitly stored percentile bounds in float32.
    low, high = np.asarray([low, high], dtype=np.float32)
    values = (values - low) / np.maximum(high - low, 1e-6)
    values = np.clip(np.nan_to_num(values, nan=0., posinf=1., neginf=0.), 0., 1.)
    tensor = torch.from_numpy(values)[None, None]
    resized = F.interpolate(tensor, size=(output_size, output_size), mode="bilinear", align_corners=False)
    u8 = np.rint(np.clip(resized[0, 0].numpy(), 0., 1.) * 255.).astype(np.uint8)
    return u8.astype(np.float32) / 255.


class ImagePyramid:
    def __init__(self, source):
        self.source = source
        self.tiff = None
        self.stores = {}
        self.arrays = {}
        path = Path(source.image_path)
        if path.suffix.lower() in {".tif", ".tiff"}:
            self.tiff = tifffile.TiffFile(path)
            levels = self.tiff.series[0].levels
            self.axes = [level.axes for level in levels]
            self.shapes = [level.shape for level in levels]
            for axes, shape in zip(self.axes, self.shapes):
                if 'X' not in axes or 'Y' not in axes:
                    raise ValueError("TIFF must declare spatial Y/X axes")
                for axis, count in zip(axes, shape):
                    if axis not in "YXZCS" and count > 1:
                        raise ValueError(f"Ambiguous TIFF axis {axis}; export one timepoint/scene as OME-TIFF")
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False) if path.suffix.lower() == ".npy" else np.asarray(Image.open(path))
            if array.ndim == 3 and array.shape[-1] in (3, 4):
                array = array[..., source.channel_index]
            if array.ndim != 2:
                raise ValueError("NumPy inputs must be 2D; use OME-TIFF with named axes for multichannel data")
            self.arrays[0] = array
            self.axes = ['YX']
            self.shapes = [array.shape]
        self.level_shapes = [(s[a.index('Y')], s[a.index('X')]) for a, s in zip(self.axes, self.shapes)]
        self.full_shape = self.level_shapes[0]

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()
        if self.tiff is not None:
            self.tiff.close()

    def choose_level(self, native, output):
        sizes = [(i, crop_size(native, self.full_shape[0], shape[0])) for i, shape in enumerate(self.level_shapes)]
        eligible = [item for item in sizes if item[1] >= output]
        return min(eligible, key=lambda item: (item[1] - output, -item[0]))[0] if eligible else min(sizes, key=lambda item: (abs(item[1] - output), item[0]))[0]

    def read(self, level, top, bottom, left, right):
        if level not in self.arrays:
            import zarr
            # Select a level explicitly: series[0].levels[0].aszarr() otherwise
            # exposes the entire pyramid as a Zarr Group, not the base array.
            store = self.tiff.aszarr(series=0, level=level)
            self.stores[level] = store
            self.arrays[level] = zarr.open(store, mode='r')
        axes = self.axes[level]
        slices, retained = [], []
        for axis, count in zip(axes, self.shapes[level]):
            if axis == 'Y':
                slices.append(slice(top, bottom)); retained.append(axis)
            elif axis == 'X':
                slices.append(slice(left, right)); retained.append(axis)
            elif axis == 'Z':
                slices.append(slice(None)); retained.append(axis)
            elif axis in 'CS':
                if not 0 <= self.source.channel_index < count:
                    raise ValueError("channel_index is outside TIFF channel range")
                slices.append(self.source.channel_index)
            else:
                slices.append(0)
        array = np.asarray(self.arrays[level][tuple(slices)])
        if 'Z' in retained:
            array = array.max(axis=retained.index('Z'))
            retained.remove('Z')
        return array if retained == ['Y', 'X'] else array.T

    def crop(self, center_x, center_y, native, output):
        if native <= 0 or output <= 0:
            raise ValueError("Crop dimensions must be positive")
        if not (np.isfinite(center_x + center_y) and 0 <= center_x < self.full_shape[1] and 0 <= center_y < self.full_shape[0]):
            raise ValueError("Cell centroid is outside the full-resolution image")
        level = self.choose_level(native, output)
        height, width = self.level_shapes[level]
        ch = crop_size(native, self.full_shape[0], height)
        cw = crop_size(native, self.full_shape[1], width)
        top = round(center_y * height / self.full_shape[0] - ch / 2)
        left = round(center_x * width / self.full_shape[1] - cw / 2)
        bottom, right = top + ch, left + cw
        a, b, c, d = max(top, 0), min(bottom, height), max(left, 0), min(right, width)
        # All padding is applied before percentile normalization, as in training.
        result = np.zeros((ch, cw), dtype=np.float32)
        if b > a and d > c:
            result[a - top:b - top, c - left:d - left] = self.read(level, a, b, c, d)
        return normalize_resize_quantize(result, output)


class ProductionCropExtractor:
    def __init__(self, sources):
        self.sources = dict(sources)
        self.backends = {}

    def crop_for_source(self, source_id, *, center_x, center_y, crop_size_px, output_size_px):
        if source_id not in self.backends:
            # Source-grouped batching allows one slide to be open at a time.
            for backend in self.backends.values():
                backend.close()
            self.backends = {source_id: ImagePyramid(self.sources[source_id])}
        return self.backends[source_id].crop(center_x, center_y, crop_size_px, output_size_px)
