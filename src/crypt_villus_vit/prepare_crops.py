"""Prepare reusable uint8 shards without changing crop values or row identities."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import ModelConfig
from .predict import CellCropDataset, SourceGroupedBatchSampler
from .sources import load_source_manifest

INPUT_PROTOCOL = "direct_pyramid_crop_normalize_resize_uint8_v1"


class Uint8CropDataset(Dataset):
    """Quantize before interprocess transfer, using the original preparation rule."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return {key: (np.rint(value * 255).astype(np.uint8) if key != "index" else value)
                for key, value in item.items()}


class ShardWriter:
    """Write by original row index; keep only a few shard mappings open at once.

    Spatially ordered reads need not change the on-disk row order. Partial files
    are never listed in metadata. A failed run must use a new output directory.
    """

    def __init__(self, root, prefix, row_count, rows_per_shard, image_shape, max_open=4):
        self.root = Path(root)
        self.row_count = row_count
        self.rows_per_shard = rows_per_shard
        self.image_shape = image_shape
        self.max_open = max_open
        self.seen = np.zeros(row_count, dtype=bool)
        self.maps = OrderedDict()
        self.created = set()
        self.paths = [self.root / f"{prefix}{i:05d}.npy" for i in
                      range((row_count + rows_per_shard - 1) // rows_per_shard)]

    def _mapping(self, shard):
        if shard in self.maps:
            self.maps.move_to_end(shard)
            return self.maps[shard]
        if len(self.maps) >= self.max_open:
            _, old = self.maps.popitem(last=False)
            old._mmap.close()
        path = self.paths[shard].with_suffix(".npy.partial")
        count = min(self.rows_per_shard, self.row_count - shard * self.rows_per_shard)
        mode = "r+" if shard in self.created else "w+"
        array = np.lib.format.open_memmap(path, mode=mode, dtype=np.uint8,
                                         shape=(count, *self.image_shape))
        self.created.add(shard)
        self.maps[shard] = array
        return array

    def write(self, indices, images):
        indices = np.asarray(indices, dtype=np.int64)
        if images.dtype != np.uint8 or images.shape != (len(indices), *self.image_shape):
            raise ValueError("Unexpected prepared crop dtype or shape")
        if (indices < 0).any() or (indices >= self.row_count).any():
            raise IndexError("Prepared index outside shard coverage")
        if len(np.unique(indices)) != len(indices) or self.seen[indices].any():
            raise ValueError("Duplicate prepared indices")
        shard_ids = indices // self.rows_per_shard
        for shard in np.unique(shard_ids):
            selection = shard_ids == shard
            self._mapping(int(shard))[indices[selection] % self.rows_per_shard] = images[selection]
        self.seen[indices] = True

    def close(self):
        for array in self.maps.values():
            array._mmap.close()
        self.maps.clear()

    def finish(self):
        if not self.seen.all():
            raise ValueError("Cannot finalize incomplete crop shards")
        self.close()
        for path in self.paths:
            partial = path.with_suffix(".npy.partial")
            # Include completed disk writes in preparation time, not just dirty pages.
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            partial.replace(path)
        return [path.name for path in self.paths]


def prepare_crops(args, *, supervised=False):
    started = time.perf_counter()
    dimensions = [args.local_crop_px, args.context_crop_px, args.input_size_px]
    if supervised:
        dimensions += [args.fine_crop_px, args.fine_input_size_px]
    if min(dimensions) <= 0 or args.rows_per_shard <= 0 or args.batch_size <= 0:
        raise ValueError("Crop dimensions, rows-per-shard, and batch-size must be positive")
    if args.num_workers < 0 or args.tile_cache_mib < 0:
        raise ValueError("num-workers and tile-cache-mib must be nonnegative")
    if not np.isfinite(args.reference_pixel_size_um) or args.reference_pixel_size_um <= 0:
        raise ValueError("reference-pixel-size-um must be finite and positive")
    sources = load_source_manifest(args.source_manifest)
    rows = pd.read_csv(args.cells_csv)
    required = {"source_id", "centroid_x_fullres_px", "centroid_y_fullres_px"}
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"Cell CSV is missing required column(s): {missing}")
    if rows.empty:
        raise ValueError("Cell CSV has no rows")
    if rows.source_id.isna().any():
        raise ValueError("source_id cannot be missing")
    rows["source_id"] = rows.source_id.astype(str)
    if unknown := sorted(set(rows.source_id) - set(sources)):
        raise ValueError(f"Unknown source_id(s): {unknown}")
    if not np.isfinite(rows[["centroid_x_fullres_px", "centroid_y_fullres_px"]].to_numpy(float)).all():
        raise ValueError("Cell centroids must be finite")
    for source_id in rows.source_id.unique():
        size = sources[source_id].pixel_size_um
        if size is None or not np.isfinite(size) or size <= 0:
            raise ValueError(f"Source {source_id} requires finite positive pixel_size_um")
    if not supervised and "split" in rows and rows["split"].astype(str).str.strip().str.lower().eq("test").any():
        raise ValueError("Pretraining input includes test rows. Supply only prespecified pretraining sections.")
    config = ModelConfig(local_crop_px=args.local_crop_px, context_crop_px=args.context_crop_px,
                         input_size_px=args.input_size_px, use_fine_branch=supervised and args.fine_branch,
                         fine_crop_px=args.fine_crop_px if supervised else 128,
                         fine_input_size_px=args.fine_input_size_px if supervised else 128)
    dataset = CellCropDataset(rows, sources, config, reference_pixel_size_um=args.reference_pixel_size_um,
                             input_protocol=INPUT_PROTOCOL, crop_backend=args.crop_backend,
                             tile_cache_mib=args.tile_cache_mib)
    spatial_order = args.spatial_order if args.spatial_order is not None else args.crop_backend != "reference"
    loader_args = dict(dataset=Uint8CropDataset(dataset), num_workers=args.num_workers)
    if spatial_order:
        loader_args["batch_sampler"] = SourceGroupedBatchSampler(
            rows, batch_size=args.batch_size, shuffle=False, spatial_order=True)
    else:
        loader_args["batch_size"] = args.batch_size
    if args.num_workers:
        loader_args.update(multiprocessing_context="spawn", prefetch_factor=2)
    loader = DataLoader(**loader_args)
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    branches = ["local", "context"] + (["fine"] if config.use_fine_branch else [])
    if supervised:
        writers = {branch: ShardWriter(root, f"{branch}_", len(rows), args.rows_per_shard,
                    (1, *([config.fine_input_size_px if branch == "fine" else config.input_size_px] * 2)))
                   for branch in branches}
        manifest = rows.copy()
        manifest["prepared_index"] = np.arange(len(rows))
    else:
        writers = {"pairs": ShardWriter(root, "shard_", 2 * len(rows), args.rows_per_shard,
                                         (1, config.input_size_px, config.input_size_px))}
        source_ids = np.repeat(rows.source_id.to_numpy(), 2)
        manifest = pd.DataFrame(dict(
            prepared_index=np.arange(2 * len(rows)), source_id=source_ids, source_group="user",
            section_id=[sources[sid].section_id or sid for sid in source_ids],
            center_id=np.repeat([f"{sid}:{i}" for i, sid in enumerate(rows.source_id)], 2),
            scale_id=np.tile([0, 1], len(rows))))
    iterator = iter(loader)
    try:
        for batch in tqdm(iterator, total=len(loader), desc="Preparing crop shards"):
            indices = batch["index"].numpy()
            if supervised:
                for branch, writer in writers.items():
                    writer.write(indices, batch[f"{branch}_image"].numpy())
            else:
                for scale, branch in enumerate(branches):
                    writers["pairs"].write(indices * 2 + scale, batch[f"{branch}_image"].numpy())
        paths = {branch: writer.finish() for branch, writer in writers.items()}
    finally:
        del iterator
        for writer in writers.values():
            writer.close()
        for reader in dataset.extractor.backends.values():
            reader.close()
    metadata = dict(manifest_path="manifest.csv", completed_row_count=len(manifest),
                    rows_per_shard=args.rows_per_shard, image_dtype="uint8", input_protocol=INPUT_PROTOCOL,
                    crop_configuration={key: getattr(config, key) for key in
                        ("local_crop_px", "context_crop_px", "fine_crop_px", "input_size_px",
                         "fine_input_size_px", "use_fine_branch")},
                    reference_pixel_size_um=args.reference_pixel_size_um,
                    preparation=dict(crop_backend=args.crop_backend, num_workers=args.num_workers,
                                     tile_cache_mib=args.tile_cache_mib, batch_size=args.batch_size,
                                     spatial_order=spatial_order, preserves_input_row_order=True),
                    preparation_seconds=time.perf_counter() - started)
    if supervised:
        metadata.update({f"{branch}_shard_paths": paths.get(branch, []) for branch in ("local", "context", "fine")})
    else:
        metadata.update(variant_count=1, shard_paths=paths["pairs"],
                        description="Own-data preparation; paper reproduction uses its frozen pretraining arrays.")
    manifest.to_csv(root / "manifest.csv", index=False)
    temporary = root / "metadata.json.partial"
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(root / "metadata.json")
    print(root / "metadata.json")
    return metadata
