"""Cached raw-image training with unchanged batches and epoch-specific augmentation."""
from __future__ import annotations

import shutil
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .augmentation import apply_intensity_parameters, sample_intensity_parameters
from .predict import CellCropDataset, SourceGroupedBatchSampler, _collate


class RamCropCache:
    """A bounded shared-memory cache of unaugmented, production-protocol crops.

    Slots have fixed row identities. Each row is visited once per complete epoch,
    so workers never write the same slot concurrently. Training and validation use
    disjoint index ranges. Store bytes, then apply augmentation after every read.
    """

    def __init__(self, row_count, config, budget_mib):
        self.shapes = {
            f"{branch}_image": (1, size, size)
            for branch, size in (("local", config.input_size_px), ("context", config.input_size_px),
                                 ("fine", config.fine_input_size_px))
            if getattr(config, f"use_{branch}_branch")
        }
        self.row_bytes = sum(int(np.prod(shape)) for shape in self.shapes.values())
        if not self.row_bytes:
            raise ValueError("At least one image branch is required")
        self.capacity = min(row_count, int(budget_mib * 1024**2) // (self.row_bytes + 1))
        self.allocated_bytes = self.capacity * (self.row_bytes + 1)
        if self.capacity and sys.platform == "linux":
            free = shutil.disk_usage("/dev/shm").free
            if self.allocated_bytes + 16 * 1024**2 > free:
                raise ValueError("Not enough /dev/shm space for the RAM crop cache. Reduce "
                                 "--ram-crop-cache-mib or increase container shared memory.")
        self.images = self.ready = None
        if self.capacity:
            self.images = torch.empty((self.capacity, self.row_bytes), dtype=torch.uint8).share_memory_()
            self.ready = torch.zeros(self.capacity, dtype=torch.uint8).share_memory_()

    def get(self, index):
        if index >= self.capacity or not self.ready[index].item():
            return None
        flat = self.images[index].numpy()
        result, start = {}, 0
        for key, shape in self.shapes.items():
            stop = start + int(np.prod(shape))
            result[key] = flat[start:stop].reshape(shape).astype(np.float32) / 255.0
            start = stop
        return result

    def put(self, index, item):
        if index >= self.capacity:
            return
        flat, start = self.images[index].numpy(), 0
        for key, shape in self.shapes.items():
            stop = start + int(np.prod(shape))
            flat[start:stop] = np.rint(item[key] * 255).astype(np.uint8).reshape(-1)
            start = stop
        self.ready[index] = 1


class RawTrainingDataset(Dataset):
    def __init__(self, rows, sources, config, *, cache=None, cache_offset=0,
                 intensity_augmentation=None, **crop_options):
        self.raw = CellCropDataset(rows, sources, config, **crop_options)
        self.rows = self.raw.rows
        self.cache = cache
        self.cache_offset = cache_offset
        self.augmentation = intensity_augmentation
        self.epoch = 0

    def __len__(self):
        return len(self.raw)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def load_base(self, index):
        cache_index = self.cache_offset + index
        item = None if self.cache is None else self.cache.get(cache_index)
        if item is None:
            item = self.raw[index]
            if self.cache is not None:
                self.cache.put(cache_index, item)
        return {"index": index, **item}

    def __getitem__(self, key):
        # Carry the epoch with each work item; parent set_epoch() alone cannot
        # update copies of a dataset inside persistent worker processes.
        epoch, index = key if isinstance(key, tuple) else (self.epoch, int(key))
        item = self.load_base(index)
        if self.augmentation is not None:
            params = sample_intensity_parameters(self.augmentation, sample_index=index, epoch=epoch)
            if params is not None:
                contrast, brightness, sigma = params
                for name in self.raw_image_keys:
                    item[name] = apply_intensity_parameters(
                        item[name], contrast=contrast, brightness=brightness, gaussian_blur_sigma=sigma)
        return item

    @property
    def raw_image_keys(self):
        return [f"{branch}_image" for branch in ("local", "context", "fine")
                if getattr(self.raw.config, f"use_{branch}_branch")]

    def close(self):
        for reader in self.raw.extractor.backends.values():
            reader.close()
        self.raw.extractor.backends.clear()


class EpochBatchSampler:
    def __init__(self, sampler):
        self.sampler = sampler
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        epoch = self.epoch
        for batch in self.sampler:
            yield [(epoch, index) for index in batch]


class RawTrainingLoader(DataLoader):
    """Keep worker caches alive without changing the model's CPU RNG sequence."""

    def __iter__(self):
        if self.persistent_workers and getattr(self, "_has_iterated", False):
            # PyTorch 2.12's ordinary DataLoader draws an int64 base seed on
            # every new iterator. Its persistent reset omits that draw. Retain
            # it here so CPU dropout matches the original loader across epochs.
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        self._has_iterated = True
        return super().__iter__()

    def close(self):
        iterator = getattr(self, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            self._iterator = None
        self.dataset.close()


def make_raw_training_loader(rows, sources, config, *, batch_size, seed, shuffle, num_workers,
                             device, crop_backend, tile_cache_mib, reference_pixel_size_um,
                             input_protocol, intensity_augmentation=None, cache=None,
                             cache_offset=0, persistent_workers=True):
    dataset = RawTrainingDataset(rows, sources, config, cache=cache, cache_offset=cache_offset,
                                 intensity_augmentation=intensity_augmentation, crop_backend=crop_backend,
                                 tile_cache_mib=tile_cache_mib, reference_pixel_size_um=reference_pixel_size_um,
                                 input_protocol=input_protocol)
    sampler = SourceGroupedBatchSampler(rows, batch_size=batch_size, shuffle=shuffle, seed=seed)
    options = dict(num_workers=num_workers, pin_memory=str(device).startswith("cuda"))
    if num_workers:
        options.update(multiprocessing_context="spawn", persistent_workers=persistent_workers, prefetch_factor=2)
    return RawTrainingLoader(dataset, batch_sampler=EpochBatchSampler(sampler), collate_fn=_collate, **options)


class _WarmCacheDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.cache.capacity

    def __getitem__(self, index):
        self.dataset.load_base(int(index))
        # Crops stay in shared RAM; only completion indices cross the queue.
        return int(index)


def warm_ram_cache(cache, train_rows, val_rows, sources, config, *, num_workers, **crop_options):
    if not cache.capacity:
        return
    rows = pd.concat([train_rows, val_rows], ignore_index=True) if val_rows is not None else train_rows
    rows = rows.iloc[:cache.capacity]
    dataset = RawTrainingDataset(rows, sources, config, cache=cache, **crop_options)
    sampler = SourceGroupedBatchSampler(rows, batch_size=128, shuffle=False, spatial_order=True)
    options = {"num_workers": num_workers}
    if num_workers:
        options.update(multiprocessing_context="spawn", prefetch_factor=2)
    iterator = None
    try:
        # Spatial ordering is only used to fill cache slots, never for SGD.
        # Warmup must not advance the model's random-number stream.
        with torch.random.fork_rng(devices=[]):
            loader = DataLoader(_WarmCacheDataset(dataset), batch_sampler=sampler, **options)
            iterator = iter(loader)
            for _ in tqdm(iterator, total=len(loader), desc="Warming RAM crop cache", unit="batch"):
                pass
        if int(cache.ready.sum()) != cache.capacity:
            raise RuntimeError("RAM crop cache warmup did not complete")
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        dataset.close()


def close_training_loader(loader):
    if loader is None:
        return
    if isinstance(loader, RawTrainingLoader):
        loader.close()
    elif isinstance(loader.dataset, CellCropDataset):
        for reader in loader.dataset.extractor.backends.values():
            reader.close()
