from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class IntensityAugmentationConfig:
    """Gaussian blur and intensity jitter for normalized single-channel crops."""

    probability: float = 0.0
    brightness_delta: float = 0.0
    contrast_range: tuple[float, float] = (1.0, 1.0)
    gaussian_blur_sigma_range: tuple[float, float] = (0.0, 0.0)
    seed: int = 0

    def __post_init__(self) -> None:
        probability = float(self.probability)
        brightness_delta = float(self.brightness_delta)
        contrast_min, contrast_max = (float(value) for value in self.contrast_range)
        blur_min, blur_max = (float(value) for value in self.gaussian_blur_sigma_range)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("augmentation probability must be between 0 and 1.")
        if brightness_delta < 0.0:
            raise ValueError("augmentation brightness_delta must be non-negative.")
        if contrast_min <= 0.0 or contrast_max <= 0.0 or contrast_min > contrast_max:
            raise ValueError("augmentation contrast_range must be positive and sorted.")
        if blur_min < 0.0 or blur_max < 0.0 or blur_min > blur_max:
            raise ValueError("augmentation gaussian_blur_sigma_range must be non-negative and sorted.")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "brightness_delta", brightness_delta)
        object.__setattr__(self, "contrast_range", (contrast_min, contrast_max))
        object.__setattr__(self, "gaussian_blur_sigma_range", (blur_min, blur_max))
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def enabled(self) -> bool:
        contrast_min, contrast_max = self.contrast_range
        blur_min, blur_max = self.gaussian_blur_sigma_range
        changes_intensity = bool(self.brightness_delta > 0.0 or contrast_min != 1.0 or contrast_max != 1.0)
        changes_blur = bool(blur_min > 0.0 or blur_max > 0.0)
        return bool(self.probability > 0.0 and (changes_intensity or changes_blur))

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["contrast_range"] = list(self.contrast_range)
        values["gaussian_blur_sigma_range"] = list(self.gaussian_blur_sigma_range)
        values["enabled"] = self.enabled
        return values


def make_intensity_augmentation_config(
    *,
    probability: float = 0.0,
    brightness_delta: float = 0.0,
    contrast_range: Iterable[float] = (1.0, 1.0),
    gaussian_blur_sigma_range: Iterable[float] = (0.0, 0.0),
    seed: int = 0,
) -> IntensityAugmentationConfig | None:
    contrast_values = tuple(float(value) for value in contrast_range)
    blur_values = tuple(float(value) for value in gaussian_blur_sigma_range)
    if len(contrast_values) != 2:
        raise ValueError("augmentation contrast_range must contain two values.")
    if len(blur_values) != 2:
        raise ValueError("augmentation gaussian_blur_sigma_range must contain two values.")
    config = IntensityAugmentationConfig(
        probability=float(probability),
        brightness_delta=float(brightness_delta),
        contrast_range=(contrast_values[0], contrast_values[1]),
        gaussian_blur_sigma_range=(blur_values[0], blur_values[1]),
        seed=int(seed),
    )
    return config if config.enabled else None


def sample_intensity_parameters(
    config: IntensityAugmentationConfig,
    *,
    sample_index: int,
    epoch: int,
) -> tuple[float, float, float] | None:
    rng_seed = (
        int(config.seed)
        + int(sample_index) * 1_000_003
        + int(epoch) * 9_176_711
    ) % (2**32)
    rng = np.random.default_rng(rng_seed)
    if float(rng.random()) >= float(config.probability):
        return None
    contrast_min, contrast_max = config.contrast_range
    blur_min, blur_max = config.gaussian_blur_sigma_range
    contrast = float(rng.uniform(contrast_min, contrast_max))
    brightness = float(rng.uniform(-float(config.brightness_delta), float(config.brightness_delta)))
    blur_sigma = float(rng.uniform(blur_min, blur_max))
    return contrast, brightness, blur_sigma


def apply_intensity_parameters(
    image: np.ndarray,
    *,
    contrast: float,
    brightness: float,
    gaussian_blur_sigma: float = 0.0,
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    blur_sigma = float(gaussian_blur_sigma)
    if blur_sigma < 0.0:
        raise ValueError("gaussian_blur_sigma must be non-negative.")
    if blur_sigma > 0.0:
        if values.ndim == 2:
            sigma = (blur_sigma, blur_sigma)
        elif values.ndim == 3:
            sigma = (0.0, blur_sigma, blur_sigma)
        else:
            raise ValueError(f"Expected a 2D or channel-first 3D image, got shape {values.shape}.")
        values = gaussian_filter(values, sigma=sigma, mode="reflect").astype(np.float32, copy=False)
    mean = float(values.mean(dtype=np.float64))
    adjusted = (values - mean) * float(contrast) + mean + float(brightness)
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32, copy=False)
