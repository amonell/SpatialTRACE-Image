from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from Figure_4_Extended.common import save_pair
from Figure_4_Extended.probability_display import probability_display, label_probability_colorbar


def contrast(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(values[np.isfinite(values)], [1.0, 99.8])
    return np.clip((values - low) / max(high - low, 1e-6), 0, 1)


def scale_bar(axis, bounds_um: tuple[float, float, float, float]) -> None:
    x0, x1, y0, y1 = bounds_um
    start = x0 + 0.06 * (x1 - x0)
    y = y1 - 0.08 * (y1 - y0)
    line = axis.plot([start, start + 100], [y, y], color="white", lw=2.5, solid_capstyle="butt")[0]
    line.set_path_effects([pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])
    text = axis.text(start + 50, y - 0.025 * (y1 - y0), "100 μm", ha="center", va="bottom", color="white", fontsize=7)
    text.set_path_effects([pe.Stroke(linewidth=2, foreground="black"), pe.Normal()])


def render_peyer(*, image: np.ndarray, rows: pd.DataFrame, condition: str, center_um: tuple[float, float], zoom_um: float, pixel_size_um: float, stem, dpi: int):
    cx, cy = center_um
    half = zoom_um / 2
    bounds_um = (cx - half, cx + half, cy - half, cy + half)
    bounds_px = tuple(int(round(value / pixel_size_um)) for value in bounds_um)
    x0, x1, y0, y1 = bounds_px
    crop = np.asarray(image[y0:y1, x0:x1])
    selected = rows.loc[rows.condition.astype(str).eq(condition) & rows.centroid_x_fullres_px.between(x0, x1) & rows.centroid_y_fullres_px.between(y0, y1)].copy()
    if selected.empty:
        raise RuntimeError(f"No {condition} cells in fixed Peyer window")
    selected["x_um"] = selected.centroid_x_fullres_px.astype(float) * pixel_size_um
    selected["y_um"] = selected.centroid_y_fullres_px.astype(float) * pixel_size_um
    figure, axes = plt.subplots(1, 2, figsize=(5.0, 2.55))
    axes[0].imshow(contrast(crop), cmap="gray", extent=(bounds_um[0], bounds_um[1], bounds_um[3], bounds_um[2]), interpolation="none")
    axes[0].set_title(f"{condition} IF image", fontsize=8.5)
    # These fixed windows contain only a few thousand cells. Preserve their
    # markers as vectors rather than embedding a low-resolution point bitmap.
    axes[1].scatter(selected.x_um, selected.y_um, s=2.0, color="#c7c7c7", linewidths=0, rasterized=False)
    display, norm = probability_display()
    order = np.argsort(selected.peyer_probability.to_numpy(float))
    artist = axes[1].scatter(selected.x_um.to_numpy()[order], selected.y_um.to_numpy()[order], c=selected.peyer_probability.to_numpy(float)[order], cmap=display["colormap"], norm=norm, s=5.5, linewidths=0, rasterized=False)
    axes[1].set_title("Image-classifier Peyer’s patch probability", fontsize=8.5)
    for axis in axes:
        axis.set_xlim(bounds_um[0], bounds_um[1]); axis.set_ylim(bounds_um[3], bounds_um[2]); axis.axis("off"); scale_bar(axis, bounds_um)
    # Allocate a dedicated colorbar rail. Calling subplots_adjust after an
    # ax-attached colorbar previously restored the plot over that colorbar.
    colorbar_axis = figure.add_axes([0.885, 0.055, 0.026, 0.80])
    colorbar = figure.colorbar(artist, cax=colorbar_axis)
    label_probability_colorbar(colorbar)
    figure.subplots_adjust(left=0.01, right=0.86, bottom=0.03, top=0.88, wspace=0.06)
    rendered = save_pair(figure, stem, dpi)
    plt.close(figure)
    return rendered, selected, bounds_um
