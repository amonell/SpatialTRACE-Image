from __future__ import annotations

from paper_paths import Path, lock_record, rendering_script

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from common import save_pair


def zoom_limits(table: pd.DataFrame, fraction: float = .22) -> tuple[float, float, float, float]:
    x, y = table.centroid_x_fullres_px.to_numpy(float), table.centroid_y_fullres_px.to_numpy(float)
    x_min, x_max, y_min, y_max = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    width, height = (x_max - x_min) * fraction, (y_max - y_min) * fraction
    centers_x = np.linspace(x_min + width / 2, x_max - width / 2, 28)
    centers_y = np.linspace(y_min + height / 2, y_max - height / 2, 28)
    tissue_x, tissue_y = float(np.median(x)), float(np.median(y))
    best = (-1.0, tissue_x, tissue_y)
    for center_x in centers_x:
        x_mask = np.abs(x - center_x) <= width / 2
        for center_y in centers_y:
            count = int(np.count_nonzero(x_mask & (np.abs(y - center_y) <= height / 2)))
            score = count + .015 * float(np.hypot(center_x - tissue_x, center_y - tissue_y))
            if score > best[0]:
                best = (score, float(center_x), float(center_y))
    _, center_x, center_y = best
    left = max(x_min, min(center_x - width / 2, x_max - width)); bottom = max(y_min, min(center_y - height / 2, y_max - height))
    return left, left + width, bottom, bottom + height


def render_map(table: pd.DataFrame, *, column: str, title: str, cmap: str, vmax: float, stem: Path, dpi: int, limits: tuple[float, float, float, float] | None, show_box: tuple[float, float, float, float] | None, dot_size: float) -> list[Path]:
    display = table
    if limits:
        x0, x1, y0, y1 = limits
        display = table[table.centroid_x_fullres_px.between(x0, x1) & table.centroid_y_fullres_px.between(y0, y1)]
    figure, axis = plt.subplots(figsize=(3.1, 3.1))
    artist = axis.scatter(display.centroid_x_fullres_px, display.centroid_y_fullres_px, c=display[column], s=dot_size, linewidths=0, cmap=cmap, norm=Normalize(0, vmax), rasterized=True)
    if show_box:
        x0, x1, y0, y1 = show_box
        axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#111111", linewidth=1.1))
    if limits:
        x0, x1, y0, y1 = limits; axis.set_xlim(x0, x1); axis.set_ylim(y1, y0)
    else:
        pad = max(float(table.centroid_x_fullres_px.max() - table.centroid_x_fullres_px.min()), float(table.centroid_y_fullres_px.max() - table.centroid_y_fullres_px.min())) * .02
        axis.set_xlim(float(table.centroid_x_fullres_px.min()) - pad, float(table.centroid_x_fullres_px.max()) + pad)
        axis.set_ylim(float(table.centroid_y_fullres_px.max()) + pad, float(table.centroid_y_fullres_px.min()) - pad)
    axis.set_aspect("equal"); axis.axis("off"); axis.set_title(title, fontweight="bold")
    figure.colorbar(artist, ax=axis, fraction=.046, pad=.02)
    figure.tight_layout(pad=.2)
    rendered = save_pair(figure, stem, dpi); plt.close(figure)
    return rendered
