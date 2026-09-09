from __future__ import annotations
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from Figure_4.common import require_file

GATE_ORDER = ("Top IE", "Top LP", "Crypt IE", "Crypt LP", "Muscularis")
GATE_COLORS = {
    "Top IE": "#7CB7C9", "Top LP": "#9FBF7A", "Crypt IE": "#D68B81",
    "Crypt LP": "#C8B96A", "Muscularis": "#B79572",
}
GATE_FILLS = {
    "Top IE": "#3A9AB244", "Top LP": "#A5C88199", "Crypt IE": "#F11B0033",
    "Crypt LP": "#BDC88166", "Muscularis": "#D8C7AC66",
}


def common_tissue_limits(frames: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float, float, float]]:
    ranges, width, height = {}, 0.0, 0.0
    for condition, frame in frames.items():
        xmin, xmax = float(frame.x_um.min()), float(frame.x_um.max())
        ymin, ymax = float(frame.y_um.min()), float(frame.y_um.max())
        ranges[condition] = xmin, xmax, ymin, ymax
        width, height = max(width, xmax - xmin), max(height, ymax - ymin)
    width *= 1.025
    height *= 1.025
    output = {}
    for condition, (xmin, xmax, ymin, ymax) in ranges.items():
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        output[condition] = cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2
    return output


def add_scale_bar(axis: plt.Axes, bounds: tuple[float, float, float, float], length_um: float = 1000.0) -> None:
    xmin, xmax, ymin, ymax = bounds
    x0, y0 = xmin + 0.065 * (xmax - xmin), ymax - 0.07 * (ymax - ymin)
    axis.plot([x0, x0 + length_um], [y0, y0], color="black", lw=2.0, solid_capstyle="butt")
    axis.text(x0 + length_um / 2, y0 - 0.022 * (ymax - ymin), "1 mm", ha="center", va="bottom", fontsize=7)


def gate_thresholds(definitions: dict) -> tuple[float, float, float]:
    """Read the supported gate schema; the release owns numerical boundaries."""
    if definitions.get("schema_version") != "figure4_coordinate_gates/v1":
        raise RuntimeError("Unsupported coordinate-gate schema")
    thresholds = definitions["thresholds"]
    split = float(thresholds["split_x"])
    low = float(thresholds["crypt_lp_cv_min"])
    high = float(thresholds["crypt_lp_cv_max"])
    if (thresholds["x_range"] != [0, 1] or thresholds["y_range"] != [0, 1]
            or not np.isfinite([split, low, high]).all()
            or not (0 < split < 1 and 0 < low < high < 1)):
        raise RuntimeError("Invalid normalized-coordinate gate thresholds")
    if set(definitions["boundary_conventions"]) != set(GATE_ORDER):
        raise RuntimeError("Unsupported coordinate-gate membership convention")
    return split, low, high


def gate_background(axis: plt.Axes, definitions: dict) -> None:
    split, low, high = gate_thresholds(definitions)
    rectangles = {
        "Top IE": (0, split, high, 1), "Top LP": (split, 1, high, 1),
        "Crypt IE": (0, split, 0, high), "Crypt LP": (split, 1, low, high),
        "Muscularis": (split, 1, 0, low),
    }
    for gate, (x0, x1, y0, y1) in rectangles.items():
        axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=GATE_FILLS[gate], edgecolor="none"))
    axis.axvline(split, color="#222222", lw=0.65)
    axis.axhline(high, color="#222222", lw=0.65)
    axis.hlines(low, split, 1, color="#222222", lw=0.65)


def frozen_density(frame, density_manifest, key):
    record = density_manifest['populations'][key]
    cells = pd.read_parquet(require_file(record['cells']['path'], record['cells']['sha256']))
    keys = ['source_id', 'tissue_cluster_label', 'object_id']
    if not cells[keys].reset_index(drop=True).equals(frame[keys].reset_index(drop=True)):
        raise RuntimeError('Frozen density cell order does not match the requested IMAP')
    columns = ['predicted_axis_coordinate', 'predicted_epithelial_distance_clipped_1p0']
    if not np.array_equal(cells[columns].to_numpy(), frame[columns].to_numpy()):
        raise RuntimeError('Frozen density uses different predictions')
    array = np.load(require_file(record['grid']['path'], record['grid']['sha256']))
    gx, gy = np.meshgrid(array['x'], array['y'], indexing='ij')
    return cells.frozen_probability_density.to_numpy(), gx, gy, array['grid']


def gate_percentages(frame: pd.DataFrame, definitions: dict) -> dict[str, float]:
    """Diagnostic validation only; publication labels use frozen summary tables."""
    split, low, high = gate_thresholds(definitions)
    x = frame.predicted_epithelial_distance_clipped_1p0.to_numpy(float)
    y = frame.predicted_axis_coordinate.to_numpy(float)
    membership = {
        "Top IE": (x <= split) & (y >= high),
        "Top LP": (x > split) & (y >= high),
        "Crypt IE": (x <= split) & (y < high),
        "Crypt LP": (x > split) & (y >= low) & (y < high),
        "Muscularis": (x > split) & (y < low),
    }
    return {gate: 100 * float(mask.sum()) / len(frame) for gate, mask in membership.items()}


def frozen_gate_percentages(table: pd.DataFrame, condition: str) -> dict[str, float]:
    column = "percent" if "percent" in table.columns else "percentage"
    rows = table.loc[table.condition.eq(condition)]
    if len(rows) != len(GATE_ORDER) or rows.gate.duplicated().any() or set(rows.gate) != set(GATE_ORDER):
        raise RuntimeError(f"Missing or ambiguous frozen gate summaries: {condition}")
    values = rows.set_index("gate")[column].reindex(GATE_ORDER).astype(float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 100):
        raise RuntimeError(f"Invalid frozen gate percentages: {condition}")
    return values.to_dict()


def add_gate_labels(axis: plt.Axes, percentages: dict[str, float], *, population: str, condition: str):
    from Figure_4.imap_labels import draw_registered_gate_labels
    return draw_registered_gate_labels(axis, percentages, population, condition, GATE_COLORS)


def render_imap(frames: dict[str, pd.DataFrame], *, population: str, density_manifest,
                gate_definitions: dict, gate_fractions: pd.DataFrame, stem, dpi: int,
                vector_points: bool = False, layout: str = "side_by_side") -> list:
    if layout not in {"side_by_side", "stacked"}:
        raise ValueError(f"Unknown IMAP layout: {layout}")
    ordered = [frames[condition] for condition in ("DMSO", "RARi")]
    density_record = json.loads(require_file(density_manifest).read_text())
    if not density_record['all_four_populations_share_normalization']:
        raise RuntimeError('P14/non-P14 density normalization is not shared')
    suffix = 'p14' if population == 'P14' else 'non_p14'
    densities = [frozen_density(frames[c], density_record, f'{c.lower()}_{suffix}') for c in ('DMSO', 'RARi')]
    if layout == "stacked":
        figure, axes = plt.subplots(2, 1, figsize=(4.5, 6.9), sharex=True, sharey=True)
    else:
        figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True, sharey=True)
    for axis, condition, frame, density_bundle in zip(axes, ("DMSO", "RARi"), ordered, densities, strict=True):
        density, grid_x, grid_y, grid = density_bundle
        norm = Normalize(*density_record['shared_color_limits'])
        gate_background(axis, gate_definitions)
        order = np.argsort(density)
        artist = axis.scatter(
            frame.predicted_epithelial_distance_clipped_1p0.to_numpy(float)[order],
            frame.predicted_axis_coordinate.to_numpy(float)[order],
            c=density[order], cmap="viridis", norm=norm,
            s=2.5 if population == "P14" else 0.35, linewidths=0, rasterized=not vector_points, alpha=0.86,
        )
        levels = [v for v in density_record['shared_contour_levels'] if grid.min() < v < grid.max()]
        if levels:
            axis.contour(grid_x, grid_y, grid, levels=levels, colors="#666666", linewidths=0.55, alpha=0.85, zorder=4)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title(f"{condition} {population} cells\nn = {len(frame):,}", fontweight="bold", fontsize=8.5, pad=24)
        axis.set_xlabel("Epithelial-distance coordinate", labelpad=26)
        if layout == "stacked" and axis is axes[0]:
            axis.set_xlabel("")
        axis.set_ylabel("Crypt-villus coordinate")
        colorbar = figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.025, shrink=0.88, label="Probability density")
        if vector_points and colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)
            # Overlap neighboring vector fills to avoid PDF-viewer white seams.
            colorbar.solids.set_edgecolor("face")
    if layout == "stacked":
        figure.subplots_adjust(left=0.17, right=0.95, bottom=0.13, top=0.89, hspace=0.85)
    else:
        figure.subplots_adjust(left=0.09, right=0.98, bottom=0.26, top=0.77, wspace=0.30)
    # Original artwork registration is shared with final-page assembly. Apply
    # it after axes/colorbar layout is fixed so point-size scaling is explicit.
    for axis, condition in zip(axes, ("DMSO", "RARi"), strict=True):
        add_gate_labels(axis, frozen_gate_percentages(gate_fractions, condition),
                        population=population, condition=condition)
    from Figure_4.common import save_pair
    rendered = save_pair(figure, stem, dpi)
    plt.close(figure)
    return rendered
