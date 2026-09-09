"""Shared, linear probability-to-color mapping for the two fixed IF windows."""
from paper_paths import Path, lock_record, rendering_script
import json

import numpy as np
from matplotlib.colors import Normalize

DISPLAY_CONFIG = Path(__file__).parent / "panel_E/processing/probability_display.json"


def probability_display():
    config = json.loads(DISPLAY_CONFIG.read_text())
    low, high = config["limits"]
    if not 0 <= low < high <= 1 or config["clip_colors_only"] is not True:
        raise RuntimeError("Invalid fixed-window probability display limits")
    endpoint = f"≥{high:g}" if high < 1 else "1"
    if config["ticks"][-1] != high or config["ticklabels"][-1] != endpoint:
        raise RuntimeError("A reduced probability maximum must mark color saturation")
    return config, Normalize(low, high, clip=True)


def probability_range(values):
    """Display audit only; the returned values never replace model predictions."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise RuntimeError("Peyer probabilities must be finite values in [0,1]")
    config, _ = probability_display()
    return {"n": len(values), "minimum": float(values.min()), "maximum": float(values.max()),
            "above_display_maximum": int((values > config["limits"][1]).sum()),
            "display_limits": config["limits"], "predictions_rescaled": False}


def label_probability_colorbar(bar):
    config, _ = probability_display()
    bar.set_ticks(config["ticks"], labels=config["ticklabels"])
    bar.set_label("Peyer’s patch probability", fontsize=7.5)
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
