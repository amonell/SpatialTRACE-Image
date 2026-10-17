"""Read the reviewed final-artwork order; never infer panel order from filenames."""
from __future__ import annotations

import hashlib
import json
from paper_paths import Path, lock_record, rendering_script

REPO = Path(__file__).resolve().parent
CATALOG = REPO / "final_figure_order.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(*, verify_artwork: bool = True) -> dict:
    catalog = json.loads(CATALOG.read_text())
    packages = [item["package"] for item in catalog["figures"]]
    if len(packages) != len(set(packages)):
        raise ValueError("Duplicate figure in final order")
    for item in catalog["figures"]:
        if verify_artwork:
            path = Path(catalog["output_root"]) / item["reference"]
            if sha256(path) != item["reference_sha256"]:
                raise ValueError(f"Final artwork changed: {path}. Review its displayed panel order before updating the lock.")
    return catalog


def panel_map(package: str) -> list[dict]:
    catalog = load_catalog(verify_artwork=False)
    item = next(item for item in catalog["figures"] if item["package"] == package)
    mapping = json.loads((REPO / package / "panel_order.json").read_text())
    panels = mapping["panels"]
    if [panel["letter"] for panel in panels] != item["panels"]:
        raise ValueError(f"{package}/panel_order.json disagrees with reviewed artwork order")
    for panel in panels:
        for script in panel.get("render_scripts", []):
            if not (REPO / package / script).is_file():
                raise FileNotFoundError(REPO / package / script)
    return panels


def ordered_components(package: str) -> list[tuple[str, str]]:
    """Return (panel letter, component stem) in explicit artist import order."""
    result = []
    for panel in panel_map(package):
        for component in panel.get("components", []):
            stem = component["stem"] if isinstance(component, dict) else component
            if Path(stem).name != stem or Path(stem).suffix:
                raise ValueError(f"Expected a bare PDF/PNG component stem, got {stem}")
            result.append((panel["letter"], stem))
    if len(result) != len(set(result)):
        raise ValueError(f"{package}: duplicate import component")
    return result
