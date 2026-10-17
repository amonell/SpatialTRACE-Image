"""Recoverably publish only declared, freshly generated final-panel exports."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from paper_paths import Path, lock_record, rendering_script

from figure_assembly.compositor import file_sha256
from figure_organization import REPO, panel_map

OWNER = "spatial_axes_manuscript.final_panel_layers"


def targets(data_root, package, letter):
    directory = Path(data_root) / package / f"panel_{letter}"
    stem = f"panel_{letter}_final"
    return {"pdf": directory / (stem + ".pdf"), "png": directory / (stem + ".png"),
            "provenance": directory / (stem + "_provenance.json")}


def preflight_publication(entries, data_root, overwrite=False):
    for entry in entries:
        for letter in entry["panels"]:
            paths = targets(data_root, entry["package"], letter)
            occupied = [path for path in paths.values() if path.exists()]
            if not occupied:
                continue
            if not overwrite:
                raise FileExistsError(f"Final panels already exist; --overwrite archives only their declared exports: {occupied}")
            provenance = paths["provenance"]
            if not provenance.is_file() or json.loads(provenance.read_text()).get("publication_owner") != OWNER:
                raise FileExistsError(f"Refusing to replace unowned final-panel files: {occupied}")
        index = Path(data_root) / entry["package"] / "final_panel_manifest.json"
        if index.exists() and json.loads(index.read_text()).get("owner") != OWNER:
            raise FileExistsError(f"Refusing to replace unowned panel index: {index}")


def source_map(package, letter):
    panel = next(p for p in panel_map(package) if p["letter"] == letter)
    fixed = package in {"Figure_2", "Figure_2_Extended", "Figure_2_Extended_2", "Figure_3_Extended"}
    style = [REPO / package / "assembly/layout.json", REPO / package / "assembly/prepare.py"]
    style += ([REPO / "Figure_2/assembly/fixed_render.py"] if fixed else
              sorted((REPO / package / "assembly").glob("*.py")))
    paths = [REPO / package / p for p in panel.get("render_scripts", [])] + style
    final_renderer = ("Figure_2/assembly/fixed_render.py" if fixed else
                      f"{package}/assembly/render.py" if package in {"Figure_1", "Figure_3"} else
                      f"{package}/assembly/prepare.py")
    return {"component_scripts": panel.get("render_scripts", []),
            "authoritative_final_renderer": final_renderer,
            "authoritative_layout": f"{package}/assembly/layout.json",
            "editing_note": "Use the authoritative final renderer/layout for finished panel styling; component_scripts identify smaller source assets.",
            "final_style_and_layout": [str(p.relative_to(REPO)) for p in dict.fromkeys(style)],
            "source_hashes": [{"path": str(p), "sha256": file_sha256(p)} for p in dict.fromkeys(paths)],
            "update_command": f"python -B update_final_figure.py {package} --overwrite"}


def publish_panels(state, data_root, overwrite=False):
    """Stage everything, archive exact owned files, and roll back on failure."""
    root = Path(data_root).resolve()
    entries = [{"package": f["package"], "panels": [p["panel"] for p in f["final_panels"]]} for f in state["figures"]]
    preflight_publication(entries, root, overwrite)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    operations, manifests = [], []
    for figure in state["figures"]:
        package = figure["package"]
        staging = root / package / "stored_outputs/final_panel_publications" / stamp
        archive = root / package / "polished_panel_archive" / f"{stamp}_final_layers"
        staging.mkdir(parents=True, exist_ok=False)
        records = []
        for panel in figure["final_panels"]:
            letter = panel["panel"]
            destination = targets(root, package, letter)
            panel_stage = staging / f"panel_{letter}"
            panel_stage.mkdir()
            provenance = json.loads(Path(panel["provenance"]).read_text())
            for key in ("pdf", "png"):
                source = Path(panel[key])
                if file_sha256(source) != panel[key + "_sha256"]:
                    raise ValueError(f"Final panel changed after QA: {source}")
                shutil.copy2(source, panel_stage / destination[key].name)
                provenance[key] = str(destination[key])
            provenance.update({"publication_owner": OWNER, "published_at_utc": stamp,
                               "source_run": state["output_root"], "source_map": source_map(package, letter),
                               "provenance": str(destination["provenance"])})
            (panel_stage / destination["provenance"].name).write_text(json.dumps(provenance, indent=2) + "\n")
            for key, target in destination.items():
                operations.append({"source": panel_stage / target.name, "target": target,
                                   "archive": archive / f"panel_{letter}" / target.name,
                                   "old_sha256": file_sha256(target) if target.exists() else None})
            records.append({"panel": letter, **{key: str(value) for key, value in destination.items()},
                            "pdf_sha256": panel["pdf_sha256"], "png_sha256": panel["png_sha256"],
                            "source_map": provenance["source_map"]})
        manifest = {"owner": OWNER, "package": package, "published_at_utc": stamp,
                    "complete_page": str(Path(state["output_root"]) / figure["pdf"]),
                    "complete_page_sha256": file_sha256(Path(state["output_root"]) / figure["pdf"]),
                    "canvas": "Registered transparent artwork canvas; import at origin without scaling.",
                    "panels": records}
        target = root / package / "final_panel_manifest.json"
        source = staging / target.name
        source.write_text(json.dumps(manifest, indent=2) + "\n")
        operations.append({"source": source, "target": target, "archive": archive / target.name,
                           "old_sha256": file_sha256(target) if target.exists() else None})
        manifests.append(str(target))
    installed, archived = [], []
    try:
        for op in operations:
            target = op["target"]
            current = file_sha256(target) if target.exists() else None
            if current != op["old_sha256"]:
                raise RuntimeError(f"Publication target changed during staging: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                op["archive"].parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(op["archive"]))
                archived.append(op)
            os.replace(op["source"], target)
            installed.append(op)
    except Exception:
        # Rollback preserves both the previous publication and the failed new
        # exports. Never remove a whole panel directory or an immutable input.
        for op in reversed(installed):
            if op["target"].exists():
                op["source"].parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(op["target"]), str(op["source"]))
        for op in reversed(archived):
            shutil.move(str(op["archive"]), str(op["target"]))
        raise
    return {"owner": OWNER, "published_at_utc": stamp, "panel_manifests": manifests,
            "published_panels": sum(len(f["final_panels"]) for f in state["figures"]),
            "archived_files": [str(op["archive"]) for op in archived]}
