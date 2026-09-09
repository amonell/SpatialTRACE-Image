"""Prepare fresh fixed-layout components, retaining each renderer's input QA."""
from __future__ import annotations

import hashlib
import json
import os
from paper_paths import Path, lock_record, rendering_script
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SATA = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")


def record(path, role):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "role": role}


def prepare_package(package, output_dir, overwrite=False):
    output_dir = Path(output_dir).resolve()
    layout_path = REPO / package / "assembly/layout.json"
    layout = json.loads(layout_path.read_text())
    reference = Path(layout["reference_pdf"])
    reference_record = record(reference, "reviewed_layout_reference_only")
    if reference_record["sha256"] != layout["reference_sha256"]:
        raise RuntimeError("Artwork changed after the fixed-layout review")
    if package in {"Figure_2", "Figure_2_Extended"}:
        from Figure_2.common import require_graph_inputs
        compact_manifest = SATA / "Figure_2/stored_outputs/shared_graph_components/graph_component_inputs_manifest.json"
        manifest = require_graph_inputs(compact_manifest, [])
        require_graph_inputs(compact_manifest, [Path(source["path"]) for source in manifest["outputs"]])
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = REPO / package / "run_all_panels.py"
    plan = [sys.executable, str(runner), "--output-root", str(output_dir), "--dry-run", "--prepared-only"]
    if package == "Figure_2_Extended_2":
        plan += ["--prepared-root", str(SATA / package / "stored_outputs/corrected_labels_v1")]
    elif package == "Figure_2":
        plan += ["--prepared-root", str(SATA / package / "stored_outputs/shared_graph_components")]
    elif package == "Figure_3_Extended":
        plan += ["--prepared-root", str(SATA / package / "stored_outputs/paired_view_repair_v1")]
    if overwrite:
        plan += ["--overwrite"]
    result = subprocess.run(plan, check=True, capture_output=True, text=True)
    commands = []
    env = dict(os.environ, MPLCONFIGDIR="/tmp/spatial_axes_assembly_mpl", PYTHONDONTWRITEBYTECODE="1")
    wrapper = REPO / "Figure_2/assembly/fixed_render.py"
    for line in result.stdout.splitlines():
        tokens = shlex.split(line)
        if len(tokens) < 2 or not tokens[1].endswith(".py"):
            continue
        if "figure_script" not in tokens[1]:
            raise RuntimeError("Assembly preparation may only invoke rendering scripts")
        command = [sys.executable, str(wrapper), str(layout_path), str(rendering_script(tokens[1])), *tokens[2:]]
        commands.append(command)
        subprocess.run(command, check=True, env=env)
    elements = list(layout["elements"])
    registration_path = layout_path.with_name("source_registration.json")
    registration = json.loads(registration_path.read_text())
    if registration["reference_sha256"] != layout["reference_sha256"]:
        raise RuntimeError("Source geometry measurements belong to another artwork version")
    inputs = [reference_record, record(layout_path, "absolute_coordinate_layout"), record(wrapper, "fixed_export_adapter"),
              record(registration_path, "measured_original_image_CTMs_and_vector_spines")]
    anchors = []
    for path in sorted(output_dir.glob("panel_*/*.geometry.json")):
        anchors.extend(json.loads(path.read_text())["rendered_anchors"])
    actual = {item["id"]: item["rect_pt"] for item in anchors}
    differences = []
    targets = {item['id']: item['rect_pt'] for item in layout['anchors']}
    reflow = layout.get('authorized_reflow')
    if reflow and (not reflow.get('user_request') or not reflow.get('date')):
        raise RuntimeError('A revised layout must record the explicit user authorization')
    for expected in layout.get("source_anchors", []):
        observed = actual[expected["id"]]
        error = max(abs(a-b) for a,b in zip(observed, expected["rect_pt"], strict=True))
        target = targets[expected['id']] if reflow else expected['rect_pt']
        target_error = max(abs(a-b) for a,b in zip(observed, target, strict=True))
        differences.append({"id":expected["id"], "source":expected["source"],
            "original_rect_pt":expected["rect_pt"], "rendered_rect_pt":observed, "max_error_pt":error,
            "approved_target_rect_pt":target, "target_error_pt":target_error})
        if target_error > .1:
            raise RuntimeError(f"Rendered axis moved relative to its approved target: {expected['id']} ({target_error:.4f} pt)")
    geometry_report = output_dir / "source_geometry_comparison.json"
    geometry_report.write_text(json.dumps({"reference_sha256":layout["reference_sha256"],
        "measurements":differences, "maximum_error_pt":max((x["max_error_pt"] for x in differences),default=0),
        "authorized_reflow":reflow,
        "maximum_target_error_pt":max((x['target_error_pt'] for x in differences), default=0),
        "measurement":"Original geometry remains recorded; an explicitly authorized reflow uses separately registered target anchors. All rendered targets must agree within 0.1 pt."},indent=2)+"\n")
    inputs.append(record(geometry_report,"actual_vs_original_geometry_QA"))
    for path in sorted(output_dir.glob("panel_*/*provenance.json")):
        provenance = json.loads(path.read_text())
        inputs.extend(provenance.get("inputs", []))
        inputs.append(record(path, "fresh_component_provenance"))
    unique = {(item["path"], item.get("sha256")): item for item in inputs}
    return {"elements": elements, "inputs": list(unique.values()), "commands": commands,
            "rendered_anchors": anchors, "warnings": layout.get("warnings", []),
            "intentional_differences": layout.get("intentional_differences", [])}
