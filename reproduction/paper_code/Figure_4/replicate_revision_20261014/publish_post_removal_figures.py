#!/usr/bin/env python3
"""Audit the post-removal Figure 4 pages and publish only after visual review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import sys

from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from figure_assembly.compositor import file_sha256
from figure_assembly.current_figures import refresh_current_figures
from figure_assembly.publish import publish_panels
from figure_organization import load_catalog


def record(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def compare(old, new):
    with Image.open(old) as a, Image.open(new) as b:
        # Compare alpha too: transparent panel layers are the final import assets.
        identical = a.size == b.size and all(
            band.getbbox() is None
            for band in ImageChops.difference(a.convert("RGBA"), b.convert("RGBA")).split())
    return {"previous": record(old), "new": record(new), "pixel_identical": identical}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--publish-visually-reviewed", action="store_true")
    args = parser.parse_args()
    report_path = args.run / ("visual_qa_and_publication.json" if args.publish_visually_reviewed
                              else "scoped_render_audit.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    state = json.loads((args.run / "complete_figure_run.json").read_text())
    if state["status"] != "AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED":
        raise RuntimeError("The complete-page run has not passed automated QA")
    if Path(state["output_root"]).resolve() != args.run.resolve():
        raise ValueError("Run path mismatch")
    if {f["package"] for f in state["figures"]} != {"Figure_4", "Figure_4_Extended"}:
        raise ValueError("Unexpected figure scope")
    pointer = json.loads((Path(__file__).with_name("active_release.json")).read_text())
    comparisons = []
    for figure in state["figures"]:
        package = figure["package"]
        unchanged = "ABCD" if package == "Figure_4" else "ABCDF"
        qa = json.loads((args.run / "QA" / package / "qa.json").read_text())
        if qa["errors"]:
            raise RuntimeError(qa["errors"])
        provenance_path = args.run / (Path(figure["pdf"]).stem + "_provenance.json")
        provenance = json.loads(provenance_path.read_text())
        if provenance["panel_recomposition"]["status"] != "PIXEL_IDENTICAL":
            raise RuntimeError("Panel layers do not reproduce the full page")
        prepared = provenance["preparation"]
        if prepared["model_inference_run"]:
            raise RuntimeError("Figure rendering must not rerun inference")
        for source in prepared["inputs"]:
            if file_sha256(source["path"]) != source["sha256"]:
                raise RuntimeError(f"Input changed after rendering: {source['path']}")
        cohort = prepared["details"]["replicate_cohort"]
        for key, pointer_key in (("cohort_manifest", "manifest"),
                                 ("statistics_manifest", "statistics_manifest"),
                                 ("display_manifest", "display_manifest")):
            if cohort[key] != pointer[pointer_key]:
                raise RuntimeError("Rendered data do not match the active release")
        if (cohort["n_cells"] != 1568177 or cohort["n_p14"] != 43424 or
                cohort["mice_per_condition"] != {"DMSO": 4, "RARi": 5} or
                not cohort["manual_removals_applied"]):
            raise RuntimeError("Unexpected retained cohort")
        if package == "Figure_4":
            summary = prepared["details"]["replicate_regional_summary"]
            if summary["endpoint"] != "observed_over_expected_p14_density":
                raise RuntimeError("Incorrect endpoint")
            if len(summary["displayed_statistics"]) != 5:
                raise RuntimeError("Missing gate statistics")
        for letter in "ABCDEF":
            relative = Path("panels") / package / f"panel_{letter}_final.png"
            comparison = compare(args.previous_run / relative, args.run / relative)
            if comparison["pixel_identical"] != (letter in unchanged):
                raise RuntimeError(f"Unexpected changed/unchanged panel: {package} {letter}")
            comparisons.append({"package": package, "panel": letter, **comparison})
        for output in provenance["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("Figure export changed after QA")
    report = {
        "status": "SCOPED_RENDER_CHECKS_PASSED", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "code": record(Path(__file__)),
        "comparisons": comparisons, "active_release": pointer,
        "changed_panels": ["Figure_4 E: filtered observed/expected means, differences and Holm-adjusted P",
                           "Figure_4 F: filtered P14 mean densities and mean gate percentages",
                           "Figure_4_Extended E: filtered non-P14 mean densities and mean gate percentages"],
        "unchanged_panels": "Main A–D and Extended A–D/F are pixel-identical to the previous published run.",
        "manuscript_documents_modified": False,
    }
    if args.publish_visually_reviewed:
        catalog = load_catalog()
        root = Path(catalog["output_root"])
        current = root / "Regenerated_Figures/Current"
        before = {p.name: file_sha256(p) for p in current.iterdir()}
        report["visual_review"] = {"status": "REVIEWED", "items": [
            "Both complete pages inspected; panel spacing and original microscopy fields retained",
            "Main E inspected at enlarged scale: updated endpoint, labels, SEM and adjusted P legible",
            "Both P14 and non-P14 IMAPs inspected; percentages fit their registered labels",
            "Automated font, bounds, anchor, raster/vector and recomposition checks passed"]}
        report["panel_publication"] = publish_panels(state, root, overwrite=True)
        report["current_figures"] = refresh_current_figures(root, catalog["figures"])
        after = {p.name: file_sha256(p) for p in current.iterdir()}
        if len(after) != 8 or set(before) != set(after):
            raise RuntimeError("Current must contain exactly the eight figure PDFs")
        if {n for n in before if before[n] != after[n]} != {"Figure4.pdf", "Figure4_extended.pdf"}:
            raise RuntimeError("Unexpected changes to other current figures")
        report["current_hashes_before"] = before
        report["current_hashes_after"] = after
        report["status"] = "REVIEWED_AND_PUBLISHED"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
