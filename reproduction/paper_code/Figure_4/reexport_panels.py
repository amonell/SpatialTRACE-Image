#!/usr/bin/env python3
"""Archive and rerender specified main/extended Figure 4 components, not artwork."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from paper_paths import Path, lock_record, rendering_script

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from Figure_4.common import file_record, sha256
from figure_organization import load_catalog, ordered_components, panel_map
from scientific_release_gate import unresolved_for


def inventory(paths):
    return {str(path): sha256(path) for path in paths if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", choices=list("ABCDEFG"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    output_project = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")
    allowed = {output_project / "Figure_4": "Figure 4",
               output_project / "Figure_4_Extended": "Extended Figure 4"}
    if root not in allowed:
        raise ValueError(f"This archival wrapper requires one of: {list(allowed)}")
    load_catalog()
    active = [panel["letter"] for panel in panel_map(root.name)]
    if set(args.panels) - set(active):
        raise ValueError(f"Only final visible panels {active} are active in {root.name}")
    selected = [letter for letter in active if letter in args.panels]
    # Check scientific inputs before moving any active exports into an archive.
    if not (root.name == 'Figure_4' and set(selected) == {'A'}):
        blockers = unresolved_for([root.name], output_project)
        if blockers:
            raise RuntimeError('No components were archived or changed.\n' + '\n'.join(blockers))
    if not args.overwrite and any((root / f"panel_{p}").exists() for p in selected):
        raise FileExistsError("Pass --overwrite to archive and replace active components")
    artwork_paths = list((root.parent / "Figures").glob("Figure4*"))
    untouched = [p for letter in "ABCDEFG" if letter not in selected
                 for p in (root / f"panel_{letter}").rglob("*") if p.is_file()]
    untouched.extend(p for p in root.glob("panel_*/panel_*_final.*") if p.is_file())
    untouched.extend(root.glob("panel_*/panel_*_final_provenance.json"))
    before = inventory([*artwork_paths, *untouched])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive = root / "polished_panel_archive" / f"{timestamp}_before_{''.join(selected)}_revision"
    archive.mkdir(parents=True, exist_ok=False)
    archived = []
    for letter in selected:
        panel = root / f"panel_{letter}"
        if panel.exists():
            # Archive only declared component files. Final layers and artist
            # subdirectories must survive a component-only refresh.
            mapping = next(p for p in panel_map(root.name) if p['letter'] == letter)
            stems = [c['stem'] if isinstance(c, dict) else c for c in mapping['components']]
            if root.name == 'Figure_4':
                stems.extend({'E': ['panel_E_P14_abundance_and_composition', 'panel_E_P14_IMAPs'],
                              'F': ['panel_F_P14_imaps', 'panel_F_P14_enrichment'],
                              'G': ['panel_G_P14_IMAPs', 'panel_G_P14_abundance']}.get(letter, []))
            if root.name == 'Figure_4_Extended' and letter == 'E':
                stems.append('panel_E_non_P14_imaps')  # archived two-mouse component
            names = {f'panel_{letter}_provenance.json', f'panel_{letter}_metrics.json',
                     'assembly_inputs_provenance.json'}
            names.update(stem + '_provenance.json' for stem in stems)
            if root.name == 'Figure_4' and letter == 'D':
                names.add('panel_D_zoom_regions.tsv')
            owned = [p for p in panel.iterdir() if p.is_file() and
                     (p.name in names or any(p.name.startswith(stem+'.') for stem in stems))]
            for path in owned:
                destination = archive / panel.name / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                archived.append(file_record(path, 'previous_panel_component'))
                shutil.move(str(path), str(destination))
    for name in ("component_index.tsv", "component_contact_sheet.png", "QA.md"):
        path = root / name
        if path.exists():
            archived.append(file_record(path, "previous_component_index"))
            shutil.copy2(path, archive / name)
    archive_record = archive / "archive_manifest.json"
    archive_record.write_text(json.dumps({"archived_at": timestamp, "files": archived,
                                         "archive": str(archive), "panels": selected}, indent=2) + "\n")
    print(f"Previous components preserved in {archive}", flush=True)
    env = {**os.environ, "MPLCONFIGDIR": "/tmp/figure4_component_matplotlib", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, str(REPO / root.name / "run_all_panels.py"),
               "--output-root", str(root), "--panels", *selected]
    subprocess.run(command, check=True, env=env)
    if inventory([*artwork_paths, *untouched]) != before:
        raise RuntimeError("An unselected panel or assembled artwork changed")
    missing_components = [(letter, root / f'panel_{letter}' / f'{stem}.pdf')
                          for letter, stem in ordered_components(root.name)
                          if not (root / f'panel_{letter}' / f'{stem}.pdf').is_file()]
    missing_selected = [str(path) for letter, path in missing_components if letter in selected]
    if missing_selected:
        raise FileNotFoundError(f'Missing regenerated components: {missing_selected}')
    warnings = []
    if missing_components:
        # A scoped update must not regenerate unrelated panels to satisfy an
        # old whole-figure component index. Keep that index unchanged and
        # report the missing components explicitly; selected PDFs still get
        # the checks below. Canonical final layers are separate exports.
        warnings.append('Whole-figure component index retained: unselected standalone components '
                        'are missing: ' + ', '.join(str(path) for _, path in missing_components))
        print(warnings[-1], flush=True)
    else:
        subprocess.run([sys.executable, str(REPO / "finalize_organized_figure_components.py"),
                        "--figure", allowed[root], "--output-root", str(root), "--overwrite"], check=True, env=env)
    pdf_checks = {}
    for letter in selected:
        for pdf in sorted((root / f"panel_{letter}").glob("*.pdf")):
            pdf_checks[str(pdf)] = {
                tool: subprocess.run([tool, *(["-list"] if tool == "pdfimages" else []), str(pdf)],
                                     check=True, capture_output=True, text=True).stdout
                for tool in ("pdfinfo", "pdffonts", "pdfimages")}
    # A vector-only IMAP must not contain hidden bitmap point clouds or colorbars.
    if root.name == "Figure_4" and "F" in selected:
        image_lines = pdf_checks[str(root / "panel_F/panel_F_P14_IMAPs.pdf")]["pdfimages"].splitlines()[2:]
        if any(line.strip() for line in image_lines):
            raise RuntimeError("P14 IMAP PDF unexpectedly contains raster images")
    if root.name == "Figure_4_Extended" and "B" in selected:
        image_lines = pdf_checks[str(root / "panel_B/panel_B_final_lineage_transfer.pdf")]["pdfimages"].splitlines()[2:]
        if any(line.strip() for line in image_lines):
            raise RuntimeError("Transfer comparison PDF unexpectedly contains raster images")
    if root.name == "Figure_4_Extended" and "E" in selected:
        pdf = root / 'panel_E/panel_E_non_P14_IMAPs.pdf'
        records = [line.split() for line in pdf_checks[str(pdf)]['pdfimages'].splitlines()[2:] if line.strip()]
        if sum(row[2] == 'image' for row in records) != 2:
            raise RuntimeError(f'{pdf}: expected the two frozen non-P14 density rasters')
    if root.name == "Figure_4_Extended" and "F" in selected:
        for condition in ("DMSO", "RARi"):
            pdf = root / f"panel_F/panel_F_peyer_classifier_{condition}.pdf"
            records = [line.split() for line in pdf_checks[str(pdf)]["pdfimages"].splitlines()[2:] if line.strip()]
            if len(records) != 1 or records[0][2] != "image" or min(int(records[0][3]), int(records[0][4])) < 4000:
                raise RuntimeError(f"{pdf}: expected only the native DAPI raster; prediction points/colorbar must be vector")
    report = {
        "created_at": datetime.now().isoformat(), "panels": selected, "archive": str(archive),
        "unchanged_artwork_and_unselected_panels": before, "new_inference": False,
        "new_model_fitting": False, "source_predictions_immutable": True,
        "outputs": [file_record(p, "regenerated_component") for letter in selected
                    for p in sorted((root / f"panel_{letter}").glob("*")) if p.is_file()],
        "pdf_preflight": pdf_checks,
        "figure_package": root.name,
        "component_index_refreshed": not missing_components,
        "warnings": warnings,
    }
    report_path = root / f"revision_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Revision record: {report_path}", flush=True)
    load_catalog()


if __name__ == "__main__":
    main()
