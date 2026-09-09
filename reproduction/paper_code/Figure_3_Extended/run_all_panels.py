#!/usr/bin/env python3
"""Render active A–H components, including the approved matched retraining panel."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script

HOME = Path("/home/amonell/Desktop/spatial_axes_manuscript")
SATA = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")
HERE = HOME / "Figure_3_Extended"
LEGACY = HOME / "Figure3"
PEYER = HOME / "Peyers_Pipeline"
PANEL_ORDER = tuple("ABCDEFGH")
AVAILABLE_PANELS = tuple("ABCDEFGH")


def selected_panels(values: list[str] | None) -> list[str]:
    if not values:
        return list(PANEL_ORDER)
    tokens = [token.upper() for value in values for token in value.replace(",", " ").split()]
    unknown = set(tokens) - set(AVAILABLE_PANELS)
    if unknown:
        raise ValueError(f"Unknown panel(s): {sorted(unknown)}; use A–H")
    return [panel for panel in AVAILABLE_PANELS if panel in tokens]


def command_plan(output: Path, *, python: str, overwrite: bool, prepared_root: Path | None = None) -> dict[str, list[str]]:
    """Construct explicit commands without reading or resolving any data file."""
    flag = ["--overwrite"] if overwrite else []
    checkpoint = LEGACY / "train_model/outputs/production_representation_model_v2/best_crypt_villus_vit_model.pt"
    paired = LEGACY / "train_model/outputs/paired_representation_pretraining"
    production = LEGACY / "train_model/outputs/production_representation_model_v2"
    region = SATA / "Figure_3/stored_outputs/panel_A/panel_A_exact_historical_region.npz"
    region_metadata = SATA / "Figure_3/stored_outputs/panel_A/panel_A_exact_historical_region.json"
    paired_views = prepared_root or SATA / "Figure_3_Extended/stored_outputs/paired_view_repair_v1"
    classifier_repair = SATA / 'frozen_preprocessing/peyer_representation_v1'
    thresholds = classifier_repair / 'image_windows'
    classifier_root = classifier_repair / 'vit'
    classifier_checkpoint = classifier_root / "full_manual_sparse/best_tissueframe_peyer.pt"
    image = Path("/mnt/synology/Naive/Spatial_Paper/Xenium_outputs_and_Alex_analysis_Spatial_Paper/timecourse_replicates/day90_SI_r2/xenium_output/morphology_mip.ome.tif")
    def command(panel: str, script: str, *arguments: object) -> list[str]:
        return [python, str(HERE / f"panel_{panel}/figure_script/{script}"), *map(str, arguments),
                "--output-dir", str(output / f"panel_{panel}"), *flag]
    return {
        "A": command("A", "make_panel_A_paired_pretraining_views.py", "--region-data", region,
                     "--region-metadata", region_metadata, "--pretraining-summary", paired / "pretraining/pretrain_summary.json", "--seed", "7",
                     "--paired-views", paired_views / "paired_views.npz", "--paired-manifest", paired_views / "manifest.json"),
        "B": command("B", "make_panel_B_pretraining_history.py", "--history", paired / "pretraining/pretrain_history.csv",
                     "--summary", paired / "pretraining/pretrain_summary.json"),
        "C": command("C", "make_panel_C_supervised_history.py", "--history", production / "history.csv",
                     "--selection", production / "production_selection.json", "--checkpoint", checkpoint),
        "D": command("D", "make_panel_D_pretraining_ablation.py", "--metrics", paired / "metrics/performance_by_seed.csv",
                     "--run-summary", paired / "run_summary.json"),
        "E": command("E", "make_panel_E_branch_input_shuffling.py", "--deltas", production / "scale_importance/scale_importance_deltas.csv",
                     "--summary", production / "scale_importance/scale_importance_summary.json", "--checkpoint", checkpoint),
        "G": command("G", "make_panel_G_peyer_prediction.py", "--predictions", thresholds / "display_window_fixed_threshold.csv",
                     "--summary", thresholds / "fixed_threshold_window_summary.json", "--threshold-manifest", thresholds / "manifest.json", "--image", image, "--checkpoint", classifier_checkpoint),
        "H": command("H", "make_panel_H_peyer_classifier_strategy.py", "--summary", classifier_root / "strategy_metrics.tsv",
                     "--checkpoint", classifier_checkpoint, "--release-manifest", classifier_root / 'sparse_evaluation_completed.json',
                     "--roc-manifest", classifier_repair / 'roc/manifest.json'),
        "F": command("F", "make_panel_F_matched_branch_retraining.py", "--release",
                     SATA / 'frozen_preprocessing/model_design_comparisons_v1/image_branches_evaluated_v1/release.json'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, help="Existing paired_view_repair_v1 directory for panel A; independent of --output-root.")
    parser.add_argument("--panels", nargs="+", help="Active panels A–H, including matched branch retraining")
    parser.add_argument("--dry-run", action="store_true", help="Print selected explicit commands without opening inputs or writing outputs")
    parser.add_argument("--prepared-only", action="store_true", help="Uniform-runner flag; every component already consumes frozen/prepared inputs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        panels = selected_panels(args.panels)
    except ValueError as error:
        parser.error(str(error))
    # Selection precedes command execution or any data-file I/O. No analysis,
    # inference, pretraining, split construction, or ablation is invoked here.
    output = args.output_root.expanduser().absolute()
    prepared_root = args.prepared_root.expanduser().absolute() if args.prepared_root else None
    commands = command_plan(output, python=sys.executable, overwrite=args.overwrite, prepared_root=prepared_root)
    for panel in panels:
        command = commands[panel]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    verb = "Planned" if args.dry_run else "Rendered"
    print(f"{verb} Extended Figure 3 panels {','.join(panels)}: {output}")


if __name__ == "__main__":
    main()
