#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script

HOME = Path("/home/amonell/Desktop/spatial_axes_manuscript")
SATA = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--panels", nargs="+", choices=list("ABCDEF"), default=list("ABCDEF"))
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepared-only", action="store_true", help="All panels already use frozen inputs; no preprocessing is run.")
    mode.add_argument("--prepare-only", action="store_true", help="No preparation is required for this package; do not render.")
    args = parser.parse_args()
    out = args.output_root.expanduser().resolve()
    shared = SATA / "Figure_2/stored_outputs/shared_graph_components"
    curve = SATA / "Figure2/section_disjoint_villus_scaling_v1"
    split = SATA / "splits/graph_model_section_disjoint_villus_scaling_v1.json"
    flag = ["--overwrite"] if args.overwrite else []

    def run(command: list[str]) -> None:
        panel = next(part[-1] for part in Path(command[1]).parts if part.startswith("panel_"))
        if args.prepare_only or panel not in args.panels:
            return
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    run([sys.executable, str(HERE / "panel_A/figure_script/make_panel_A_label_prediction_maps.py"), "--section-cells", str(shared / "section_cells.parquet"), "--sparse-labels", str(shared / "sparse_labels.parquet"), "--frozen-registry", str(HOME / "frozen_preprocessing/frozen_model_registry.json"), "--zoom-manifest", str(SATA / "Figure_2_Extended/stored_outputs/panel_A/prediction_zoom_v1/manifest.json"), "--reference-zoom-manifest", str(SATA / "Figure_2_Extended/stored_outputs/panel_A/reference_zoom_v1/manifest.json"), "--output-dir", str(out / "panel_A"), *flag])
    benchmark = SATA / 'Figure_2_Extended/stored_outputs/runtime_benchmark_v1'
    run([sys.executable, str(HERE / "panel_B/figure_script/make_panel_B_runtime.py"), "--runtime-records", str(benchmark / 'runtime_records.tsv'), '--benchmark-manifest', str(benchmark / 'completed.json'), "--output-dir", str(out / "panel_B"), *flag])
    run([sys.executable, str(HERE / "panel_C/figure_script/make_panel_C_heldout_mae.py"), "--metrics", str(curve / "per_fold_metrics.tsv"), '--split-manifest', str(split), "--output-dir", str(out / "panel_C"), *flag])
    run([sys.executable, str(HERE / "panel_D/figure_script/make_panel_D_heldout_r2.py"), "--metrics", str(curve / "per_fold_metrics.tsv"), '--split-manifest', str(split), "--output-dir", str(out / "panel_D"), *flag])
    run([sys.executable, str(HERE / "panel_E/figure_script/make_panel_E_graph_components.py"), "--release", str(SATA / "frozen_preprocessing/model_design_comparisons_v1/graph/release_components/manifest.json"), "--output-dir", str(out / "panel_E"), *flag])
    run([sys.executable, str(HERE / "panel_F/figure_script/make_panel_F_annotation_budgets.py"), "--release", str(SATA / "frozen_preprocessing/model_design_comparisons_v1/graph/release_all/manifest.json"), "--output-dir", str(out / "panel_F"), *flag])
    print(f"Extended Figure 2: {'dry run' if args.dry_run else 'no preparation required' if args.prepare_only else 'components regenerated'}: {out}")


if __name__ == "__main__":
    main()
