#!/usr/bin/env python3
"""Regenerate every Figure 2a-e data component from frozen numeric outputs."""
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
    parser.add_argument("--prepared-root", type=Path, help="Existing shared_graph_components directory; independent of the output directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--panels", nargs="+", choices=list("ABCDE"), default=list("ABCDE"))
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepared-only", action="store_true", help="Explicitly select the default: use frozen prepared inputs.")
    mode.add_argument("--prepare-only", action="store_true", help="Prepare shared inputs without rendering panels.")
    args = parser.parse_args()
    out = args.output_root.expanduser().resolve()
    default_prepared = (out / "stored_outputs/shared_graph_components" if args.prepare_only
                        else SATA / "Figure_2/stored_outputs/shared_graph_components")
    compact = (args.prepared_root or default_prepared).expanduser().resolve()
    flag = ["--overwrite"] if args.overwrite else []

    def run(command: list[str]) -> None:
        script = Path(command[1])
        panel = next((part[-1] for part in script.parts if part.startswith("panel_")), None)
        processing = "figure_script" not in script.parts
        if panel is not None and panel not in args.panels:
            return
        if processing and not set(args.panels).intersection("ABC"):
            return
        if (processing and not args.prepare_only) or (not processing and args.prepare_only):
            return
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    run([sys.executable, str(HERE / "dataset_processing/build_graph_component_tables.py"),
         "--h5ad", "/mnt/synology/Naive/Spatial_Paper/Xenium_outputs_and_Alex_analysis_Spatial_Paper/timecourse_replicates/analysis/cleaned/full_xenium_replicates_and_reference_no_peyers.h5ad",
         "--labels", str(HOME / "Figure2/labelme_parse/parsed_villus_labels.csv"),
         "--crypt-predictions", str(HOME / "Figure2/train_models/outputs/crypt_villus.predictions.npy"),
         "--epithelial-predictions", str(HOME / "Figure2/train_models/outputs/epithelial_distance.predictions.npy"),
         "--expected-labels-sha256", "6eabc5f269efcb8c2a263b44016404aa8e34900fa49a1cca1a7b2f447d63be2c",
         "--expected-crypt-sha256", "89f3f115349118bb74a1ef5eaa6669dfb935c9d73fefeb1ac8c91af9c4a5c5e0",
         "--expected-epithelial-sha256", "cc6cdc767b97ead0b40596fa8273ce9638ba463534487e209dcf8b5fef40b5aa",
         "--output-dir", str(compact), "--seed", "42", *flag])
    common = ["--section-cells", str(compact / "section_cells.parquet"), "--input-manifest", str(compact / "graph_component_inputs_manifest.json")]
    run([sys.executable, str(HERE / "panel_A/figure_script/make_panel_A_components.py"), *common,
         "--graph-nodes", str(compact / "graph_example_nodes.parquet"), "--graph-edges", str(compact / "graph_example_edges.tsv"), "--output-dir", str(out / "panel_A"), *flag])
    peyer = SATA / "Figure_2_Extended_2/stored_outputs/corrected_labels_v1"
    run([sys.executable, str(HERE / "panel_A/figure_script/make_panel_A_schematic_insets.py"),
         "--zoom-cells", str(compact / "day90_zoom_cells.parquet"),
         "--sparse-labels", str(compact / "sparse_labels.parquet"),
         "--input-manifest", str(compact / "graph_component_inputs_manifest.json"),
         "--peyer-cells", str(peyer / "panel_c_window_cells.parquet"),
         "--peyer-manifest", str(peyer / "prepared_inputs_manifest.json"),
         "--output-dir", str(out / "panel_A"), *flag])
    run([sys.executable, str(HERE / "panel_B/figure_script/make_panel_B_components.py"), *common,
         "--sparse-labels", str(compact / "sparse_labels.parquet"), "--zoom-cells", str(compact / "day90_zoom_cells.parquet"), "--output-dir", str(out / "panel_B"), *flag])
    run([sys.executable, str(HERE / "panel_C/figure_script/make_panel_C_components.py"), *common,
         "--zoom-cells", str(compact / "day90_zoom_cells.parquet"), "--frozen-registry", str(HOME / "frozen_preprocessing/frozen_model_registry.json"), "--output-dir", str(out / "panel_C"), *flag])
    run([sys.executable, str(HERE / "panel_D/figure_script/make_panel_D_section_disjoint_scatter.py"),
         "--predictions", str(SATA / "Figure2/section_disjoint_evaluation/predictions.parquet"), "--metrics", str(SATA / "Figure2/section_disjoint_evaluation/metrics.json"),
         "--split-manifest", str(SATA / "splits/graph_model_section_disjoint_oof_v1.json"), "--checkpoint-manifest", str(SATA / "Figure2/section_disjoint_evaluation/fold_checkpoint_manifest.tsv"),
         "--bootstrap-metrics", str(SATA / "Figure2/section_disjoint_evaluation/section_bootstrap_metrics.tsv"), "--output-dir", str(out / "panel_D"), *flag])
    scaling = SATA / "Figure2/section_disjoint_villus_scaling_v1"
    run([sys.executable, str(HERE / "panel_E/figure_script/make_panel_E_training_villus_scaling.py"),
         "--metrics", str(scaling / "per_fold_metrics.tsv"),
         "--split-manifest", str(SATA / "splits/graph_model_section_disjoint_villus_scaling_v1.json"),
         "--expected-split-sha256", "e1779c3caab90164eea05ba38d2dc93c3432ad845711fd7428f281bd52eaa0f8",
         "--checkpoint-manifest", str(scaling / "checkpoint_manifest.tsv"),
         "--output-dir", str(out / "panel_E"), *flag])
    print(f"Figure 2: {'dry run' if args.dry_run else 'preparation completed' if args.prepare_only else 'components regenerated'}: {out}")


if __name__ == "__main__":
    main()
