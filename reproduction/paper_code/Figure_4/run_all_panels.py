#!/usr/bin/env python3
"""Regenerate Figure 4 data/image components from frozen production-v2 outputs."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script

HOME = Path("/home/amonell/Desktop/spatial_axes_manuscript")
SATA = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")
HERE = HOME / "Figure_4"


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print explicit plotting commands without running them")
    parser.add_argument("--prepared-only", action="store_true", help="Compatibility flag: every active component already uses frozen inputs")
    parser.add_argument("--panels", nargs="+", choices=list("ABCDEF"), default=list("ABCDEF"))
    args = parser.parse_args()
    out = args.output_root.expanduser().resolve()
    flag = ["--overwrite"] if args.overwrite else []
    python = sys.executable
    def run_if_selected(command):
        panel = Path(command[1]).parent.parent.name.removeprefix("panel_")
        if panel in args.panels:
            if args.dry_run:
                print(shlex.join(command), flush=True)
            else:
                run(command)

    repair = SATA / "Figure_4/stored_outputs/scientific_repair_v1"
    source = repair / "evaluation"
    release_flags = ["--release-manifest", str(source / "release_manifest.json")]
    dmso, rari = source / "whole_tissue/dmso_predictions.parquet", source / "whole_tissue/rari_predictions.parquet"
    checkpoint = repair / "stage2_full/best_crypt_villus_vit_model.pt"
    image = HOME / "Figure4/imap_if/outputs/image_cache/figure4_if_series0_channel0_uint16.npy"
    manifest = repair / "supervised_manifest.csv"
    snapshots = SATA / "Data/Histology/histology_snapshots"
    style = SATA / "Figures/Old_Figures/2026_09_04_Figures/Figure4.png"
    run_if_selected([python, str(HERE / "panel_A/figure_script/make_panel_A_if_images.py"), "--dmso-image", str(snapshots / "DMSO.png"), "--rari-image", str(snapshots / "RAR.png"), "--region-lock", str(HERE / "panel_A/processing/figure4a_region_lock.json"), "--scale-calibration", str(HERE / "panel_A/processing/figure4a_scale_calibration.json"), "--style-reference", str(style), "--output-dir", str(out / "panel_A"), *flag])
    run_if_selected([python, str(HERE / "panel_B/figure_script/make_panel_B_sparse_if_labels.py"), "--manifest", str(manifest), "--image-cache", str(image), "--checkpoint", str(checkpoint), "--calibration-config", str(repair / "config.json"), "--example-manifest", str(repair / "illustration_inputs/manifest.json"), "--output-dir", str(out / "panel_B"), *release_flags, *flag])
    run_if_selected([python, str(HERE / "panel_C/figure_script/make_panel_C_strict_test_scatter.py"), "--predictions", str(source / "strict_test/predictions.parquet"), "--metrics", str(source / "strict_test/metrics.json"), "--split-manifest", str(SATA / "splits/if_final_strict_test_v1.json"), "--checkpoint", str(checkpoint), "--output-dir", str(out / "panel_C"), *release_flags, *flag])
    run_if_selected([python, str(HERE / "panel_D/figure_script/make_panel_D_whole_tissue_maps.py"), "--dmso-predictions", str(dmso), "--rari-predictions", str(rari), "--checkpoint", str(checkpoint), "--zoom-config", str(HOME / "Figure4/imap_if/config_all_cells_production_v2_if_sft.yaml"), "--output-dir", str(out / "panel_D"), *release_flags, *flag])
    run_if_selected([python, str(HERE / "panel_E/figure_script/make_panel_E_abundance_composition.py"), "--abundance", str(source / "imap/p14_abundance_by_gate.tsv"), "--composition", str(source / "imap/p14_composition_by_gate.tsv"), "--checkpoint", str(checkpoint), "--output-dir", str(out / "panel_E"), *release_flags, *flag])
    run_if_selected([python, str(HERE / "panel_F/figure_script/make_panel_F_p14_imaps.py"), "--dmso-predictions", str(dmso), "--rari-predictions", str(rari), "--gate-definitions", str(source / "imap/p14_gate_definitions.json"), "--gate-percentages", str(source / "imap/p14_composition_by_gate.tsv"), "--density-manifest", str(source / "imap/density_manifest.json"), "--checkpoint", str(checkpoint), "--style-reference", str(style), "--output-dir", str(out / "panel_F"), *release_flags, *flag])
    print(f"Figure 4 components {'planned' if args.dry_run else 'regenerated'}: {out}")


if __name__ == "__main__":
    main()
