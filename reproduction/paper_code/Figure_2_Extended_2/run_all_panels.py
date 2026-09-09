#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--panels", nargs="+", choices=list("ABCD"), default=list("ABCD"))
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepared-only", action="store_true", help="Use existing hash-locked prepared inputs.")
    mode.add_argument("--prepare-only", action="store_true", help="This runner only renders; prepared inputs are produced by dataset_processing.")
    args = parser.parse_args()
    prepared = args.prepared_root.expanduser().resolve()
    out = args.output_root.expanduser().resolve()
    manifest = prepared / "prepared_inputs_manifest.json"
    prepared_info = json.loads(manifest.read_text())
    flag = ["--overwrite"] if args.overwrite else []

    def run(command: list[str]) -> None:
        panel = next(part[-1] for part in Path(command[1]).parts if part.startswith("panel_"))
        if args.prepare_only or panel not in args.panels:
            return
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    run([
        sys.executable,
        str(HERE / "panel_A/figure_script/make_panel_A_training_seeds.py"),
        "--background", str(prepared / "panel_a_background.parquet"),
        "--training-seeds", str(prepared / "panel_a_training_seeds.parquet"),
        "--prepared-manifest", str(manifest),
        "--output-dir", str(out / "panel_A"),
        *flag,
    ])
    run([
        sys.executable,
        str(HERE / "panel_B/figure_script/make_panel_B_training_history.py"),
        "--history", str(prepared / "panel_b_history.tsv"),
        "--prepared-manifest", str(manifest),
        "--output-dir", str(out / "panel_B"),
        *flag,
    ])
    run([
        sys.executable,
        str(HERE / "panel_C/figure_script/make_panel_C_heldout_window.py"),
        "--cells", str(prepared / "panel_c_window_cells.parquet"),
        "--morphology", str(prepared_info["prepared_outputs"]["panel_c_morphology"]["path"]),
        "--window-metadata", str(prepared / "panel_c_window_metadata.json"),
        "--prepared-manifest", str(manifest),
        "--output-dir", str(out / "panel_C"),
        *flag,
    ])
    run([
        sys.executable,
        str(HERE / "panel_D/figure_script/make_panel_D_heldout_performance.py"),
        "--metrics", str(prepared / "panel_d_section_metrics.tsv"),
        "--prepared-manifest", str(manifest),
        "--output-dir", str(out / "panel_D"),
        *flag,
    ])
    print(f"Extended Figure 2 page 2: {'dry run' if args.dry_run else 'use explicit dataset_processing preparation command' if args.prepare_only else 'components regenerated'}: {out}")


if __name__ == "__main__":
    main()
