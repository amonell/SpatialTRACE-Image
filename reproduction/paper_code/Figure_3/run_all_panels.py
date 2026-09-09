#!/usr/bin/env python3
"""Regenerate Figure 3 data/image components from frozen production-v2 outputs."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script

HOME = Path("/home/amonell/Desktop/spatial_axes_manuscript")
HERE = Path(__file__).resolve().parent
LEGACY = HOME / "Figure3"
PREPARED = Path("/mnt/sata2/alex_storage/Desktop/giovanni/IF/if_xenium_multiscale_vit/baseline_512_2048/vit_256_16/prepared")
IF_PIPELINE = Path("/home/amonell/Desktop/Goldrath_Collaborations/giovanni/2026_04_07_IF_Processing_Pipeline")
IF_DATA = Path("/mnt/sata3/Giovanni_storage/260401_KPC_Gna13_D7")
IF_MULTISCALE = Path("/mnt/sata2/alex_storage/Desktop/giovanni/IF/if_xenium_multiscale_vit/baseline_512_2048")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--panels", nargs="+", choices=list("ABCDE"), default=list("ABCDE"))
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepared-only", action="store_true", help="Render existing frozen inputs; do not extract microscopy")
    mode.add_argument("--prepare-only", action="store_true", help="Extract the fixed microscopy inputs, then stop before rendering")
    args = parser.parse_args()
    out = args.output_root.expanduser().resolve()
    flag = ["--overwrite"] if args.overwrite else []
    def run(command: list[str]) -> None:
        script = Path(command[1])
        processing = "processing" in script.parts
        letter = next((part.removeprefix("panel_") for part in script.parts if part.startswith("panel_")), None)
        if letter is not None and letter not in args.panels:
            return
        if processing and not args.prepare_only:
            return
        if not processing and args.prepare_only:
            return
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    checkpoint = LEGACY / "train_model/outputs/production_representation_model_v2/best_crypt_villus_vit_model.pt"
    pretraining_checkpoint = LEGACY / "train_model/outputs/paired_representation_pretraining/pretraining/dapi_encoder_pretrain.pt"
    selection = LEGACY / "train_model/outputs/production_representation_model_v2/production_selection.json"
    training_summary = LEGACY / "train_model/outputs/production_representation_model_v2/training_summary.json"
    pretraining_summary = LEGACY / "train_model/outputs/paired_representation_pretraining/pretraining/pretrain_summary.json"
    sample008 = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure3/production_sample_008_v2")
    references = Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3/stored_outputs/training_matched_inputs_v1')
    paired_root = references.parents[2] / 'Figure_3_Extended/stored_outputs/paired_view_repair_v1'
    paired_flags = ['--paired-manifest', str(paired_root/'manifest.json'), '--paired-views', str(paired_root/'paired_views.npz')]
    predictions = references / "predictions_current_references.parquet"
    ref_flags = ["--reference-manifest", str(references / "manifest.json")]
    split = sample008 / "split_manifest.json"
    stored_a = out / "stored_outputs/panel_A"
    if_python = IF_PIPELINE / ".venv/bin/python"
    run([
        str(if_python),
        str(HERE / "panel_A/processing/extract_exact_historical_region.py"),
        "--if-pipeline-src", str(IF_PIPELINE / "src"),
        "--nuclei-table", str(IF_MULTISCALE / "cache/if:cont1_if_nuclei.csv"),
        "--pretraining-manifest", str(IF_MULTISCALE / "vit_256_16/if_xenium_pretraining_manifest.csv"),
        "--pretraining-shard-dir", str(PREPARED / "pretraining_shards"),
        "--vsi", str(IF_DATA / "Image_Cont1.vsi"),
        "--sidecar-dir", str(IF_DATA / "_Image_Cont1_"),
        "--nuclei-mask", str(IF_DATA / "Cellpose_Train/full_slide_inference_cellpose_nuclei/cont1/cont1_cellpose_nuclei_labels.npy"),
        "--tissue-metadata", str(IF_DATA / "Cellpose_Train/labelme_tissue_qc_tissuenet_cp3_rf070/cont1/cont1_labelme_tissue_metadata.json"),
        "--nucleus-label-id", "352082",
        "--output-dir", str(stored_a),
        *flag,
    ])
    run([sys.executable, str(HERE / "panel_A/figure_script/make_panel_A_image_components.py"), "--region-data", str(stored_a / "panel_A_exact_historical_region.npz"), "--region-metadata", str(stored_a / "panel_A_exact_historical_region.json"), "--checkpoint", str(checkpoint), "--pretraining-checkpoint", str(pretraining_checkpoint), "--production-selection", str(selection), "--training-summary", str(training_summary), "--pretraining-summary", str(pretraining_summary), "--supervised-manifest", str(stored_a/'supervised_xenium_v1/manifest.json'), *paired_flags, "--output-dir", str(out / "panel_A"), "--seed", "7", *flag])
    run([sys.executable, str(HERE / "panel_B/figure_script/make_panel_B_embedding_umap.py"), "--umap", str(references / "figure3f_umap.csv"), "--metadata", str(references / "embedding_metadata_current_references.csv"), "--selection", str(sample008 / "panel_inputs/embedding_selection.json"), "--checkpoint", str(checkpoint), "--split-manifest", str(split), "--output-dir", str(out / "panel_B"), *ref_flags, *flag])
    run([sys.executable, str(HERE / "panel_C/figure_script/make_panel_C_coordinate_scatter.py"), "--predictions", str(predictions), "--metrics-record", str(references / "metrics.json"), "--checkpoint", str(checkpoint), "--split-manifest", str(split), "--output-dir", str(out / "panel_C"), *ref_flags, *flag])
    run([sys.executable, str(HERE / "panel_D/figure_script/make_panel_D_crypt_villus_maps.py"), "--predictions", str(predictions), "--checkpoint", str(checkpoint), "--split-manifest", str(split), "--output-dir", str(out / "panel_D"), *ref_flags, *flag])
    run([sys.executable, str(HERE / "panel_E/figure_script/make_panel_E_epithelial_distance_maps.py"), "--predictions", str(predictions), "--checkpoint", str(checkpoint), "--split-manifest", str(split), "--output-dir", str(out / "panel_E"), *ref_flags, *flag])
    print(f"Figure 3 {'commands listed' if args.dry_run else 'requested stages completed'}: {out}")


if __name__ == "__main__":
    main()
