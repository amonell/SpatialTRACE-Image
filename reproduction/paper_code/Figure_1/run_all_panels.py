#!/usr/bin/env python3
"""Regenerate all Figure 1 data/image components; manual artwork is excluded."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script

HOME=Path('/home/amonell/Desktop/spatial_axes_manuscript');SATA=Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript');HERE=Path(__file__).resolve().parent


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--graph-input-dir',type=Path,required=True)
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--panels',nargs='+',choices=list('ABCDE'),default=list('ABCDE'))
    parser.add_argument('--dry-run',action='store_true',help='Print exact commands without reading or changing inputs/outputs.')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepared-only',action='store_true',help='Explicitly select the default: render existing frozen prepared inputs.')
    mode.add_argument('--prepare-only',action='store_true',help='Run explicit preparation only, without rendering panels.')
    args=parser.parse_args();out=args.output_root.expanduser().resolve();graph=args.graph_input_dir.expanduser().resolve();stored=out/'stored_outputs/figure1_inputs';flag=['--overwrite'] if args.overwrite else []

    def run(command:list[str])->None:
        script=Path(command[1])
        panel=next((part[-1] for part in script.parts if part.startswith('panel_')),None)
        processing='figure_script' not in script.parts
        if panel is not None and panel not in args.panels:
            return
        if (processing and not args.prepare_only) or (not processing and args.prepare_only):
            return
        print(shlex.join(command),flush=True)
        if not args.dry_run:
            subprocess.run(command,check=True)
    region_lock=HERE/'panel_D/processing/figure1d_region_lock.json'; registry=HOME/'frozen_preprocessing/frozen_model_registry.json'; vit_checkpoint=HOME/'Figure3/train_model/outputs/production_representation_model_v2/best_crypt_villus_vit_model.pt'; vit_predictions=stored/'panel_D_training_input_parity_v1'
    inference_contract=HERE/'panel_D/processing/figure1d_inference_contract.json'
    crop_pipeline=Path('/home/amonell/Desktop/Goldrath_Collaborations/giovanni/2026_04_07_IF_Processing_Pipeline')
    run([sys.executable,str(HERE/'dataset_processing/build_figure1_inputs.py'),'--graph-input-dir',str(graph),'--day90-dapi','/mnt/synology/Naive/Spatial_Paper/Xenium_outputs_and_Alex_analysis_Spatial_Paper/timecourse_replicates/day90_SI_r2/xenium_output/morphology_mip.ome.tif','--peyer-predictions',str(SATA/'frozen_preprocessing/peyer_label_repair_v1/gat/gat_peyer_predictions.csv'),'--vit-cells-manifest',str(HOME/'Figure3/derived/manifests/all_supervised_cells.csv'),'--panel-d-region-lock',str(region_lock),'--rari-predictions',str(SATA/'Figure_4/stored_outputs/scientific_repair_v1/evaluation/whole_tissue/rari_predictions.parquet'),'--frozen-registry',str(registry),'--output-dir',str(stored),*flag])
    run([str(crop_pipeline/'.venv/bin/python'),str(HERE/'panel_D/processing/run_frozen_image_model.py'),'--source-manifest',str(HOME/'Figure3/derived/manifests/source_manifest.csv'),'--cells-csv',str(stored/'day90_inference_vit_cells.csv'),'--region-lock',str(region_lock),'--inference-contract',str(inference_contract),'--checkpoint',str(vit_checkpoint),'--expected-checkpoint-sha256','d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137','--package-root','/home/amonell/Desktop/crypt-villus-vit','--pipeline-root',str(crop_pipeline),'--output-dir',str(vit_predictions),'--batch-size','64','--device','cuda',*flag])
    run(['/home/amonell/mambaforge/bin/python',str(HERE/'panel_E/processing/extract_if_multichannel_crop.py'),'--image',str(SATA/'Data/Histology/Image/merscope-slide5-RARiduodenum_01.vsi'),'--companion',str(SATA/'Data/Histology/Image/_merscope-slide5-RARiduodenum_01_/stack1/frame_t_0.ets'),'--output-npy',str(stored/'rari_if_multichannel_rgb.npy'),'--metadata',str(stored/'rari_if_multichannel_rgb_metadata.json'),'--x-min','17000','--x-max','23000','--y-min','48000','--y-max','52500','--pixel-size-um','0.16247698804244627',*flag])
    shared=['--section-cells',str(graph/'section_cells.parquet'),'--zoom-cells',str(graph/'day90_zoom_cells.parquet')]
    source_regions=HERE/'assembly/source_region_lock.json'
    run([sys.executable,str(HERE/'panel_A/figure_script/make_panel_A_components.py'),*shared,'--peyer-dapi',str(stored/'day90_peyer_dapi.npy'),'--figure1-input-manifest',str(stored/'figure1_input_manifest.json'),'--source-region-lock',str(source_regions),'--output-dir',str(out/'panel_A'),*flag])
    peyer_inputs=out/'stored_outputs/peyer_repair_v1'; peyer_args=['--peyer-cells',str(peyer_inputs/'peyer_region_cells.parquet'),'--peyer-manifest',str(peyer_inputs/'manifest.json')]
    run([sys.executable,str(HERE/'panel_B/figure_script/make_panel_B_components.py'),'--zoom-cells',str(graph/'day90_zoom_cells.parquet'),'--sparse-labels',str(graph/'sparse_labels.parquet'),'--source-region-lock',str(source_regions),*peyer_args,'--output-dir',str(out/'panel_B'),*flag])
    run([sys.executable,str(HERE/'panel_C/figure_script/make_panel_C_components.py'),'--sparse-labels',str(graph/'sparse_labels.parquet'),'--training-graph-cells',str(stored/'day90_training_graph_cells.parquet'),'--training-region-lock',str(region_lock),*peyer_args,'--frozen-registry',str(HOME/'frozen_preprocessing/frozen_model_registry.json'),'--output-dir',str(out/'panel_C'),*flag])
    run([sys.executable,str(HERE/'panel_D/figure_script/make_panel_D_components.py'),'--training-dapi',str(stored/'day90_training_dapi.npy'),'--training-graph-cells',str(stored/'day90_training_graph_cells.parquet'),'--inference-dapi',str(stored/'day90_inference_dapi.npy'),'--image-model-predictions',str(vit_predictions/'predictions.csv'),'--image-model-provenance',str(vit_predictions/'provenance.json'),'--inference-contract',str(inference_contract),'--region-lock',str(region_lock),'--input-manifest',str(stored/'figure1_input_manifest.json'),'--frozen-registry',str(registry),'--output-dir',str(out/'panel_D'),*flag])
    repaired_if=SATA/'Figure_4/stored_outputs/scientific_repair_v1/evaluation'
    run([sys.executable,str(HERE/'panel_E/figure_script/make_panel_E_components.py'),'--if-rgb-crop',str(stored/'rari_if_multichannel_rgb.npy'),'--if-crop-metadata',str(stored/'rari_if_multichannel_rgb_metadata.json'),'--rari-zoom-predictions',str(repaired_if/'figure1e/rari_zoom_predictions.parquet'),'--release-manifest',str(repaired_if/'release_manifest.json'),'--frozen-registry',str(HOME/'frozen_preprocessing/frozen_model_registry.json'),'--output-dir',str(out/'panel_E'),*flag])
    print(f'Figure 1: {"dry run" if args.dry_run else "preparation completed" if args.prepare_only else "components regenerated"}: {out}')


if __name__=='__main__':main()
