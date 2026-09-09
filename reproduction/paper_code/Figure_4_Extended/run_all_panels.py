#!/usr/bin/env python3
"""Regenerate Extended Figure 4 components from frozen release outputs."""
from __future__ import annotations
import argparse, shlex, subprocess, sys
from paper_paths import Path, lock_record, rendering_script

HOME=Path('/home/amonell/Desktop/spatial_axes_manuscript'); SATA=Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript'); HERE=HOME/'Figure_4_Extended'
def run(cmd): print(' '.join(map(str,cmd)),flush=True); subprocess.run(list(map(str,cmd)),check=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--overwrite',action='store_true'); p.add_argument('--panels',nargs='+',choices=list('ABCDEF'),default=list('ABCDEF'),help='Active panels A–F, including fixed gate backprojection.'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--prepared-only',action='store_true',help='Compatibility flag: all active components already use frozen outputs'); a=p.parse_args(); out=a.output_root.resolve(); flag=['--overwrite'] if a.overwrite else []; py=sys.executable
    if 'A' in a.panels:
        print('Panel A: manual IF fine-tuning workflow; no image/data component to regenerate', flush=True)
    def run_if_selected(command):
        panel=Path(command[1]).parent.parent.name.removeprefix('panel_')
        if panel in a.panels:
            if a.dry_run:
                print(shlex.join(list(map(str,command))), flush=True)
            else:
                run(command)
    ifroot=SATA/'Figure_4/stored_outputs/scientific_repair_v1'; source=ifroot/'evaluation'; release=['--release-manifest',source/'release_manifest.json']; ifck=ifroot/'stage2_full/best_crypt_villus_vit_model.pt'; xenck=HOME/'Figure3/train_model/outputs/production_representation_model_v2/best_crypt_villus_vit_model.pt'; split=SATA/'splits/if_final_strict_test_v1.json'; image=HOME/'Figure4/imap_if/outputs/image_cache/figure4_if_series0_channel0_uint16.npy'; dmso=source/'whole_tissue/dmso_predictions.parquet'; rari=source/'whole_tissue/rari_predictions.parquet'
    run_if_selected([py,HERE/'panel_B/figure_script/make_panel_B_final_transfer.py','--metrics',source/'transfer_comparison/final_lineage_transfer_metrics.tsv','--bootstrap',source/'transfer_comparison/grouped_bootstrap_metrics.tsv','--xenium-checkpoint',xenck,'--if-checkpoint',ifck,'--split-manifest',split,'--output-dir',out/'panel_B',*release,*flag])
    run_if_selected([py,HERE/'panel_C/figure_script/make_panel_C_if_training_history.py','--stage1-history',ifroot/'stage1_heads/history.csv','--stage1-summary',ifroot/'stage1_heads/training_summary.json','--stage2-history',ifroot/'stage2_full/history.csv','--stage2-summary',ifroot/'stage2_full/training_summary.json','--checkpoint',ifck,'--output-dir',out/'panel_C',*release,*flag])
    run_if_selected([py,HERE/'panel_D/figure_script/make_panel_D_gate_backprojection.py','--manifest',SATA/'Figure_4_Extended/stored_outputs/panel_F/gate_backprojection_v1/manifest.json','--output-dir',out/'panel_D',*flag])
    run_if_selected([py,HERE/'panel_E/figure_script/make_panel_E_non_p14_imaps.py','--dmso-predictions',dmso,'--rari-predictions',rari,'--gate-definitions',source/'imap/p14_gate_definitions.json','--gate-percentages',source/'imap/non_p14_counts_and_fractions.tsv','--density-manifest',source/'imap/density_manifest.json','--checkpoint',ifck,'--style-reference',SATA/'Figures/Old_Figures/2026_09_04_Figures/Figure4_extended.png','--output-dir',out/'panel_E',*release,*flag])
    peyer_repair=SATA/'frozen_preprocessing/peyer_representation_v1'
    common=['--predictions',peyer_repair/'image_windows/if_predictions/predictions.csv','--summary',peyer_repair/'image_windows/if_predictions/summary.json','--image-cache',image,'--checkpoint',peyer_repair/'vit/full_manual_sparse/best_tissueframe_peyer.pt','--release-manifest',peyer_repair/'image_windows/manifest.json']
    run_if_selected([py,HERE/'panel_F/figure_script/make_panel_F_peyer_conditions.py',*common,'--output-dir',out/'panel_F',*flag])
    print(f"Extended Figure 4 components {'planned' if a.dry_run else 'regenerated'}: {out}")
if __name__=='__main__': main()
