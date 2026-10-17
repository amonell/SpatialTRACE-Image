Panel A processing reconstructs the exact microscopy region imported into the
manual Figure 3a artwork. The locked source is Cont1 IF nucleus label 352082,
centered at full-resolution `(x=15709.439790575916, y=22534.952879581157)`.
The script extracts native 4096-, 2048-, 512-, and 128-pixel crops from the
source VSI and segmentation and loads this same center's frozen multimodal
pretraining inputs (local prepared index 2352; context prepared index 5035).
The superseded prepared-index-5220 Xenium example is not active.

The user approved a separate Xenium example for the supervised side on
6 September 2026. `supervised_xenium_source.json` locks prepared index 4540
(`sample_010_cell_0214763`) from the already verified Figure 1d Day90 region.
The center is selected solely by distance to the existing window center among
its 25 saved golden input cells. The original Cont1 pretraining region does
not change.

`prepare_supervised_xenium.py` copies the illustrated model's exact saved uint8
inputs, subsets existing verified predictions to the native 3,132-pixel field,
and extracts the corresponding native DAPI crop. It does not run a model,
change predictions or construct an evaluation split. The source contract and
manifest hash-lock the original DAPI, supervised shards and prediction corpus.
Cell-centroid dots are display markers, not segmentation masks.

One-time preparation command (the completed immutable snapshot is retained;
use a fresh output directory for an explicit re-extraction):

```bash
/home/amonell/Desktop/Goldrath_Collaborations/giovanni/2026_04_07_IF_Processing_Pipeline/.venv/bin/python -B \
  Figure_3/panel_A/processing/prepare_supervised_xenium.py \
  --source-lock Figure_3/panel_A/processing/supervised_xenium_source.json \
  --pipeline-root /home/amonell/Desktop/Goldrath_Collaborations/giovanni/2026_04_07_IF_Processing_Pipeline \
  --output-dir /mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_3/stored_outputs/panel_A/supervised_xenium_v1
```

Outputs are `images.npz`, `predictions.csv` and `manifest.json`. Any explicit
new snapshot needs source/parity review and an updated `assembly/input_lock.json`
before becoming a rendering input. Render active components with:

```bash
python -B regenerate_all_organized_figure_components.py --figures Figure_3 --panels A --overwrite
python -B update_final_figure.py Figure_3 --overwrite
```

The Day90 section is excluded from released SFT fitting and checkpoint
selection but has been examined during development. The schematic is a
qualitative inference illustration, not a fresh independent accuracy estimate.
