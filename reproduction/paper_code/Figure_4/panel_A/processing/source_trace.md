# Figure 4a microscopy source trace

Verified 2026-09-04. The original imported microscopy views are the QuPath
snapshots below. They are distinct from the whole-section DAPI cache used by
the former organized panel-A renderer.

| Condition | Snapshot path | SHA256 |
| --- | --- | --- |
| DMSO | `/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Data/Histology/histology_snapshots/DMSO.png` | `284b4661d6c1448b8a4ce01bc4a90dfdc49974cf05cee0ea8164c2f87744acca` |
| RARi | `/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Data/Histology/histology_snapshots/RAR.png` | `951188d7134fbc709576859bac86033ea2924d793d1f6d5f7e92eb139902d06e` |

Both snapshots are 2171 × 1290 pixels, RGBA. Their QuPath inset, scale bar,
and cursor-status overlays occur outside the interior microscopy regions
imported into Figure 4a. The artwork-to-snapshot registration is recorded
separately in the panel's region lock; cropping and rotation must follow that
lock rather than choosing a new field.

## Saved QuPath channel display settings

Source:
`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Data/Histology/Metadata/merscope-slide5-RARiduodenum_01.vsi - 40x.qpdata`

SHA256: `d8eab2b226b55b11874a7f47a3919198b517703a4aab3da7eabb44239f1fdff1`

This Java-serialized QuPath 0.5.1 file contains the following JSON in its
`qupath.lib.display.ImageDisplay` property. All four channels are selected.

| Channel index | Saved name | Display color | Display minimum | Display maximum |
| --- | --- | --- | ---: | ---: |
| 0 | DAPI 50% (C1) | `#0000FF` | 833.3605 | 7756.6626 |
| 1 | FITC CD8 (C2) | `#00FF00` | 1917.8932 | 4830.508 |
| 2 | Cy5 E cadherin(C3) (C3) | `#FF00FF` | 693.67444 | 4508.884 |
| 3 | TRITC .1 (C4) | `#FFE000` | 1237.5775 | 2277.2673 |

The TRITC marker identity is not expanded beyond its saved channel name here.
These are saved QuPath display settings, not model normalization parameters.
The metadata alone does not prove the exact display state at the instant each
snapshot was captured. The preserved snapshot pixel values are therefore the
direct evidence for the imported images' colors. Later artwork export may
also have transformed those colors.

## Raw acquisition identity and unresolved raw coordinates

The QuPath server identifies series 0 of
`merscope-slide5-RARiduodenum_01.vsi`, with a 74262 × 87922-pixel image,
four UINT16 channels, and 40× magnification. The saved pixel calibration is
0.16247600650830532 µm in x and 0.16247796957658722 µm in y. The local raw source is:

`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Data/Histology/Image/merscope-slide5-RARiduodenum_01.vsi`

Its companion acquisition data are:

`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Data/Histology/Image/_merscope-slide5-RARiduodenum_01_/stack1/frame_t_0.ets`

The QuPath file does not expose a saved viewport origin or ROI annotation for
these snapshots. Cursor coordinates visible in the snapshot status overlay are
not viewport bounds. Registration attempts against the raw DAPI cache did not
produce a reliable fit and were rejected. No raw acquisition ROI coordinates
are asserted or frozen on that basis.

The legacy `Figure4/Figure4_RARi_zoom_underlying_IF/` export uses a separately
specified region, different RGB mappings, per-crop percentile limits, and
gamma correction. Those settings are not evidence for the exact Figure 4a
imports and must not substitute for the snapshot registration.

No model inference, fitting, or modification of source microscopy was performed
for this trace.
