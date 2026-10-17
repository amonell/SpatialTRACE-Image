# Figure 3A transformer-vector repair

The assembled `reaudit_final_2026_09_05/Figure3.pdf` showed rear transformer-card
outlines through the front faces. This was an assembly error: the previous
area-based partition sent each large card fill to the background and every
card stroke to the foreground. The original PDF correctly interleaves each
fill and stroke so that subsequent front cards cover earlier rear edges.

## Bounded change

- Preserve original paint events 3974–3987 and 4323–4336 together, in original
  order, in the panel-A-only `manual_transformer_stacks.pdf` vector layer.
  Each group contains six fill/stroke pairs and the two inner operation boxes.
- Remove those same events from both general manual layers to avoid duplicate
  painting. No shapes were reconstructed, moved, rescaled, or rasterized.
- Replace the overlong adjacent label with `Scale embeddings`, in 5.1-point
  Arial at the original baseline. The label fits the original 43.389-point
  text footprint. Both `Shared transformer × 6` labels remain unchanged.
- Lock source operation sequences and group bounds in `manual_path_lock.json`.
  Validate exact extracted operations, path counts, and geometry before use.

The original source remains unchanged, SHA256
`af288a503dd65070919e1638b36bd4e153ce72ffd70125c576d8b703a98217b5`.
`figure_inputs.py`, scientific rendering code, frozen values, masks,
microscopy, inference outputs, and all prediction overlays were untouched.

## Checks and temporary proof

Proof: `/tmp/figure3-transformer-repair.z02JDp/complete.pdf`, with 300-dpi
`complete.png`. The original, broken, and repaired magnified views are
`original_blocks.png`, `current_blocks.png`, and `repaired_blocks.png` in the
same directory. The repaired view restores the original rear-card occlusion,
leaves the operation labels unobstructed, and clears the scale-label overlap.

Comparison with the latest assembled baseline at 300 dpi found 16,871 changed
pixels. All changes lie inside the two transformer stack regions or the
scale-embedding label; zero changed pixels occur elsewhere on the page.
The allowed diagnostic regions, in top-left PDF points, were
`[403.6,49.9,73.8,47.8]`, `[403.6,97.4,73.8,47.8]`, and `[352,47,66,8]`.

Four focused regression tests pass, including a rendered-pixel check that
front-card fills conceal rear edges, and failures on missing, reordered, or
changed source paths. All ten shared assembly tests also pass. The full-page
font, text, anchor, private-content, and raster checks report no errors.
Composition from the separately generated final panel layers is pixel-identical
to direct full-page composition at the mandatory 150-dpi check.

Reproduce the focused tests with:

```bash
python3 -B -m unittest Figure_3.assembly.test_transformer_art -v
```

This proof has not been published to the canonical SATA output tree. The
adjacent assembly README also corrects its stale teacher-augmentation sentence
to match the already repaired frozen student/teacher protocol.
