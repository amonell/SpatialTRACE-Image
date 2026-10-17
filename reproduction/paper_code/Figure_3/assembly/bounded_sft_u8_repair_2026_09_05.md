# Figure 3A supervised-input and prediction check

Historical repair record. On 6 September the user approved replacing the
supervised side with Xenium. The Cont1 outputs documented here remain preserved
but are no longer active Figure 3A prediction components. See
`supervised_xenium_review.md` for the current source and final exports.

## Diagnosis

The released coordinate model's supervised input recipe selects the raw image
pyramid for the final 256-pixel local/context or 128-pixel fine input, applies
the tissue-removal mask, normalizes and resizes that crop, rounds to uint8,
and divides the stored bytes by 255. Nine local/context/fine arrays from three
fixed supervised cells match this recipe exactly. The alternative
512-intermediate recipe fails all six local/context comparisons. The source
generator is `prepare_supervised_shards_256_16_10x.py` under the IF pipeline's
`processing_pipelines/20_export_if_xenium_multiscale_vit_dataset/` directory.
The released training summary resolves its 92,160 supervised crops to this
direct-256 dataset.

The Cont1 **pretraining** recipe is different: normalized 512-pixel model-input
crops were rounded to uint8, resized bilinearly to 256, and rounded again.
Eight local/context arrays from four fixed nuclei, including the illustrated
nucleus, match that sequence exactly. The illustrated fine crop also matches
its original 128-pixel source. These checks do not establish a need to apply
the pretraining recipe during coordinate inference.

The existing bounded Cont1 inference had the correct physical fields and
pyramid levels but omitted the supervised uint8 roundtrip. The schematic's
two supervised thumbnails also reused the separately prepared pretraining
arrays. The correction adds the byte roundtrip to inference and saves the
actual illustrated-cell inputs for the supervised thumbnails. All pretraining
student/teacher tiles retain their correct frozen inputs.

## Frozen results

New immutable SATA snapshot:
`Figure_3/stored_outputs/complete_figure_bounded_inference_sft_u8_v2/`.
The previous `complete_figure_bounded_inference_v1/` snapshot is preserved.

The corrected run retains all 2,755 nucleus IDs and their exact centroids,
the original context bounds, the byte-identical segmentation crop, and the
released checkpoint `d0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137`.
There was no fitting, selection, source substitution, or change to main
Figure 3B–E results.

| Difference from previous saved predictions | Crypt-villus | Epithelial distance |
| --- | ---: | ---: |
| Mean absolute change | 0.000902 | 0.001253 |
| Maximum absolute change | 0.01054 | 0.01361 |
| Old-versus-new correlation | 0.999970 | 0.999966 |
| Illustrated nucleus, old | 0.203617 | 0.775808 |
| Illustrated nucleus, corrected | 0.203044 | 0.775874 |

These are numerical consistency comparisons, not accuracy estimates. There
is no independent coordinate ground truth for this illustrative Cont1 IF
field. The correction produces no evidence of improved coordinate accuracy.

The input/output hashes are locked in `input_lock.json`. In the new snapshot:

- `predictions.csv`: SHA256 `00d66ccca3ff8a4034f437df836e91cc99c1cfe5e815cde17d85b401708d2c26`.
- `illustrated_sft_inputs.npz`: SHA256 `1ea003e4076174dfd6e1bb375a2cb041311e11fc3e9cb0cc7b8300c797eda445`.
- `provenance.json`: SHA256 `4277a168c6bd27a57c8ae4b27b164f943c1826de1221cd1f46bff416bc75fb74`.

## Display and validation

The nucleus-ID join, y orientation, axis columns, and color limits are correct.
The illustrated nucleus 352082 has 382 mask pixels; its fine mask exactly
matches the corresponding region-segmentation slice. The head and regional
overlays use the same saved values and viridis mapping over 0–1. Original
artist-PDF colors lack a recovered prediction/checkpoint lineage and are not
an accuracy reference.

Both full-page and component renderers read the same saved supervised input
bytes and reject an obsolete protocol, altered input hash, changed nucleus,
or unexpected array dimensions. Five input-protocol tests and four
transformer-artwork tests pass. The full proof is generated separately under
`/tmp/figure3-transformer-repair.z02JDp/sft_u8_v2/`; canonical publication is
handled by the root runner after review.

The full proof passes page, font, text, anchor, private-content, and raster
checks. Final panel-layer recomposition is pixel-identical to the direct page.
At 300 dpi, comparison with the transformer-repaired proof finds 27,905 changed
pixels, all inside the two supervised input thumbnails and saved-prediction
slots. The fine input, all four pretraining tiles, every other part of panel A,
and panels B–E have zero changed pixels. Visual review of
`supervised_section.png` confirms clean transformer faces, readable text, and
correctly aligned input images and overlays.

```bash
python3 -B -m unittest Figure_3.assembly.test_sft_input_protocol Figure_3.assembly.test_transformer_art -v
```

Rendering reads these frozen outputs and never invokes the explicit bounded
inference script. Repeating inference requires a new output directory; the
script refuses to overwrite an existing snapshot.
