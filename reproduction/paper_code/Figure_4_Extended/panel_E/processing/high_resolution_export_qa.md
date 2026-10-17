# Extended Figure 4e resolution QA

Verified 2026-09-04 using the preview in
`/tmp/figure4extended_d_high_resolution_20260904/`.

- Previous active PDF: scatter layers 196 × 150 and 201 × 148 pixels at 100 ppi.
- New PDF: scatter layers 1174 × 893 and 1196 × 888 pixels at 600 ppi.
- `pdfimages -list` reports 600 ppi for every embedded image; none remain 100 ppi.
- `pdfinfo`: one page, no encryption, 485.802 × 202.968 points.
- `pdffonts`: embedded Arial and Arial Bold TrueType fonts.
- The PDF was rendered back to PNG at 300 dpi and visually inspected. Cell
  points and contours are sharp; external gate-percentage rails do not cover
  cells, and labels and colorbars are not clipped.
- `py_compile` passed for the updated rendering script.

The same SHA256-locked production-v2 prediction tables retain 154,888 DMSO and
134,443 RARi non-P14 cells. All ten displayed gate percentages reproduce the
frozen `Figure4/production_v2_final/imap/non_p14_gate_fractions.tsv` to within
3.6 × 10⁻¹⁵ percentage points (floating-point rounding only). No model inference,
new cell selection, gate change, or density-method change occurred. The data
layers remain dense rasterized scatters; labels, axes, contours, and gates
remain vector elements.

Preview PDF SHA256:
`c66b77971010bc452af85c4c8bfe780d70bdef94312c2f3e70f06505f8f346f6`.

This verification does not claim that the manually assembled Extended Figure 4
has been replaced. Promotion to SATA is a separate archived export step.

## Display-placement update

The historical export check above predates the user's 5 September request to
restore original-artwork gate-label positions. Current exports use that exact
registration, including on-plot labels, while retaining the 600-dpi point
layers. See `Figure_4/assembly/display_update_review.md` for the current check.
