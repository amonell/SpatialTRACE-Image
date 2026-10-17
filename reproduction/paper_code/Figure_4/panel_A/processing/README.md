# Original Figure 4a microscopy regions

The old exporter used whole-section grayscale DAPI selected by prediction-table
bounds. That did not reproduce either multichannel villus close-up. The actual
originals are SATA `Data/Histology/histology_snapshots/DMSO.png` and `RAR.png`,
registered to the original `Figures/Figure4.png`, now preserved at
`Figures/Old_Figures/2026_09_04_Figures/Figure4.png`. The immutable region lock
records its location at registration time; active commands use the archived
path with the same required image hash. Current panel order comes from the
root `Figures/Figure4.pdf`, not this older style/region image.

`figure4a_region_lock.json` hash-locks sources and artwork. DMSO uses snapshot
x=539:1407, y=242:1080 and 180-degree rotation; RARi uses x=481:1350,
y=144:981 without rotation. Median landmark residuals are below one source
pixel. These are snapshot coordinates, not raw VSI pixel coordinates.

`audit_artwork_registration.py` reproduces the landmark audit (scikit-image,
fixed seed 0). Pass explicit `--artwork`, `--snapshots`, and a new `--output-dir`.
It does not change the lock. The diagnostic affine RGB coefficients in its
report are **not** applied to final images.

Rendering applies the locked crop and original orientation to the actual
microscopy snapshots. No whole-dataset inference, new normalization, channel
recoloring, or copying from assembled-figure artwork occurs. Native snapshot
pixels are preserved in PNG and PDF without artificial upsampling.

## Scale bars

`figure4a_scale_calibration.json` (SHA256
`7acc25d5ae4bbdb3e4562aa91166ca67f50ded5798850eb046db6c8bcceec4e8`)
records the original QuPath 50 µm scale bar in both snapshots. Its native
stroke is exactly x=10:89, y=1257:1260 (79×3 pixels); the visible label reads
50 µm. The renderer rechecks these source pixels before adding a 79-pixel-long
50 µm bar to each unresized crop. This inherits the original snapshot scale,
without claiming independently verified raw-VSI calibration.

Bars occupy an appended black footer, never the microscopy region. Native
RGB crop pixels are still preserved exactly. PDF bar and 8.5-pt Arial label are
vectors; the PNG companion also includes them. Scale image and bar together
when placing the component in Illustrator. Pass the calibration explicitly
with `--scale-calibration`; the orchestration supplies its locked path.

The artwork's muted colors differ from original RGB PNGs; none has an embedded
ICC profile. A standard SWOP roundtrip looks similar, but the historical export
transform is unverified and is not applied. Exact post-export color matching
remains pending original artwork color settings. See `source_trace.md` for the
separately recovered QuPath channel colors and display limits.
