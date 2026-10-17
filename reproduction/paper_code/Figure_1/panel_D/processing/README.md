Panel D has four importable rectangles corresponding to the supplied Figure 1
artwork. Their exact Day90 replicate level-0 pixel bounds are frozen in
`figure1d_region_lock.json`. The left pair uses the five-villus training window
and frozen graph-derived dense coordinates. The right pair uses the historical
alternate Day90 DAPI window and bounded inference from the released
`paired-representation-shared-scale-v2` Xenium checkpoint.

`run_frozen_image_model.py` verifies the checkpoint SHA256, source section,
cell coordinates, center-aware architecture, and output row order before
writing predictions. It performs no training, selection, or tuning. The
superseded density-selected SI2 region is not part of Figure 1d.

The active immutable release is SATA
`Figure_1/stored_outputs/figure1_inputs/panel_D_training_input_parity_v1/`.
`figure1d_inference_contract.json` locks native local/context/fine fields of
783/3,132/196 Xenium pixels, original pyramid selection, direct 256/256/128
resizing, uint8 quantization and float32/255 model inputs. These reproduce the
actual supervised cache, which differs from the 512-to-256 pretraining cache.
All 75 tensors from the 25 overlapping prepared cells match exactly; maximum
prediction difference from the frozen prepared evaluation is 7.16e-8.

`cell_identity.py` reconstructs the documented original anonymous-ID policy,
verifies every centroid against the H5AD, and checks the production global-index
crosswalk against original prepared centers. Anonymous row IDs and prepared
`obs_index` use different numbering. `test_cell_identity.py` guards this
distinction. `compare_locked_window.py` exports a same-cell diagnostic against
frozen dense graph references; it does not produce independent manual-label
performance estimates.

The older `panel_D_production_v2_inference/` output remains intact for auditing;
it used unscaled native widths and a different raw-pixel pipeline. Rendering
now requires the golden-verified release and cannot silently fall back to it.
