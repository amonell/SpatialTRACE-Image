from pathlib import Path

from crypt_villus_vit.qupath import load_qupath_centroid_rows


def test_load_qupath_centroid_rows_converts_microns(tmp_path: Path):
    table = tmp_path / "cells.csv"
    table.write_text("centroid_x_um,centroid_y_um,classification\n32.5,65.0,P14\nbad,3,P14\n")
    rows, summary = load_qupath_centroid_rows(
        qupath_input_path=table,
        source_id="if:test",
        class_filter="P14",
        pixel_size_um=0.325,
    )
    assert len(rows) == 1
    assert rows.loc[0, "centroid_x_fullres_px"] == 100.0
    assert rows.loc[0, "centroid_y_fullres_px"] == 200.0
    assert summary.dropped_non_finite_count == 1
