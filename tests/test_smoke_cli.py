from pathlib import Path

import numpy as np

from crypt_villus_vit.cli import main


def test_predict_qupath_smoke(tmp_path: Path):
    image = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
    image_path = tmp_path / "demo.npy"
    np.save(image_path, image)
    source_manifest = tmp_path / "sources.csv"
    source_manifest.write_text(f"source_id,image_path,pixel_size_um\nif:test,{image_path},0.325\n")
    qupath_csv = tmp_path / "cells.csv"
    qupath_csv.write_text("centroid_x_fullres_px,centroid_y_fullres_px,class\n64,64,P14\n128,128,P14\n")
    checkpoint = tmp_path / "demo.pt"
    output_dir = tmp_path / "out"
    main(["create-demo-checkpoint", "--output", str(checkpoint)])
    main(
        [
            "predict-qupath",
            "--source-manifest",
            str(source_manifest),
            "--qupath-csv",
            str(qupath_csv),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
    )
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "gate_percentages.csv").exists()
    assert (output_dir / "prediction_scatter.png").exists()
