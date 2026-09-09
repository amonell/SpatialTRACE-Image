import pandas as pd

from crypt_villus_vit.gates import assign_axis_epithelial_gates


def test_assign_axis_epithelial_gates():
    table = pd.DataFrame(
        {
            "predicted_axis_coordinate": [0.8, 0.8, 0.1, 0.1, 0.1],
            "predicted_epithelial_distance_clipped_1p0": [0.1, 0.4, 0.1, 0.4, 0.7],
        }
    )
    out = assign_axis_epithelial_gates(table)
    assert out["predicted_gate_name"].tolist() == [
        "Top IE",
        "Top LP",
        "Crypt IE",
        "Crypt LP",
        "Muscularis",
    ]
