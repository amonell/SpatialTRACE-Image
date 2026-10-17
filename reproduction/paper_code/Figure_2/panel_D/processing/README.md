# Panel D processing

No additional preprocessing is performed here. The authoritative input is the locked, eight-fold section-disjoint out-of-fold prediction table produced by:

`/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure2/section_disjoint_evaluation/train_section_disjoint_oof.py`

The rendering script independently recomputes every displayed metric and verifies the prediction, metric, split, checkpoint-manifest, and grouped-bootstrap hashes before plotting.

