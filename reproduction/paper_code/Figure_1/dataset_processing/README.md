# Figure 1 compact-input processing

`build_figure1_inputs.py` performs deterministic cropping, subsetting, and
descriptive aggregation from the frozen graph, Peyer's GAT, production-v2
Xenium ViT input manifest, and production-v2 IF outputs. It performs no
inference or fitting. Figure 1d's bounded released-model inference is a
separate, explicit panel-processing step after these spatial inputs are locked.
