# Shared Figure 2 plotting inputs

`build_graph_component_tables.py` reads the immutable H5AD, sparse labels, and
two frozen all-label graph prediction arrays once, then writes compact tables
for panels A-C. It does not train, infer, select, or evaluate a model.
