# Examples

Run `tissuemapper-image create-demo --output-dir runs/demo` to generate the complete synthetic tutorial, including microscopy, source and cell tables, annotations, and prespecified section splits. No download is needed. See the repository README for tested training and inference commands.

For real images, adapt the small templates below. Paths in `sources.csv` are relative to that CSV. These rows describe a schema, not real data; replace them before running inference. Cell coordinates are full-resolution pixels and pixel size is micrometres per pixel.

The previous static demo and Conda environment are superseded by the generated tutorial and locked uv installation.
