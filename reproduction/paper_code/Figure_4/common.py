"""Shared publication helpers for Figure 4 and Extended Figure 4."""
from paper_paths import Path, lock_record, rendering_script

from Figure_3.common import (  # noqa: F401
    CODE_ROOT,
    OUTPUT_ROOT,
    configure_figure_fonts,
    file_record,
    git_commit,
    protect_outputs,
    require_file,
    sha256,
    write_provenance,
)


def save_pair(figure, stem: Path, dpi: int, *, pad: float = 0.02) -> list[Path]:
    """Preserve vector artists; raster layers must use the requested export DPI.

    Matplotlib otherwise saves rasterized PDF scatter layers at its default
    100 dpi even when the companion PNG was explicitly exported at 300 dpi.
    """
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight", pad_inches=pad, dpi=dpi)
    figure.savefig(png, bbox_inches="tight", pad_inches=pad, dpi=dpi)
    return [pdf, png]
