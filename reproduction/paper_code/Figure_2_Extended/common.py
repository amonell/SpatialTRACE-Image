from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from paper_paths import Path, lock_record, rendering_script
from zoneinfo import ZoneInfo

import matplotlib as mpl

CODE_ROOT = Path("/home/amonell/Desktop/spatial_axes_manuscript")
OUTPUT_ROOT = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")


def configure_figure_fonts() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
        }
    )


configure_figure_fonts()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: str | Path, expected_sha256: str | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if expected_sha256 and sha256(resolved) != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {resolved}")
    return resolved


def file_record(path: str | Path, role: str) -> dict[str, object]:
    resolved = require_file(path)
    return {"role": role, "path": str(resolved), "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def protect_outputs(paths: list[Path], overwrite: bool) -> bool:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite:\n" + "\n".join(map(str, existing)))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return bool(existing)


def save_pair(figure, stem: Path, dpi: int) -> list[Path]:
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    figure.savefig(pdf, dpi=dpi, bbox_inches="tight")
    figure.savefig(png, bbox_inches="tight", dpi=dpi)
    return [pdf, png]


def git_commit() -> str | None:
    result = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_provenance(path: Path, *, script: Path, inputs: list[dict[str, object]], outputs: list[Path], extra: dict[str, object], overwritten: bool) -> None:
    packages = {}
    for name in ("numpy", "pandas", "matplotlib", "scipy", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "script": file_record(script, "rendering_script"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": packages,
        "font_family": "Arial",
        "inputs": inputs,
        "outputs": [file_record(output, "panel_output") for output in outputs],
        "overwrote_existing_output": overwritten,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
