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
DEFAULT_DATA_RELEASE = OUTPUT_ROOT / 'Figure_3/stored_outputs/training_matched_inputs_v1/manifest.json'


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
        raise RuntimeError(f"SHA256 mismatch for {resolved}: expected {expected_sha256}, observed {sha256(resolved)}")
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


def save_pair(figure, stem: Path, dpi: int, *, pad: float = 0.02) -> list[Path]:
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight", pad_inches=pad, dpi=dpi)
    figure.savefig(png, bbox_inches="tight", pad_inches=pad, dpi=dpi)
    return [pdf, png]


def data_release(manifest_path: Path = DEFAULT_DATA_RELEASE) -> dict:
    """Resolve the authorized training-matched snapshot; never fall back to raw crops."""
    if Path(manifest_path).resolve() != DEFAULT_DATA_RELEASE:
        raise RuntimeError('Figure 3 requires the current training-matched release')
    lock = json.loads((CODE_ROOT / 'Figure_3/assembly/input_lock.json').read_text())['inputs']
    expected = lock_record(lock, DEFAULT_DATA_RELEASE).get('sha256')
    if expected is None:
        raise RuntimeError('Training-matched preprocessing has not been frozen and activated')
    manifest = json.loads(require_file(manifest_path, expected).read_text())
    for flag in ('cell_membership_changed', 'embedding_membership_changed', 'reference_values_changed', 'zoom_changed', 'model_refitted'):
        if manifest.get(flag) is not False:
            raise RuntimeError(f'Training-matched release does not preserve {flag}')
    if (manifest.get('status') != 'VERIFIED' or manifest.get('cells') != 50000 or manifest.get('umap_cells') != 30000
            or manifest.get('input_protocol') != 'direct_pyramid_crop_normalize_resize_uint8_v1'
            or manifest.get('checkpoint', {}).get('sha256') != 'd0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137'):
        raise RuntimeError('Invalid frozen Figure 3 training-matched release')
    return manifest


def repaired_reference(path: Path, manifest_path: Path, output_name: str) -> Path:
    """Load a frozen output with unchanged current graph references and cell identities."""
    manifest = data_release(manifest_path)
    record = manifest["outputs"][output_name]
    resolved = require_file(path, record["sha256"])
    if resolved != Path(record["path"]).resolve():
        raise RuntimeError("Reference path differs from the frozen repair manifest")
    return resolved


def git_commit() -> str | None:
    result = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_provenance(path: Path, *, script: Path, inputs: list[dict[str, object]], outputs: list[Path], extra: dict[str, object], overwritten: bool) -> None:
    packages = {}
    for name in ("numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "pyarrow"):
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
