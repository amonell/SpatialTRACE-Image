from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from paper_paths import Path, lock_record, rendering_script
from zoneinfo import ZoneInfo

import matplotlib as mpl

CODE_ROOT = Path("/home/amonell/Desktop/spatial_axes_manuscript")
OUTPUT_ROOT = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript")
LEGACY_FIGURE2_ROOT = CODE_ROOT / "Figure2"
FIGURE_OUTPUT_ROOT = OUTPUT_ROOT / "Figure_2"


def configure_figure_fonts() -> None:
    """Apply the manuscript-wide typography policy before any panel is drawn."""
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
        }
    )


configure_figure_fonts()

GRAPH_COMPONENT_MANIFEST_SHA256 = "0f989ad916276a0117730c1ac83849052b0e559470ba0e21af28e1862fe62a01"


def sha256(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: str | Path, expected_sha256: str | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if expected_sha256:
        observed = sha256(resolved)
        if observed != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {resolved}: expected {expected_sha256}, observed {observed}"
            )
    return resolved


def require_graph_inputs(manifest_path: str | Path, paths: list[Path]) -> dict:
    """Validate selected compact inputs against the reviewed frozen manifest.

    Only the requested panel inputs are opened. Byte-identical copies may live
    in a separate prepared directory; output-directory choice is independent.
    """
    manifest = json.loads(require_file(manifest_path, GRAPH_COMPONENT_MANIFEST_SHA256).read_text())
    records = {Path(item["path"]).name: item for item in manifest["outputs"]}
    for path in paths:
        if Path(path).name not in records:
            raise RuntimeError(f"Input is absent from the frozen graph manifest: {path}")
        require_file(path, records[Path(path).name]["sha256"])
    return manifest


def protect_outputs(paths: list[Path], overwrite: bool) -> bool:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(map(str, existing))
        raise FileExistsError(f"Outputs already exist; pass --overwrite:\n{joined}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return bool(existing)


def stage_file(
    source: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
    overwrite: bool,
) -> dict[str, object]:
    source_path = require_file(source, expected_sha256)
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    copied_hash = sha256(destination_path)
    if copied_hash != expected_sha256:
        raise RuntimeError(f"Copy verification failed for {destination_path}")
    return {
        "source": str(source_path),
        "source_sha256": expected_sha256,
        "output": str(destination_path),
        "output_sha256": copied_hash,
        "bytes": destination_path.stat().st_size,
    }


def git_commit(path: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_record(path: str | Path, role: str) -> dict[str, object]:
    resolved = require_file(path)
    return {
        "role": role,
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def command_text() -> str:
    return shlex.join([sys.executable, *sys.argv])


def write_provenance(
    path: str | Path,
    *,
    script: str | Path,
    inputs: list[dict[str, object]],
    outputs: list[str | Path],
    seeds: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    extra: dict[str, object] | None = None,
    overwritten: bool = False,
) -> None:
    output_records = [file_record(output, role="panel_output") for output in outputs]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "command": command_text(),
        "script": file_record(script, role="rendering_script"),
        "code_root": str(CODE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "git_commit": git_commit(CODE_ROOT),
        "python": platform.python_version(),
        "packages": package_versions(
            ("numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "pyarrow")
        ),
        "inputs": inputs,
        "outputs": output_records,
        "seeds": seeds or {},
        "warnings": warnings or [],
        "overwrote_existing_output": overwritten,
    }
    if extra:
        payload.update(extra)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
