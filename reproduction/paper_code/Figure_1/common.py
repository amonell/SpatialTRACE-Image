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
FIGURE_OUTPUT_ROOT = OUTPUT_ROOT / "Figure_1"


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


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, role: str) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"role": role, "path": str(resolved), "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def require_frozen_input(path: str | Path) -> Path:
    """Use the same reviewed numeric/image input lock as complete-page assembly."""
    path = Path(path).expanduser().resolve()
    lock = json.loads((Path(__file__).parent / "assembly/input_lock.json").read_text())["inputs"]
    if sha256(path) != lock_record(lock, path)["sha256"]:
        raise RuntimeError(f"Unlisted or changed Figure 1 frozen input: {path}")
    return path


def load_source_regions(path: str | Path) -> dict:
    """Return the artwork-registered A/B regions, shared by all export modes."""
    regions = json.loads(Path(path).read_text())
    layout = json.loads((Path(__file__).parent / "assembly/layout.json").read_text())
    if regions["artwork"]["sha256"] != layout["reference_sha256"]:
        raise RuntimeError("Figure 1 source-region lock belongs to different artwork")
    for record in regions["source_assets"]:
        if sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"Changed source-region evidence: {record['path']}")
    return regions


def verify_training_input_parity(provenance: dict, contract_path: Path) -> Path:
    """Accept only the reviewed, golden-verified Figure1D inference release."""
    require_frozen_input(contract_path)
    contract = json.loads(contract_path.read_text())
    record = next(r for r in provenance['inputs'] if r['role'] == 'physical_field_of_view_contract')
    if record['sha256'] != sha256(contract_path):
        raise RuntimeError('Figure1D inference contract differs from its frozen provenance')
    if provenance.get('checkpoint_sha256') != contract['checkpoint_sha256']:
        raise RuntimeError('Figure1D inference contract belongs to a different checkpoint')
    if provenance.get('native_crop_sizes_px') != contract['native_crop_sizes_px']:
        raise RuntimeError('Figure1D inference does not reproduce trained physical fields')
    if not provenance.get('golden_tensors_all_exact'):
        raise RuntimeError('Figure1D inference lacks exact supervised-tensor parity')
    parity_record = next(r for r in provenance['outputs'] if Path(r['path']).name == 'training_parity.json')
    path = require_frozen_input(parity_record['path'])
    if sha256(path) != parity_record['sha256']:
        raise RuntimeError('Figure1D golden parity record changed')
    parity = json.loads(path.read_text())
    count = provenance['golden_prepared_cell_count']
    if len(parity['tensor_checks']) != 3 * count or not all(r['exact'] for r in parity['tensor_checks']):
        raise RuntimeError('Figure1D supervised-tensor checks are incomplete')
    if provenance['frozen_prediction_max_abs_difference'] > 1e-5:
        raise RuntimeError('Figure1D predictions disagree with the frozen prepared evaluation')
    return path


def protect_outputs(paths: list[Path], overwrite: bool) -> bool:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite:\n" + "\n".join(map(str, existing)))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return bool(existing)


def verify_peyer_component(cells: Path, manifest_path: Path) -> dict:
    """Require corrected-label, fixed-window inputs for the two Figure 1 examples."""
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'FROZEN_CORRECTED_LABEL_FIGURE1_PEYER_WINDOW' or manifest['window_changed']:
        raise RuntimeError('Incorrect Figure 1 Peyer component lineage')
    from checkpoint_lineage import validate_graph_checkpoint_reference
    validate_graph_checkpoint_reference(manifest['checkpoint']['path'], manifest['checkpoint']['sha256'])
    for item in [manifest['cells'], manifest['label_lock']]:
        if sha256(item['path']) != item['sha256']:
            raise RuntimeError(f'Peyer component hash mismatch: {item["path"]}')
    if cells.resolve() != Path(manifest['cells']['path']).resolve():
        raise RuntimeError('Unlisted Peyer component input')
    return manifest


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in names:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def git_commit(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_provenance(path: Path, *, script: Path, inputs: list[dict[str, object]], outputs: list[Path], extra: dict[str, object], overwritten: bool) -> None:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "script": file_record(script, "rendering_script"),
        "git_commit": git_commit(CODE_ROOT),
        "python": platform.python_version(),
        "packages": package_versions(("numpy", "pandas", "matplotlib", "scipy", "pyarrow", "Pillow")),
        "inputs": inputs,
        "outputs": [file_record(output, "panel_output") for output in outputs],
        "overwrote_existing_output": overwritten,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
