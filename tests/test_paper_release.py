"""Keep the paper renderers and released weight identities fixed as the tool evolves."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_release_files_are_preserved():
    record = json.loads((ROOT / "reproduction/paper_release.json").read_text())
    for relative, expected in record["preserved_files"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative


def test_frozen_figure_code_matches_paper_manifest():
    source = ROOT / "reproduction/paper_code"
    manifest = json.loads((ROOT / "reproduction/code_manifest.json").read_text())
    assert manifest["files"]
    for item in manifest["files"]:
        path = (source / item["path"]).resolve()
        assert path.is_relative_to(source)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
