"""Run a figure-specific preparation adapter in an isolated Python process."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from paper_paths import Path, lock_record, rendering_script

CODE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    module_path = CODE_ROOT / args.package / "assembly/prepare.py"
    if module_path.resolve().parents[2] != CODE_ROOT:
        raise ValueError("Package must be an immediate organized figure directory")
    spec = importlib.util.spec_from_file_location(f"complete_figure_{args.package}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.result.exists():
        raise FileExistsError(args.result)
    result = module.prepare(args.output_dir.resolve(), overwrite=False)
    if not isinstance(result, dict) or not isinstance(result.get("elements"), list):
        raise TypeError("prepare() must return a dictionary containing an elements list")
    args.result.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
