#!/usr/bin/env python3
"""Run bounded released-model inference for the locked Figure 1d output region."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from paper_paths import Path, lock_record, rendering_script

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
from common import file_record, protect_outputs, sha256, write_provenance  # noqa: E402


def git_state(repository: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"repository": str(repository), "commit": commit, "dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cells-csv", type=Path, required=True)
    parser.add_argument("--region-lock", type=Path, required=True)
    parser.add_argument("--inference-contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_manifest = args.source_manifest.expanduser().resolve()
    cells_path = args.cells_csv.expanduser().resolve()
    region_lock_path = args.region_lock.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    contract_path = args.inference_contract.expanduser().resolve()
    package_root = args.package_root.expanduser().resolve()
    for path in (source_manifest, cells_path, region_lock_path, checkpoint, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("Released Xenium checkpoint SHA256 does not match the locked value.")
    contract = json.loads(contract_path.read_text())
    if contract['checkpoint_sha256'] != args.expected_checkpoint_sha256:
        raise RuntimeError('Physical-FOV contract belongs to a different checkpoint')
    for role in ('supervised_manifest', 'prepared_metadata', 'prepared_manifest', 'development_predictions',
                 'prepared_pipeline_source', 'prepared_generator_source', 'raw_dapi',
                 'anonymous_manifest_generator', 'anonymous_manifest_summary'):
        record = contract[role]
        if sha256(record['path']) != record['sha256']:
            raise RuntimeError(f'Training physical-FOV evidence changed: {role}')
    for record in contract['golden_shards']:
        if sha256(record['path']) != record['sha256']:
            raise RuntimeError(f'Golden prepared shard changed: {record["path"]}')
    record = contract['h5ad_identity']
    stat = Path(record['path']).stat()
    if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
        raise RuntimeError('Immutable H5AD index identity changed')
    training = pd.read_csv(contract['supervised_manifest']['path'])
    for branch in ('local', 'context'):
        if set(training[f'{branch}_native_crop_fullres_px']) != {contract['native_crop_sizes_px'][branch]}:
            raise RuntimeError(f'Training {branch} native crop differs from inference contract')
    if set(training.pixel_size_um) != {contract['source_pixel_size_um']}:
        raise RuntimeError('Training source pixel size differs from inference contract')

    region_lock = json.loads(region_lock_path.read_text())
    section = region_lock["source_section"]
    if section['vit_source_id'] != contract['source_id'] or section['pixel_size_um'] != contract['source_pixel_size_um']:
        raise RuntimeError('Locked section and physical-FOV contract disagree')
    if Path(section['dapi_path']).resolve() != Path(contract['raw_dapi']['path']).resolve():
        raise RuntimeError('Locked microscopy image differs from hashed inference source')
    x0, x1, y0, y1 = map(float, region_lock["regions"]["unseen_image_example"]["pixel_window_xyxy"])
    rows = pd.read_csv(cells_path)
    if rows.empty or set(rows["source_id"].astype(str)) != {str(section["vit_source_id"])}:
        raise RuntimeError("Figure 1d inference rows do not resolve uniquely to the locked source section.")
    inside = (
        rows["centroid_x_fullres_px"].between(x0, x1)
        & rows["centroid_y_fullres_px"].between(y0, y1)
    )
    if not bool(inside.all()):
        raise RuntimeError("Figure 1d inference rows extend beyond the locked pixel window.")

    package_src = package_root / "src"
    if not package_src.is_dir():
        raise FileNotFoundError(package_src)
    sys.path.insert(0, str(package_src))
    from training_crop_backend import predict_training_protocol  # noqa: E402
    from crypt_villus_vit.sources import load_source_manifest  # noqa: E402
    from crypt_villus_vit.crops import native_crop_size_px  # noqa: E402
    sources = load_source_manifest(source_manifest)
    source = sources[contract['source_id']]
    if source.image_path.resolve() != Path(section['dapi_path']).resolve() or source.pixel_size_um != contract['source_pixel_size_um']:
        raise RuntimeError('Inference image or calibration differs from the locked section')
    resolved_sizes = {branch: native_crop_size_px(size,
                       reference_pixel_size_um=contract['reference_pixel_size_um'],
                       source_pixel_size_um=source.pixel_size_um)
                      for branch, size in contract['nominal_crop_sizes_px'].items()}
    if resolved_sizes != contract['native_crop_sizes_px']:
        raise RuntimeError('Physical crop scaling does not reproduce the training field sizes')

    out = args.output_dir.expanduser().resolve()
    predictions = out / "predictions.csv"
    summary = out / "summary.json"
    gates = out / "gate_percentages.csv"
    provenance = out / "provenance.json"
    parity = out / 'training_parity.json'
    overwritten = protect_outputs([predictions, summary, gates, provenance, parity], args.overwrite)

    result = predict_training_protocol(rows, section=section, contract=contract,
        checkpoint=checkpoint, pipeline_root=args.pipeline_root.resolve(), output_dir=out,
        batch_size=args.batch_size, device=args.device)
    model_config = result["model_config"]
    if result['resolved_crop_sizes_by_source'] != {contract['source_id']: contract['native_crop_sizes_px']}:
        raise RuntimeError('Actual inference used different native fields than the training contract')
    expected = {
        "local_readout": "center_4x",
        "context_readout": "center_4x",
        "local_crop_px": 512,
        "context_crop_px": 2048,
        "fine_crop_px": 128,
        "use_local_branch": True,
        "use_context_branch": True,
        "use_fine_branch": True,
    }
    mismatches = {key: (model_config.get(key), value) for key, value in expected.items() if model_config.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Released model configuration mismatch: {mismatches}")
    exported = pd.read_csv(predictions)
    if len(exported) != len(rows) or not exported["row_id"].astype(str).equals(rows["row_id"].astype(str)):
        raise RuntimeError("Released-model predictions do not preserve the locked input rows exactly.")

    model_sources = [
        package_src / "crypt_villus_vit/model.py",
        package_src / "crypt_villus_vit/predict.py",
        package_src / "crypt_villus_vit/crops.py",
        package_src / "crypt_villus_vit/sources.py",
    ]
    inputs = [
        file_record(source_manifest, "xenium_source_manifest"),
        file_record(cells_path, "locked_region_cells"),
        file_record(region_lock_path, "figure1d_region_lock"),
        file_record(contract_path, "physical_field_of_view_contract"),
        file_record(contract['supervised_manifest']['path'], "production_SFT_crop_evidence"),
        file_record(contract['prepared_metadata']['path'], "production_SFT_prepared_metadata"),
        *[file_record(contract[role]['path'], role) for role in
          ('development_predictions', 'prepared_pipeline_source', 'prepared_generator_source', 'prepared_manifest', 'raw_dapi',
           'anonymous_manifest_generator', 'anonymous_manifest_summary')],
        {**contract['h5ad_identity'], 'role': 'immutable_h5ad_identity'},
        *[file_record(record['path'], 'golden_prepared_shard') for record in contract['golden_shards']],
        file_record(HERE.parent / 'training_crop_backend.py', 'training_protocol_inference_adapter'),
        file_record(HERE.parent / 'cell_identity.py', 'verified_cell_identity_crosswalk'),
        file_record(checkpoint, "released_xenium_checkpoint"),
        *[file_record(path, "released_model_package_source") for path in model_sources],
    ]
    write_provenance(
        provenance,
        script=HERE,
        inputs=inputs,
        outputs=[predictions, summary, gates, parity],
        extra={
            "panel": "Figure 1d",
            "operation": "bounded inference on the prespecified artwork region; no training or selection",
            "row_count": len(exported),
            "pixel_window_xyxy": [x0, x1, y0, y1],
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "model_config": model_config,
            "reference_pixel_size_um": contract['reference_pixel_size_um'],
            "native_crop_sizes_px": resolved_sizes,
            "preprocessing_correction": contract['pixel_protocol'],
            "golden_tensors_all_exact": result['golden_tensors_all_exact'],
            "golden_prepared_cell_count": result['golden_prepared_cell_count'],
            "frozen_prediction_max_abs_difference": result['frozen_prediction_max_abs_difference'],
            "package_git_state": git_state(package_root),
        },
        overwritten=overwritten,
    )
    print(provenance)


if __name__ == "__main__":
    main()
