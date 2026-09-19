"""Lightweight run records without re-reading whole-slide microscopy files."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def record_cli(args, argv=None):
    output = getattr(args, 'output_dir', None)
    if output is None or getattr(args, 'command', None) not in {
            'train', 'predict-cells', 'predict-qupath', 'pretrain-representation', 'prepare-pretraining'}:
        return
    output = Path(output)
    destination = output/'run_provenance.json'
    if destination.exists():
        raise FileExistsError(destination)
    inputs = []
    for key, value in vars(args).items():
        if isinstance(value, Path) and key not in {'output_dir', 'output'} and value.is_file():
            inputs.append(dict(argument=key, **file_record(value)))
    images = []
    source_manifest = getattr(args, 'source_manifest', None)
    if source_manifest is not None:
        from .sources import load_source_manifest
        for source in load_source_manifest(source_manifest).values():
            path = source.image_path.resolve()
            stat = path.stat()
            images.append(dict(source_id=source.source_id, path=str(path), bytes=stat.st_size,
                               mtime_ns=stat.st_mtime_ns, pixel_size_um=source.pixel_size_um,
                               channel_index=source.channel_index))
    outputs = [file_record(p) for p in sorted(output.iterdir()) if p.is_file() and p != destination]
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  command=sys.argv if argv is None else ['spatialtrace-image', *argv],
                  configuration={k: v for k, v in vars(args).items() if k != 'func'},
                  python=platform.python_version(),
                  packages={name: importlib.metadata.version(name) for name in
                            ['spatialtrace-image', 'torch', 'numpy', 'tifffile', 'imagecodecs', 'zarr']},
                  inputs=inputs, source_images=images, outputs=outputs,
                  image_identity_scope='Image paths, size and mtime are recorded, not content hashes. '
                  'Archive image hashes separately for immutable scientific reproduction.')
    destination.write_text(json.dumps(record, default=str, indent=2))
