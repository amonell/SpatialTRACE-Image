"""Verified, atomic release downloads independent of a source checkout."""
from __future__ import annotations
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.parse
import urllib.request

def registry():
    data = importlib.resources.files(__package__).joinpath("artifacts.json").read_text()
    return json.loads(data)["artifacts"]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download(name, output_dir, *, source_dir=None, base_url=None):
    specs = registry()
    if name not in specs:
        raise ValueError(f"Unknown artifact {name!r}; available: {', '.join(specs)}")
    spec = specs[name]
    filename, expected = spec["filename"], spec["sha256"]
    if Path(filename).name != filename or len(expected) != 64:
        raise ValueError("Invalid packaged artifact identity")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError(f"Checksum mismatch: {destination}; existing file preserved")
        return destination
    if source_dir is not None and base_url is not None:
        raise ValueError("Specify only one of --from-dir or --base-url")
    url = (base_url.rstrip("/") + "/" + urllib.parse.quote(filename)) if base_url else spec.get("url")
    if source_dir is None and not url:
        raise RuntimeError("Public hosting has not been configured for this release. "
                           "Use --from-dir with the verified release bundle, or --base-url "
                           "after its upload. No placeholder URL is treated as a download.")
    if url:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
            raise ValueError("Use HTTPS (HTTP is permitted only for isolated localhost tests)")
    fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", suffix=".partial", dir=root)
    os.close(fd)
    try:
        with open(temporary, "wb") as target:
            if source_dir is not None:
                with (Path(source_dir) / filename).open("rb") as source:
                    shutil.copyfileobj(source, target)
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    shutil.copyfileobj(response, target)
        if sha256(temporary) != expected:
            raise ValueError(f"Checksum mismatch for {name}; no model installed")
        # A concurrent downloader may have completed meanwhile.
        if destination.exists():
            if sha256(destination) != expected:
                raise FileExistsError(destination)
        else:
            os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this function's private partial download.
    return destination
