"""Checksum download tests independent of author data and network services."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
import pytest
from crypt_villus_vit import artifacts


def test_download_atomic_and_rejects_corruption(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    (source/'model.pt').write_bytes(b'verified fixture, not a model')
    spec = {'demo': {'filename': 'model.pt', 'sha256': artifacts.sha256(source/'model.pt'), 'url': None}}
    monkeypatch.setattr(artifacts, 'registry', lambda: spec)
    destination = artifacts.download('demo', tmp_path/'weights', source_dir=source)
    assert destination.read_bytes() == (source/'model.pt').read_bytes()
    destination.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Checksum'):
        artifacts.download('demo', destination.parent, source_dir=source)
    assert destination.read_bytes() == b'corrupt'


def test_localhost_download_and_bad_hash_cleanup(tmp_path, monkeypatch):
    source = tmp_path/'serve'
    source.mkdir()
    (source/'model.pt').write_bytes(b'local fixture')
    spec = {'demo': {'filename': 'model.pt', 'sha256': artifacts.sha256(source/'model.pt'), 'url': None}}
    monkeypatch.setattr(artifacts, 'registry', lambda: spec)
    server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(SimpleHTTPRequestHandler, directory=source))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}'
        path = artifacts.download('demo', tmp_path/'valid', base_url=base)
        assert path.read_bytes() == b'local fixture'
        spec['demo']['sha256'] = '0'*64
        with pytest.raises(ValueError, match='Checksum'):
            artifacts.download('demo', tmp_path/'invalid', base_url=base)
        assert list((tmp_path/'invalid').iterdir()) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unconfigured_url_is_actionable(tmp_path):
    name = next(iter(artifacts.registry()))
    if artifacts.registry()[name].get('url'):
        pytest.skip('Public hosting configured')
    with pytest.raises(RuntimeError, match='Public hosting'):
        artifacts.download(name, tmp_path)
