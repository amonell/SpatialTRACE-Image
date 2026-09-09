from crypt_villus_vit.cli import build_parser
from crypt_villus_vit.artifacts import registry


def test_production_v2_is_the_cli_default():
    args = build_parser().parse_args(["download-weights"])
    assert args.model == 'xenium'
    assert registry()['xenium']['source_checkpoint_sha256'] == 'd0e02005b18a670c77c7956d5500d8c599879e183def79badae7fbb526aaf137'


def test_all_registered_weights_have_fixed_checksums():
    assert set(registry()) == {'xenium', 'if', 'peyer', 'representation'}
    assert all(len(item['sha256']) == 64 for item in registry().values())
