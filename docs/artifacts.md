# Verified model artifacts

The packaged src/crypt_villus_vit/artifacts.json is the authority for filenames, SHA256 hashes and original checkpoint identities. It is included in built wheels.

The four checkpoints are available on the private GitHub prerelease `v1.0.0rc1`. Repository access and an authenticated GitHub CLI are required:

```bash
gh release download v1.0.0rc1 --repo amonell/TissueMapper-Image --pattern '*.pt' --pattern checksums.json --pattern LICENSE --dir release-weights
uv run --locked --extra cpu tissuemapper-image download --model xenium --from-dir release-weights --output-dir weights
uv run --locked --extra cpu tissuemapper-image download --model if --from-dir release-weights --output-dir weights
uv run --locked --extra cpu tissuemapper-image download --model peyer --from-dir release-weights --output-dir weights
uv run --locked --extra cpu tissuemapper-image download --model representation --from-dir release-weights --output-dir weights
```

The second through fifth commands verify the packaged SHA256 identities before installing the downloaded files. Use a new download directory; neither workflow silently replaces existing files.

Anonymous public URLs remain unset while the repository is private. The Python downloader does not read GitHub credentials; use `gh release download` for authentication and then `--from-dir` for verified installation. An eventual public HTTPS release can use `--base-url`. Existing corrupt files are rejected, not overwritten. Downloads use private temporary files and validate hashes before installation.

Only tensor weights and basic model metadata are exported. All parameter tensors match the frozen models. Code and the four designated Image checkpoints are licensed under GPL-3.0-only. Microscopy and figure data have separate terms. Never download checkpoints from an untrusted source.
