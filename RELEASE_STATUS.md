# Release candidate status

The release candidate includes uv-locked CPU/GPU installs, own-data training and inference, four verified model artifacts, synthetic examples, and complete paper rendering code. The source is hosted in the private `amonell/TissueMapper-Image` GitHub repository; its `v1.0.0rc1` prerelease carries the four checkpoints. See `VALIDATION.md` for executed local tests and GitHub Actions for hosted checks. This is not a public release.

Before public release:

- Obtain authorization to change the repository from private to public.
- Configure anonymous artifact URLs after public publication; private downloads use the authenticated GitHub CLI.
- Upload the verified paper-input bundle after confirming permission to redistribute it.

Code and the four designated Image checkpoints use GPL-3.0-only. The earlier MIT notice is retained in LICENSES/MIT-legacy.txt. Trained artifact tensors are unchanged; payloads exclude optimizer state and private workstation metadata. The original development repository is preserved separately.

The new repository starts from the reviewed, history-free source export. The older repository and its history, including superseded model binaries, are preserved separately and have not been pushed here.
