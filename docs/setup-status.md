# Setup status

Date: 16 October 2026 (Asia/Singapore).

## Desktop location

The complete project was moved to `C:\Users\Lenovo\Desktop\Leafy AI` at the user's
request. The original generated output folder no longer contains a second copy.
Git history, the Python environment and all three model checkpoints moved with it.
The editable package, activation scripts and console launchers were repaired for
the new path. Python imports now resolve to the Desktop source directory.

No Codex/ChatGPT sidebar project has been created; this is the actual local Git
project directory. The current chat remains attached to its original workspace.

## GitHub

Origin: `https://github.com/Leafy-AI-90/AI.git`.

The GitHub connector authenticated as Leafy-AI-90, but repository writes returned
HTTP 403, `Resource not accessible by integration`. The local Git credential
helper was also inaccessible in the sandbox. The signed-in Microsoft Edge
session provides the alternative publication route requested by the user.

The web upload publishes the tracked source snapshot with its directory layout.
Large dataset archives, downloaded model weights and `.venv` are excluded; the
repository contains pinned download scripts, manifests and verification records.
Original local setup history is retained in the Desktop Git repository. A web
upload creates its own GitHub commit rather than transferring those local commits.
No credentials are stored in source files.

The original local setup commits used the placeholder project author
`Leafy AI <leafy-ai@users.noreply.github.com>`. Subsequent local changes use the
verified account's standard GitHub noreply identity.

## Installed and validated

- Python 3.12 CPU environment in `.venv`; dependencies in `requirements-lock.txt`.
- Three official ImageNet checkpoints in ignored `models/weights`; exact sources,
  versions and hashes in `models/manifest.json`.
- Six original integrity tests plus three dataset tests: 9 passed.
- `python -m pip check`: no broken requirements.
- All three image backbones passed 384 x 384 forward inference, embedding and
  five-class output-head checks.
- PLSR, PLS-DA, logistic regression, random forest, SVM, MLP and XGBoost passed
  synthetic fitting checks; F1/F2 fusion fits and F3 forward/backward passed.
- Both datasets are downloaded, checksummed, fully decoded and indexed by split.
  See `data/dataset-summary.json` and `docs/datasets.md` for counts and limitations.
- Real-data batch checks are recorded in `docs/data-verification.json`.

No full dataset training or test-set evaluation has been performed. The datasets
support image pretraining/adaptation; they do not replace the paired Choy Sum
measurements needed for the proposed spectral/fusion study.

## Grad-CAM packaging

The official Grad-CAM 1.5.5 source archive was verified against PyPI SHA-256
`690c433d226d35c89c9eb170462db204909cb06b39c7381e6880a49b6fc37015`.
A small wheel was built without code changes because pip's archive permission
restoration failed in the sandbox. See `vendor/README.md` for its checksum.
