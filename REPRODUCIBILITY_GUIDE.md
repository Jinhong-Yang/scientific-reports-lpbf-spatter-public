# Reproducibility guide

## Evidence layers

- `H0_HISTORICAL` is the copied, read-only prior project and is excluded from
  the public archive.
- `R1_REPRODUCTION` records bounded recovery and implementation checks.
- `N2_NEW_STUDY` contains the frozen experiment, validation, locked test, and
  final statistical evidence used in the manuscript.

## Public no-source-data verification

After extracting the release archive, create a Python 3.11 environment, install
`requirements-reporting.txt`, and run `reproduce.ps1` on Windows or
`reproduce.sh` on POSIX. This route runs the public-package tests and verifies
the published SHA-256 ledger. Pass `-RebuildManuscript` on Windows or set
`REBUILD_MANUSCRIPT=1` on POSIX to regenerate manuscript text, tables, and
figures from the released aggregate matrices. It does not recompute the paired
bootstrap, because the required prediction payloads are intentionally excluded;
it does not require or reconstruct licensed pixels.

## Authorized full-data route

Obtain AI-Hub dataset 71476 separately and preserve its provider license. Mount
or link the expected archive tree outside Git, then follow the paths described
in `PROJECT_IDENTITY.example.json` in the release archive. Run the staged
materialization, preflight, generator, detector, and evaluation commands only
in the order documented by `NEXT_ACTION.md` and the immutable receipts. Do not
open the held-out payload before the pretest freeze and unlock steps.

The full study is compute-intensive and was executed on a local NVIDIA GeForce
RTX 5080. Small mechanics fixtures are supplied for end-to-end code validation;
they are not scientific result substitutes.

## Interpretation limits

The inherited bounding boxes are an operational target with unverified physical
semantics because independent human review was excluded. W16 is internal OOF
robustness, not external validation. Equal detector updates do not imply equal
real-image exposure, and 64×64 synthetic images upsampled to 300×300 do not have
the native information resolution of 300×300 real images.
