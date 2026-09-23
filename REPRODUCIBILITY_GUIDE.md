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

Obtain AI-Hub dataset 71476 separately under its provider terms. The public
package is an aggregate manuscript reconstruction package, not a turnkey
full-training environment: local source mounts, the recovered historical
`metal_spatter_pinn` package, checkpoints and prediction payloads are excluded.
The archived runners document the original execution but require those
additional authorized dependencies and reconstruction of the frozen manifests.
Do not infer that a public verification PASS reruns training, inference, or
the specimen bootstrap. Do not open a new held-out payload before an applicable
pretest freeze and unlock. Private operational files are not public instructions.

The full study is compute-intensive and was executed on a local NVIDIA GeForce
RTX 5080. Small mechanics fixtures are supplied for end-to-end code validation;
they are not scientific result substitutes.

## Interpretation limits

The inherited bounding boxes are an operational target with unverified physical
semantics because independent human review was excluded. W16 is internal OOF
robustness, not external validation. Equal detector updates do not imply equal
real-image exposure, and 64×64 synthetic images upsampled to 300×300 do not have
the native information resolution of 300×300 real images.
