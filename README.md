# Optical residual regularization for LPBF spatter synthesis and detection

This repository contains the frozen, source-data-free reproducibility materials and Scientific Reports submission draft for the LPBF synthetic-image control study.

## Study boundary

The N2 study uses 5,632 images from 176 specimens with specimen-grouped training, validation, and one locked internal held-out campaign. Results concern the inherited operational `inherited_bbox_region` target. Independent human label review was excluded by the project decision owner, and no external cohort was available; physical particle semantics and external validity are therefore not claimed.

## Reproduce the public package

Create a Python 3.11 environment, install `requirements-reporting.txt`, and run `reproduce.ps1` on Windows or `reproduce.sh` on POSIX. The default route verifies the archive and public-package tests. The optional manuscript rebuild consumes only released aggregate matrices and does not recompute the bootstrap from excluded prediction payloads.

## Availability

- Repository: https://github.com/Jinhong-Yang/scientific-reports-lpbf-spatter-public
- Immutable release: `v1.0.1`
- Source dataset: AI-Hub Metal 3D-Printing Spark Image Data, dataset 71476; obtain separately under the provider's terms.
- Raw source pixels, checkpoints, and predictions are not redistributed.
- No project-wide open-source license is asserted in this release; third-party components retain their own terms.

See `REPRODUCIBILITY_GUIDE.md`, `DATA_AVAILABILITY.md`, `CODE_AVAILABILITY.md`, and `docs/SUBMISSION_READINESS.md` for the audit boundary and submission status.
