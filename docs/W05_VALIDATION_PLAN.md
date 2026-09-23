# W05 execution environment and evaluator validation plan

Status: implementation in progress. This phase validates interfaces on fixtures
and toy tensors; it does not start a study training trajectory.

## Environment boundary

- Create `.venv-w05` as a fresh virtual environment.
- Install the exact core versions in `configs/w05_core_requirements.txt` from
  public package indexes.
- Record installed versions, executable, platform, CUDA device, dependency
  lock hash, and a `pip freeze` receipt.
- Keep the historical `.venv` available only as R1 evidence and fallback for
  comparing behavior; do not call it the fresh N2 environment.

## Required validations

1. Raw 300×300 grayscale pixel to tensor conversion.
2. Canonical `xywh` to `xyxy` box conversion and horizontal reflection.
3. Historical 13-element condition vector on a known manifest row.
4. COCO oracle fixtures: perfect prediction, empty prediction, empty ground
   truth, invalid foreground mapping, and mixed true/false detection.
5. Specimen-grouped deterministic data order and resume identity.
6. Finite-difference/manufactured-solution checks for the coordinate/PDE
   interface without claiming a calibrated physical field.
7. Generator second-derivative loss forward/backward on toy tensors.
8. Faster R-CNN and RetinaNet single-step forward/backward and foreground-label
   semantics on toy tensors, subject to bounded GPU memory.

## Gate boundaries

Human label semantics remain a W04 gate. W05 can validate coordinate and metric
mechanics with synthetic boxes, but it cannot certify that historical boxes are
scientifically valid supervision. No test split is used for calibration or
model selection, and no full training run is authorized.

W05 passes only after the fresh environment, model/transform specifications,
tests, and `evaluator_validation.json` are all present and reproducible.
