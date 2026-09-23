# W05 environment and evaluator validation report

Status: `PASS_WITH_LIMITATION`. No study training trajectory was started.

## Fresh environment

A new `.venv-w05` environment was created independently of the historical
environment. All ten pinned core packages matched the requested versions,
including torch 2.11.0+cu128, torchvision 0.26.0+cu128, and pycocotools 2.0.11.
CUDA 12.8 and the NVIDIA GeForce RTX 5080 were available. The environment and
freeze receipts are under `evidence/environment/`.

## Validation results

- Fresh-environment regression suite: 129 tests passed, plus 37 parametrized
  subtests; one expected torchvision configuration warning.
- Data interface: one hash-verified validation image decoded to a 3×300×300
  float32 tensor in [0,1]; `xywh` converted to the expected `xyxy`; the
  13-element condition vector was finite.
- Dataset compatibility finding: the recovered `RowsDataset` metadata omits
  `split`. W05 therefore verifies the split from the manifest row before
  constructing the dataset and records the omission; it does not infer split
  from paths.
- COCO oracle: perfect prediction produced AP≈1, empty prediction AP=0, empty
  ground truth remained undefined, and foreground label 0 was rejected at the
  canonical COCO interface.
- Manufactured solution: for `u(x,y)=x²+y²`, finite-difference and autograd
  Laplacians matched 4 with maximum errors about 1.32×10⁻⁸ and 0.
- Deterministic order: a 1,024-item seeded permutation was reproduced and hash
  locked.
- Generator: second-derivative regularizer forward/backward passed on toy CUDA
  tensors without study images.
- Detectors: random-initialized Faster R-CNN and RetinaNet completed one toy
  forward/backward/optimizer step, produced finite losses and gradients, and
  ran prediction outputs through canonical COCO conversion/evaluation. No
  pretrained weights were downloaded.

## Limitations and gate impact

PyTorch warned that ROI Align backward has no deterministic implementation on
this runtime. Data order is deterministic, but detector backward is not claimed
bitwise deterministic. The AP values from random toy weights are pipeline
fixtures, not scientific outcomes.

W05 validates mechanics only. It does not resolve W04 box semantics, authorize
full training, make the optical residual a physical field, or use the test split
for calibration or selection.
