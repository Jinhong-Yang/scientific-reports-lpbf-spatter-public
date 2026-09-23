# Protocol amendment 003 — W12 RetinaNet invalid-prediction handling

Date: 2026-09-16  
Classification: pre-test technical recovery

## Trigger

The first secondary RetinaNet trajectory (`secondary_retinanet_N1_s62001`)
stopped during validation at the 4,480-update milestone.  The frozen COCO
adapter correctly rejected a detector output with non-finite or non-positive
geometry as `Invalid COCO detection`.  The training checkpoint, sampler state,
and prior 896- and 1,792-update validation receipts were retained.  No held-out
test payload was accessed.

## Recovery

The original W12 runner and `detector_common.py` remain byte-for-byte unchanged
so that all completed W12 receipts and resumable states retain their original
identity binding.  A separate, W12-secondary-RetinaNet-only recovery wrapper
uses the original runner with a prediction boundary adapter.  The adapter:

1. retains canonical foreground detections whose coordinates and scores are
   finite and whose `x2 > x1` and `y2 > y1`;
2. discards only model outputs that cannot be represented as a valid COCO
   detection; and
3. writes the per-image candidate and discarded counts into the normal
   prediction artifact for audit.

It neither clips, relabels, reorders, scores, augments, nor selects valid
detections.  Valid predictions are passed to the existing official COCO
adapter unchanged.  The wrapper verifies that the underlying W12 and common
source hashes match the run identity before resuming a trajectory.

## Scientific invariants

- No source image, split, inherited label, arm, seed, model weight, optimizer,
  update count, checkpoint rule, endpoint, or statistical comparison changes.
- The frozen 100-trajectory roster is unchanged.
- Existing completed W12 artifacts are not overwritten or regenerated.
- Failed attempts and their failure records remain retained in-place.
- Held-out test access remains prohibited until the separately frozen W17
  unlock.

This amendment repairs an output-serialization boundary for invalid detector
outputs; it does not authorize a scientific protocol change or outcome-based
model selection.
