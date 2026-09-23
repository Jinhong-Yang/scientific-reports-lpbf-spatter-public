# W12 core-detector execution protocol

Status: `FROZEN BEFORE W12 DETECTOR OUTCOMES`

The machine-readable specification is `configs/w12_detector_core.json`.  It
defines the 80 Faster R-CNN and 20 RetinaNet core trajectories already counted
in the approved 244-trajectory matrix.  No held-out test image, label, or model
outcome may be opened during W12.

## Ratio interpretation

The protocol approved a four-value validation grid but allocated no extra
detector trajectories for that grid.  Executing it would add at least 20
unapproved trajectories before replication.  The preapproved incomplete-grid
fallback of 0.25 is therefore frozen before detector outcomes.  From each
3,584-row pool, every fourth row beginning at index zero forms the common
896-row additional bank.  A seeded without-replacement stream samples the
3,584 base-real and 896 additional rows, giving an exact 20% additional-row
fraction per complete roster epoch.

This is an equal-update and equal-total-image-exposure comparison.  N0 and N1
receive 35,840 real presentations over 8,960 updates.  Mixed synthetic arms
replace one fifth of those presentations with synthetic samples in expectation
over partial epochs; NR uses additional augmented-real rows in the same slots.
Accordingly, NP versus NR is the control for replacing additional real exposure
with synthetic exposure under the fixed batch.  The design does not establish
equal real exposure and that limitation must remain explicit.

## Frozen implementation

- Native detector tensors are 300×300.  Synthetic 64×64 arrays are bilinearly
  resized to 300×300; they are never described as native-resolution images.
- Real presentations in N1 and every mixed arm receive the same horizontal
  flip, brightness, and contrast pipeline.  Synthetic presentations receive no
  conventional transform.
- Training uses SGD, bfloat16 autocast, batch four, and exactly 8,960 optimizer
  updates.  Validation at updates 896, 1,792, 4,480, and 8,960 is diagnostic;
  only update 8,960 is the primary checkpoint.
- Every milestone retains raw validation predictions and a complete checkpoint.
  Failure, interruption, memory exhaustion, or nonfinite values are recorded
  without changing the seed, precision, batch, arm, or endpoint.
- RetinaNet transfers the common pool ratio and uses the first five paired pool
  replicates.  It is a robustness family, not an independently optimized model.

Labels remain the operational `inherited_bbox_region`; their physical object
semantics were not independently reviewed at the user's direction.
