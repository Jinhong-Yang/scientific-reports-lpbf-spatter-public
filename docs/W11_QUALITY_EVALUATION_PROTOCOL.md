# W11 common-roster quality evaluation protocol

Status: `FROZEN BEFORE W11 POOL OUTPUT`

The machine-readable specification is `configs/w11_quality_metrics.json`.
The frozen population is the ordered set of 1,024 validation targets.  Every
NB, NF, NP, NS, and ND seed is evaluated against that same real-image roster;
the held-out test partition is prohibited at this stage.

## Endpoints

- PFFD10 uses the recovered morphology definition after removing only
  `radiance_mass`, which is algebraically identical to `mean_intensity` for
  fixed-size images.  Coordinate standardization is fit once on the 3,584
  training-real images.
- The reported FID and KID sensitivity uses 768-dimensional spatially averaged
  `Mixed_6e` activations from torchvision Inception-v3 with the checksum-bound
  ImageNet weights.  This exact feature layer is included in every column name.
  It is descriptive because natural-image features are not a validated LPBF
  perceptual scale.
- Texture compares radial log-power profiles and Laplacian variance on paired
  generated/real target rows.
- Operational geometry compares intensity morphology to the inherited target
  box.  The result is not a physical particle, streak, or plume validation.
- Diversity is the mean of all 45 pairwise `1-SSIM` values across the ten
  generator seeds at each identical validation target, then averaged over all
  1,024 targets.
- The automated memorization screen is flip invariant and calibrated from
  validation-real-to-training-real distances.  Its union rule flags exact
  equality, the lowest 1% of ResNet-18 cosine distances, or the highest 1% of
  nearest-reference SSIM values.  pHash is diagnostic only.  A flag is a
  similarity screen and neither proves memorization nor establishes its
  absence.

## Scope limitation

The user excluded the independent-rater process.  W11 therefore verifies array
hashes, counts, common-roster alignment, source restrictions, and operational
inherited-box mapping only.  `NEW_POOL_LABEL_VALIDITY.json` must retain
`SCOPE_LIMITED_INHERITED_OPERATIONAL_TARGET_ONLY`; human agreement, corrected
label effects, and physical object validity are prohibited claims.

The analysis must produce exactly 50 arm-by-seed aggregate rows, preserve the
per-image measurements, and report all outcomes irrespective of direction.
