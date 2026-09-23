# W10 prior-path and operator results

Status: `PASS`

All 18 frozen lambda-by-seed cells completed without replacement, failure, or
held-out test access.  The results below are validation diagnostics and do not
establish a calibrated physical field or detector benefit.

## Regularization path

| Proxy lambda | Mean validation MSE | SD |
|---:|---:|---:|
| 0 | 0.0006011 | 0.0000725 |
| 0.00035 | 0.0005329 | 0.0000507 |
| 0.0035 | 0.0005256 | 0.0000358 |
| 0.01 | 0.0005828 | 0.0000392 |
| 0.035 | 0.0007403 | 0.0000824 |
| 0.1 | 0.0009067 | 0.0001015 |

The primary value 0.0035 was fixed before these outcomes.  Relative to zero,
it reduced mean validation MSE by 12.5%, the common finite-difference residual
by 15.9%, and the high-frequency PSD fraction by 65.7%.  At lambda 0.1, the
common residual was 71.1% lower than at zero while validation MSE was 50.9%
higher.  The path therefore shows a strength-dependent smoothness/fidelity
trade-off; low residual alone is not a fidelity endpoint.

## Smoothing control

Gaussian sigma 0.5 pixels and the paired heat value tau 0.125 pixel-squared are
numerically the same implemented smoothing operation.  Their mean validation
MSE was 0.0005998, 14.1% above the lambda-0.0035 target of 0.0005256.  Both fail
the frozen plus-or-minus 5% reconstruction-match criterion and remain
`UNMATCHED_CLOSEST_ONLY`.  No matched-control claim is permitted.

## Operator and condition specificity

Across lambda 0 to 0.1, the common residual declined from 0.3478 to 0.1005,
Laplacian RMS from 12.42 to 3.27, and high-frequency PSD fraction from 0.00569
to 0.000122.  These jointly support a smoothing interpretation of the tested
regularizer.

Rotating the power, speed, and line-energy triplet across validation rows
changed mean scored residual by only 0.0000007 to 0.0000244 across the six
lambda values, although individual differences ranged roughly from -0.018 to
+0.029.  This diagnostic does not show a meaningful mean specificity to that
condition triplet.  It narrows the mechanism claim to the behavior of the
implemented image-space operator and forbids interpreting the score as
validated LPBF physics.

Source artifacts are `evidence/prior_sweep/regularization_path_aggregate.csv`,
`evidence/prior_sweep/smoothing_match.csv`, and
`evidence/prior_sweep/mechanism_diagnostics.parquet`; their checksums are bound
in `evidence/prior_sweep/W10_ANALYSIS.json`.
