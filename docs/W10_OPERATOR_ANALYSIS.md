# W10 optical-prior and smoothing analysis

Protocol SHA-256: `acb2303267117bb374c4c10fe59aea5701a52cb56d98ca2bd5b73f9e5296e587`

The analysis uses validation reconstructions only; no held-out test payload was accessed.

## Matched-control decision

- gaussian_sigma_px: strength 0.5, relative MSE gap 0.1410, status `UNMATCHED_CLOSEST_ONLY`.
- heat_tau_px2: strength 0.125, relative MSE gap 0.1410, status `UNMATCHED_CLOSEST_ONLY`.

A control marked `UNMATCHED_CLOSEST_ONLY` is retained diagnostically and is not described as a matched comparator.

Common optical residuals use one frozen finite-difference scoring operator for every family. Own-operator roughness is the Laplacian RMS. Condition-triplet diagnostics rotate the power, speed, and line-energy coefficients across the frozen validation roster; this is a specificity diagnostic, not an intervention on the physical process.
