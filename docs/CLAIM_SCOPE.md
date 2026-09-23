# Claim scope for the new Scientific Reports study

Status: W01 outcome-neutral scope lock. This document separates recovered facts
from claims that require new evidence.

## Recovered facts that may be stated with H0 qualification

- The accessible local corpus contains 5,632 selected images from 176
  specimens, two views, and 32 process-condition strata.
- The corpus was selected from complete validation archives; the recovered
  study record says official training volumes `TS.z05`, `TS.z06`, and `TS.z07`
  were unavailable.
- Historical experiments include optical-residual neural fields, donor-matched
  controls, generator and detector checkpoints, per-image metrics, and raw
  detector predictions.
- The old 32-specimen test partition has already been accessed. It is not a
  pristine external cohort.
- The historical model's residual is an optical-radiance proxy. It is not a
  calibrated temperature or multiphysics field.

## Outcome-neutral contribution statements allowed before N2 results

- “We provide a label- and exposure-controlled evaluation of optical-residual
  regularization for LPBF spatter image synthesis.”
- “We separate image-generation effects, inherited-label policy, donor
  support, and additional training exposure.”
- “We compare neural-field, non-neural, smoothing, real-exposure, and compact
  diffusion controls under a frozen evaluation protocol.”
- “We report positive, small, unresolved, or adverse effects according to the
  same predeclared rules.”

These statements describe the planned design. They do not assert that the
planned cells have run or that any method improves detection.

## Claims prohibited by evidence status or approved scope

- No “physically validated generator”, “thermal PINN”, or “physics-correct
  diffusion” claim without a calibrated observation model and physical-field
  validation.
- No “external validation” claim for a resplit or reprocessed version of the
  already accessed 176-specimen corpus.
- No “equivalent” or “no effect” claim from a confidence interval that merely
  crosses zero; equivalence/non-inferiority requires a frozen margin and the
  corresponding analysis.
- No statement that inherited boxes have human-validated or physically verified
  semantics. The user excluded W04 independent-rater review and W13
  corrected-label interaction before N2 held-out outcomes. Detector conclusions
  must therefore remain conditional on the operational
  `inherited_bbox_region` target even after W11 quality analysis is complete.
- No first-ever or broad superiority claim based only on abstract-level
  literature review.
- No new headline number until it can be recomputed from preserved predictions
  and ground truth.

## RQ evidence mapping

| Research question | Evidence required before a result claim |
|---|---|
| RQ1 optical-prior trade-off | Matched architecture factorial, weight path, non-PDE smoothing control, and operator analysis |
| RQ2 supervision validity | Scope-limited control of a fixed inherited-target policy; no human-validity or corrected-label claim because W04/W13 are excluded |
| RQ3 robustness | Compact diffusion baseline, resolution and exposure controls, repeated low-data realizations |
| RQ4 new evaluation unit | Truly independent cohort or explicitly limited protocol-frozen grouped internal evaluation |
