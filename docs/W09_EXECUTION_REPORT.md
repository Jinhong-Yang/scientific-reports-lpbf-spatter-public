# W09 generator factorial execution report

Status: PASS  
Completed: 2026-09-14T21:42:44+09:00  
Protocol SHA-256: `acb2303267117bb374c4c10fe59aea5701a52cb56d98ca2bd5b73f9e5296e587`

## Execution completeness

- Frozen cells: F00, F01, F10, F11.
- Paired seeds: 61001–61010.
- Completed fits: 40/40; failed or interrupted fits: 0; missing fits: 0.
- Budget per fit: 10 epochs and 1,120 optimizer updates.
- Cumulative measured generator time: 1.065657 GPU-hours on the approved local RTX 5080.
- Data roles: training and validation only. Held-out test rows, images, labels, and outcomes accessed: 0.
- Forty validation reconstruction arrays were regenerated from the immutable best checkpoints. Their measured MSE values reproduce the saved checkpoint-selection values within the fixed numerical tolerance.

## Validation-only factorial result

Mean best validation MSE was 0.000569 for F00, 0.000490 for F01,
0.000564 for F10, and 0.000501 for F11. Across paired seeds, the optical-proxy
main effect on validation MSE was 0.00000249 (95% Student-t interval
−0.0000275 to 0.0000325; exact two-sided sign-flip p=0.8438). The
boundary-and-moment main effect was −0.0000709 (95% interval −0.0001237 to
−0.0000180; unadjusted sign-flip p=0.0215, Holm-adjusted p=0.0645 across the
three factorial effects). The interaction estimate was 0.0000152 (95%
interval −0.0000330 to 0.0000633; sign-flip p=0.4883).

These values are development-stage reconstruction diagnostics. They do not
establish held-out image quality, detector utility, physical validity,
equivalence, or superiority. In particular, the optical proxy did not produce
a detectable mean reconstruction-MSE shift in this validation factorial, while
the boundary-and-moment estimate did not cross zero in the ordinary interval
but did not pass the three-effect Holm threshold.

## Historical bridge

Recovered R1 summaries and N2 summaries are compared descriptively by arm and
seed ordinal in `evidence/generator_factorial/reproduction_comparison.csv`.
Those rows use different seed identities and evidence layers and are not
treated as paired population effects.

## Primary evidence

- `evidence/generator_factorial/W09_COMPLETENESS.json`
- `evidence/generator_factorial/W09_ANALYSIS.json`
- `evidence/generator_factorial/factorial_seed_metrics.parquet`
- `evidence/generator_factorial/factorial_effects_validation.csv`
- `evidence/generator_factorial/factorial_effects_summary.csv`
- `evidence/generator_factorial/validation_arrays/MANIFEST.json`
