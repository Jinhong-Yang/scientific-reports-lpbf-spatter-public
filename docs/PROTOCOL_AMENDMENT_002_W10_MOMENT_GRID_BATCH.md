# Protocol amendment 002 — W10 moment-grid batch dimension

Date: 2026-09-14  
Classification: pre-outcome implementation correction  
Affected run: W10 `L0p00000_s66001`, attempt 1 only

The first W10 cell stopped before completing epoch 1 because the shared
moment-grid tensor was passed as `[points, 2]` rather than expanded to
`[batch, points, 2]`. The completed W09 implementation already used the
required expansion. W10 now applies the same expansion to every cell.

Attempt 1 produced no checkpoint eligible for selection and no scientific
metric. Its identity, configuration, initial resume state, and failure record
are retained under `runs/new_study/W10_prior_sweep_failed_attempts/`.

No data, roster, arm, seed, weight, objective, update count, checkpoint rule,
or analysis changes. Test access remains prohibited.
