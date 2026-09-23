# W16 grouped-internal out-of-fold protocol

Status: `FROZEN BEFORE W16 OUTCOMES`

W16 is a secondary internal robustness analysis, not external validation. It
uses only the 144 specimens already assigned to W09 training or validation and
does not read the 32-specimen N2 held-out payload. For each of seeds
64001--64003, specimens are ordered within their acquisition stratum by a
seeded SHA-256 key and distributed round-robin across three outer folds. Both
views and all 32 images from a specimen remain together.

Each partition crosses three outer folds with N1, NR, NB, and NP, yielding the
approved 36 detector trajectories. Every trajectory trains on two folds and
predicts only its untouched outer fold. OOF predictions from the three folds
are concatenated only after all fold-level runs for that partition are complete.

NP is refit independently inside every outer training fold (nine generator
fits). It may not reuse the full-development or W14 generator, latent bank,
scaler, or donor plan. Its checkpoint is the fixed final tenth epoch; the OOF
fold is never used for generator selection. NB donors and NR repeats are also
restricted to the outer training fold. Detector architecture, optimizer,
8,960-update budget, and 0.25 mixed ratio inherit the frozen W12 contract.

This design estimates robustness to specimen grouping inside the available
development cohort. It does not isolate manufacturing build, acquisition site,
or a genuinely external cohort, and it cannot support transfer or external-
validity claims. The inherited operational box target remains physically
unverified because independent human review was excluded.
