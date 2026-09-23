# Statistical analysis plan — W08-frozen analysis policy

Status: `FROZEN_PRE_N2_OUTCOME`. No observed N2 outcome or test result informed
this plan. The approved values are bound by `configs/PROTOCOL_LOCK.yaml`.

## Primary estimand proposal

For each independently trained pipeline replicate, calculate pooled COCO
AP@[.50:.95] on held-out real images from preserved raw predictions and ground
truth. The primary estimand is the mean across pipeline replicates. Do not pool
predictions from different models into an ensemble AP. Historical mean-specimen
AP remains a required secondary bridge to H0 and cannot replace pooled AP.

Frozen primary family at the final operating point:

1. `NP − NB`
2. `NP − NF`
3. `NP − N1`
4. `NP − NR`

These four comparisons were frozen at W08. They were designed with knowledge
of prior work and must not be described as pristine confirmatory superiority
tests on the internal corpus.

## Units and pairing

- Resampling unit: the strongest evidenced independent unit. Currently this is
  specimen, not frame or manufacturing build.
- Every resample uses the same specimen draw and pipeline replicate draw for
  both arms of a contrast.
- A duplicated sampled specimen receives newly enumerated evaluation image IDs
  when raw prediction/ground-truth records are recomputed.
- Multi-view images from one specimen stay in the same sampled cluster.
- Nested-CV analyses use only out-of-fold predictions for each specimen and do
  not count overlapping training folds or seeds as new specimens.

## Intervals and families

- Report effect size and an ordinary 95% paired cluster interval.
- For the four proposed primary comparisons, use Bonferroni family-wise 0.05,
  corresponding to individual two-sided 98.75% intervals, unless W07 validates
  and W08 freezes a different simultaneous procedure.
- The proposed production bootstrap count is 20,000. W06 must measure tail
  Monte Carlo error and cost before freeze.
- A percentile bootstrap is not called exact or assumed to have guaranteed
  coverage. Conditional and population-level intervals must be named according
  to what is resampled.

## Generator factorial

For paired seed `s`:

- Optical-proxy main effect:
  `0.5 × [(F10_s − F00_s) + (F11_s − F01_s)]`
- Boundary/moment main effect:
  `0.5 × [(F01_s − F00_s) + (F11_s − F10_s)]`
- Interaction: `(F11_s − F10_s) − (F01_s − F00_s)`

Exact sign-flip inference requires sign symmetry/exchangeability. With five
paired seeds, the minimum attainable two-sided exhaustive sign-flip p-value is
0.0625. Constant nonzero difference vectors remain mathematically enumerable;
all-zero vectors yield p=1 rather than an invented significance result.

## Missing and failed cells

- Undefined AP strata and no-ground-truth groups remain null, not zero.
- Failed model runs remain failed records and are not silently replaced or
  omitted.
- A primary contrast is not reported if its predeclared cells, raw predictions,
  or pairing keys are incomplete.
- Seed, milestone, subset, detector family, validation, internal test, and
  external cohort families are not combined or split after inspecting results.

## Precision and utility margin

Values 0.005, 0.01, and 0.02 AP appear only as descriptive planning scenarios.
No confirmatory practical-equivalence or noninferiority margin is adopted. The
synthetic W07 grid is not a power claim because its variance components are
fixtures.

## Frozen limitations

- A/A, known-effect, and raw-record pooled COCO fixtures validate mechanics,
  not observed-study coverage.
- W04 independent human review is excluded; inherited boxes remain an
  operational region target with unverified physical semantics.
- The route is grouped internal evaluation, not external validation.
- Actual replicate variance and bootstrap tail stability remain execution
  diagnostics and cannot alter the frozen primary family.
