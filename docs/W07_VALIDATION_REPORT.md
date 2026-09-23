# W07 statistical-plan prevalidation report

Status: `PASS_WITH_LIMITATION`  
Evidence layer: `SYNTHETIC_FIXTURE_ONLY`  
Validated: 2026-09-14 (KST)

## Outcome

The proposed paired cluster-and-pipeline-replicate inference mechanics pass all
eight predeclared fixture checks. No observed N2 model outcome and no held-out
test record was used. The result validates implementation behavior; it does not
freeze the W08 analysis plan or support a scientific comparison.

## Checks completed

- A 300-simulation null fixture produced 95% interval coverage of 0.9533
  (Monte Carlo standard error 0.0122) and a zero-rejection rate of 0.0467.
- A known 0.02 AP effect fixture produced mean estimate 0.01997, coverage 0.94,
  and zero rejection in 0.9967 of simulations.
- The five-paired-seed exhaustive two-sided sign-flip floor is 0.0625; the
  all-zero vector returns p=1.
- The 2×2 factorial main-effect and interaction formulas and Holm adjustment
  match their reference fixtures while undefined comparisons remain undefined.
- The raw-record pooled-COCO fixture resamples the same specimen-cluster and
  pipeline-replicate draws in both arms, duplicates sampled records, assigns
  fresh evaluation image IDs in the COCO wrapper, and recomputes AP rather than
  averaging cached image-level metrics.
- Its A/A path returned exactly zero difference in all 20 draws. The
  known-difference path exercised duplicate clusters in 35 of 40 draws.
- The 36-row precision/power grid is explicitly marked synthetic and not
  pilot-calibrated.

## Reproducibility and validation

- `src/analysis/prevalidate_statistics.py` generated the JSON and CSV evidence.
- `tests/test_prevalidate_statistics.py` contributes six focused tests.
- The fresh W05 environment ran the full suite: 135 tests passed, one benign
  torchvision configuration warning, and 37 subtests passed.
- `notebooks/W07_statistical_prevalidation.ipynb` executed top to bottom,
  rendered to HTML, and its 1440×5000 preview was visually inspected. No cell
  error, truncation affecting interpretation, or layout overlap was observed.

## Limitations and next gate

The variance components are declared synthetic fixtures. W06 must measure
dev-pilot variance, run-time/memory, and bootstrap-tail Monte Carlo error. W04
must resolve inherited label semantics. W08 must then approve and freeze the
primary endpoint and comparison family, resampling unit, practical margin,
production draw count, experiment matrix, and compute budget. Manufacturing
build uncertainty and external validity remain unidentifiable without new
provenance or a genuinely independent cohort.
