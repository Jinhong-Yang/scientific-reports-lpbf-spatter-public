# W06 bounded development-pilot report

Status: `PASS — W08 APPROVED PRE-TEST`
Technical run: `PASS_WITH_LIMITATION`  
Evidence layer: `N2_NEW_STUDY_DEV_ONLY`

W08 approved the bfloat16 detector path, fixed budgets, HPO grids,
compact-diffusion specification, local compute ceiling, and selection rules.
The canonical decisions are in `configs/PROTOCOL_LOCK.yaml`.

## Outcome

The requested 200-update primary-detector timing, memory, finite-loss, and
validation-path measurement completed on training/development data only. The
first fp16 attempt stopped at update 1 on an infinite gradient norm; the
documented bfloat16 numerical diagnostic completed all 200 updates. All five
mechanical acceptance flags in `pilot_summary.json` are true.

A separate compact conditional-diffusion pilot also completed 200 bfloat16
updates at 64×64 and batch 16. It has 3,626,049 parameters, averaged 0.01824
seconds per post-warm-up update, used 503.12 MiB peak allocated CUDA memory, and
passed all five technical acceptance flags. It had no external pretraining and
used no test data.

## Data and leakage controls

- Training: 64 frozen manifest rows, one per 32-stratum × 2-view bucket, drawn
  only from the historical training role; 53 specimens.
- Validation: 64 rows under the same bucket rule, drawn only from validation;
  all 32 validation specimens and both views represented.
- Test: not loaded or used. The output receipt records `test_split_used=false`.
- Inputs came from the managed project copies, including the checksum-verified
  COCO initialization weights (`dd69338a…`).
- The sample-ID rosters are SHA-256 locked in the summary receipt.

## Interpretation

The detector pilot establishes a usable bfloat16 technical path at batch 4, and
the diffusion pilot establishes finite epsilon-prediction training mechanics at
64×64. They record actual compute/memory but do not resolve ROI Align bitwise
nondeterminism, label semantics, detector or generator efficacy, a plateau, a
common checkpoint, or the variance components required by W07. The validation
sequences are too small and reused to choose any scientific operating point.
A 10-step random-weight DDIM fixture validates deterministic finite reverse
sampling mechanics only; trained 50-step sampling and image-quality metrics
remain unvalidated.

## Remaining W06 gate

`HPO_BUDGET.json` intentionally leaves the non-PDE numeric strength grid,
selection criterion, stopping/checkpoint rule, practical AP margin, and
affordable trajectory tier unset. W06 therefore remains a partial blocked gate
until those choices, the diffusion full-fit/sampling budget, and the bfloat16
precision policy are approved before W08.
