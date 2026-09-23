# W17 single locked N2 test campaign

Status: `FROZEN BEFORE FIRST N2 TEST PAYLOAD ACCESS`

The N2 test campaign contains 208 final-checkpoint evaluations: all 100 W12
core trajectories and all 108 W14 low-data trajectories. Only update 8,960 is
eligible. W16 remains a development-only out-of-fold analysis and W15 remains a
validation-only resolution sensitivity; neither adds a test checkpoint.

Before the test manifest can be materialized, the gate requires complete,
test-free receipts for W09--W12 and W14--W16, including W11 quality, W15
resolution, the complete checkpoint roster, failure accounting, the frozen
environment, and the source/config hashes. These are sealed in
`evidence/pretest/PRETEST_FREEZE.json`. A separate matching unlock receipt is
then written before the first test payload read.

Every test checkpoint is evaluated once within the same named campaign. A
failed or interrupted transaction resumes with the identical checkpoint and
run identity; it cannot replace a seed or overwrite a completed result. Test
outcomes cannot select checkpoints, ratios, thresholds, arms, endpoints, or
additional analyses. Raw predictions, pooled COCO metrics, and specimen-level
metrics are retained.

The split was used by historical work before N2 and is not described as a
pristine external cohort. It is nevertheless held out from all N2 model fitting
and outcome selection. The bounded pretest identifier-display deviation is
separately disclosed and revealed no image, label coordinate, prediction, or
metric.

The 20,000-draw paired bootstrap is decomposed into 50 independent arm-by-seed
cell calculations and evaluated by ten CPU workers after the held-out campaign.
Every worker receives the identical frozen specimen-draw matrix; paired
pipeline resample counts are then applied to the returned cell vectors. This is
an exact scheduling optimization of the predeclared estimator, not a change in
resampling, confidence levels, contrasts, or endpoint.
