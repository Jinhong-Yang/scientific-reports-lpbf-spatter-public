# W14 nested low-data protocol

Status: `FROZEN BEFORE W14 OUTCOMES`

Three outcome-blind specimen rosters are selected only from the 112-specimen
training partition.  Within every realization, 14 is nested in 28 and 28 is
nested in 56.  Selection uses standardized acquisition and box-summary
coordinates, initializes with one specimen from each material, and then uses
seeded farthest-point coverage.  No validation metric or held-out test payload
is used to choose a specimen.

The 108 detector trajectories cross three sizes, three subset realizations,
four arms (N1, NR, NB, NP), and three pipeline seeds.  NP is refit for every
size-realization-seed tuple, for 27 training-only generator fits.  A full-data
generator, latent bank, scaler, or donor pool may not be reused.  NB and NR are
rebuilt from the same subset and common target indices.

Training subsets contain 448, 896, or 1,792 images, but all use the fixed
32-specimen/1,024-image validation partition.  Results therefore report the
training-label and validation-label budgets separately.  The optional
synthetic-only diagnostic is not part of the approved 108 trajectories.

Execution is implemented by `src/detectors/run_w14_lowdata_detectors.py` and
inherits the frozen W12 detector architecture, optimizer, 8,960-update schedule,
validation milestones, checkpoint/resume behavior, and exact exposure ledger.
Each low-data roster completes an integer number of shuffled full-roster cycles,
so every frozen row has equal exposure within a trajectory. W14 detector
training remains gated until both the W12 core and all 27 subset-specific
generator pools are complete.

The operational target remains the inherited box region with unverified
physical semantics; independent human review was excluded by the user.
