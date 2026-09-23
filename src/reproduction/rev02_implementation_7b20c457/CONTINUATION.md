# LPBF REV02 continuation and completion boundaries

The active user request is to execute the 2026-09-02 Korean REV02 work order and fully update the IEEE Access manuscript from the completed evidence. This is not a request to repeat the historical v49 experiments or deliver a narrative-only revision.

## Authoritative locations

- Revision root: `C:/Codex Research/metal_spatter_pinn/revision_rev02_20260902`.
- Large artifacts and retained checkpoints: `D:/Codex Research/LPBF_REV02_20260902`.
- Python: `C:/Codex Research/metal_spatter_pinn/.venv/Scripts/python.exe`.
- User work order: `C:/Users/WIN/Downloads/작업지시서_LPBF_WEAK_PHYSICS_REV02.md`.
- Scientific specification: the fourteen files bound by `protocol/hashes.json`; initial protocol commit `8e3aa248dd9849149561f7c938ab455392d7b5b0`.
- Preserved historical manuscript/package: `../revision_20260902/10_release/submission_release_20260902_v49`.

## Inspect before taking action

Read `PIPELINE_STATUS.json`, `MANIFEST.md`, `access_chronology.json`, the implementation lock and current command log. Check the recorded coordinator and child process identities; do not launch a second active coordinator. Only inspect the scoped REV02 processes, not unrelated experiments. An unchanged active training process is expected, not a reason to interrupt or restart it.

The public one-time split has 112/32/32 specimens and 3584/1024/1024 images. `data/train.csv` and `data/validation.csv` are public. **Do not open the sealed test manifest, test IDs, images or label JSON before the verified global T7 gate.** Historical corpus exposure is disclosed; this is not a new external holdout corpus.

## Managed execution

`scripts/run_rev02_pipeline.py run` serializes the frozen pretest work and deliberately stops at `pretest_runs_finished_acceptance_audit_required`. It does not unlock T7. `--max-commands` pauses only at a command boundary and does not change the planned roster. Completed commands are reused by identity; interrupted or failed commands stop for documented technical review. Preserve every failure record and seed. Do not delete a started receipt or replace a seed to make the pipeline advance.

No scientific changes, new ratio candidates, seed additions, architecture changes, batch reductions, retuning or source changes are permitted merely to improve outcomes. Any necessary implementation repair requires a transparent technical amendment preserving the original protocol, source locks, partial outputs and scientific comparability. Source identity failures are not permission to overwrite provenance.

## Before T7

1. Require all 60 core and 18 sensitivity generator fits/validation records, all three 7168-image pools, the 450 audit controls, complete 30-cell ratio grid, and all 45 or 57 unique budget trajectories with four milestones.
2. Validate the component `pretest_evidence()` outputs and all raw files, source/input/checkpoint hashes, matched real exposure, aliases, pool boxes, train-only calibration and audit contribution-demotion rule. Complete and inspect required pretest quantitative figures from locked validation results.
3. Run `pretest-freeze` only when all nine task components are complete. Independently rehash every target in every `PRETEST_COMPLETE.output_hashes`, and audit model rosters, counts, no missing cells, chronology and unchanged specifications.
4. Write a genuine acceptance-audit JSON with status PASS, the exact `PRETEST_FREEZE.json` hash, an empty failed-check list and a nonzero number of actually passed checks. A static/toy test report is not this acceptance audit.
5. Only then use `unlock --audit-report <accepted audit>` and the single fixed `test-campaign`. The campaign contains all fixed generator and detector checkpoints; it is not a single forward pass. Test results never select models, ratios, thresholds or hyperparameters. Any necessary repeated evaluation must be explicitly recorded and disclosed.

## Manuscript and final delivery

Update a new copy, preserving old source/PDF/zips. Keep approved author metadata, photos, biographies and exact AI-use disclosure. Use the clean v49 author order for the supplement too; its old hard-coded author order differs.

Rewrite claims only from complete REV02 results. Do not import historical effect sizes, audit rates or confidence intervals as new findings. In particular:

- New G3/D3 is prospectively omitted. Its historical PFFD eligibility failure stays explicitly historical/exploratory in the supplement.
- Ten generator seeds, exact sign-flip primary inference, degenerate p undefined, Holm families and 42-test BH sensitivity must remain explicit; no TOST or equivalence claim.
- B4 matches real exposure and optimizer updates, not compute or effective batch size.
- Low residual alone is not quality; G0, blur, real-real floor and matched-capacity G2N controls must be reported without assuming their direction.
- If interpolation-control sensitivity fails the frozen rule, demote zero audit flags to a limitation and remove the audit contribution claim.
- A lowest-grid ratio does not prove that every unmeasured ratio is unhelpful. Cross-family agreement/disagreement governs title scope; one previously examined corpus cannot justify broad manufacturing generalization.
- REV02 generated pools render at64 then bilinearly resize to300. Do not retain the old statement about direct continuous300 rendering.
- The actual GPU toy check emitted a torchvision ROIAlign-backward nondeterminism warning under deterministic warn-only settings. Do not claim bitwise determinism.

Target main paper: at most12 pages and10 tables, one consolidated detector table with both study-specific and official COCO metrics, a one-page portable eligibility checklist, and detailed supplement. Abstract has no internal arm codes. Required figures include budget comparisons, ratio grid, factorial effects and audit operating characteristics. Read the quantitative-visualization and PDF skills before producing those artifacts; render and inspect every final page. A current Overleaf package, all raw/result summaries, environment locks, run manifest and completed requirement-by-requirement `REVISION_RESPONSE.md` are required before claiming completion.

## Communication

During long running jobs, report only actual completions, meaningful failures or required user action. Do not repeat unchanged progress, promise a finished paper before evidence exists, or label v49 as the completed REV02 deliverable. Local background work requires the computer and app to remain running.
