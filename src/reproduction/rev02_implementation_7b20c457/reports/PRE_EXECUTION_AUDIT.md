# REV02 independent pre-execution audit

Status: **PASS — INDEPENDENT SOURCE/PROTOCOL REVIEW FOR PRETEST EXECUTION ONLY; NOT A T7 UNLOCK OR EXPERIMENT-COMPLETION APPROVAL**

Record started: 2026-09-02 13:13:09 UTC.

Final bounded source review and independent toy-suite verification: 2026-09-02 13:36:57 UTC.

## Scope and frozen specification

- Protocol ID: `LPBF-WEAK-PHYSICS-REV02-20260902-v1`.
- Protocol commit reported by the coordinator: `8e3aa248dd9849149561f7c938ab455392d7b5b0`.
- Independently read `protocol/hashes.json` SHA-256: `d4cb8fa32ffdc36fbf0770558c92103e75461aad74ecbeae8df60d7d6b07014d`.
- The complete work order and all frozen protocol documents were reviewed for consistency. Earlier contradictions were resolved before this commit: prior-corpus exposure, B4 option (b), endpoint counting, the T7 chronology, common generator evaluation, contrast signs, provenance field naming, audit resolution, and detector multiplicity.
- The independent reviewer has read no REV02 test identifiers, test payloads, or experiment outcomes. This review does not itself implement an access-control boundary.

## Current disposition

| Layer | Status | Meaning |
| --- | --- | --- |
| Pre-code scientific specification consistency | PASS | Previously reported cross-document blockers were resolved before protocol freeze. |
| Data-free/synthetic regression tests | PASS | Final complete suite: 116 tests and 37 subtests passed; only source, protocol documents and fabricated temporary fixtures were used. |
| Shared access/provenance gates | STATIC REVIEW PASSED | Empty-receipt defects corrected; actual coordinator roster builders pass independent tests; gate writes now have a preflight and exact-content recovery policy. |
| Generator and pool implementation | STATIC REVIEW PASSED | Objectives, paired random streams, evaluation and pools agree with inspected design. Exact T7 roster/checkpoint guard is before test-plan preparation, including F00 blur sources. |
| Detector implementation | STATIC REVIEW PASSED | Core methods, frozen completion binding and empty-ledger rejection reviewed. The 32-camera-category guard was corrected to 32 unique sample IDs; the independent regression now passes. |
| Memorization/statistics implementation | STATIC REVIEW PASSED | Control transformations and statistical formulas reviewed; image/model/pool binding, nonempty ledgers and post-unlock audit-stage restriction inspected. |
| Pretest execution source readiness | PASS, SUBJECT TO COORDINATOR SOURCE FREEZE | No unresolved blocking defect in the reviewed source; the coordinator must commit/freeze these exact sources and retain its environment/input checks before starting. |
| T7 unlock | LOCKED/PENDING | Requires completed and hashed T1–T6 pretest work; protocol tests cannot unlock it. |

## Expected run-roster arithmetic (specification only)

These counts are calculated from the frozen design, not from executed data:

- Generator core: 6 unique arms × 10 seeds = **60 fits**. G1/F00 and G2/F11 are aliases, not additional fits.
- Coefficient sensitivity: 3 constants × 2 changes × 3 seeds = **18 additional fits**; total unique generator fits **78**.
- Main-family ratio selection: 2 neural arms × 5 ratios × 3 seeds = **30 trajectories**, two epochs each; **95,424 optimizer updates** in total under the frozen dataset-size assumptions.
- Let `u` be the number of distinct selected D2/D4 ratios, either 1 or 2. Main-family fixed-budget trajectories: `6 + 3*3*(2+u)` = **33 or 42**. Six real-only/conventional-augmentation trajectories serve prospectively identical budget aliases; they are not independent replicates.
- Second-family B4 trajectories: `3*(3+u)` = **12 or 15**.
- Total detector training trajectories, including ratio selection: **75 or 87**. Selection candidates are not test-evaluated.
- Unique planned generator/detector checkpoint evaluations in T7: `78 + 4*(main-budget trajectories + second-family trajectories)` = **258 or 306**, excluding non-model blur/real-real diagnostics and without double-counting aliases.
- Pools: 7,168 images per G1/G2/D5 arm, allocated across ten generator seeds as eight counts of 717 and two of 716.
- Audit: 9 control classes × 50 = **450 controls**, including four paired donor-interpolation alpha levels.
- Factorial nominal test count: 3 effects × 7 primary endpoints × 2 splits = **42**. Reconstruction SSIM is mandatory secondary.
- Detector multiplicity: **48** primary contrasts for the main family and **16** for the second family, separately per split; D1 context is outside this primary family.

## Required independent code-review checks

### Shared preparation and access control

- Only the one-time metadata steward may read the combined source manifest. No test image bytes or label JSON may be opened in that stage.
- Test payload resolution must fail closed before a verified global unlock receipt. A caller must not bypass the guard with a supplied path, default split, legacy loader, or self-authored success flag.
- Pretest receipts must be checked by content hash, required task completeness, and actual chronology, not only by file existence or a mutable `status` string.
- Event-chain hashes, protocol hashes, environment/code/input hashes, and checkpoint identities must be verified on resume/reuse.
- Split creation is once only; partial failure must not silently regenerate a new partition. Historical data and artifacts remain untouched.

### Generator and pool behavior

- Actual losses, weights, warmups, paired initialization and independent random streams must match T1/T3A/T3C.
- Fresh evaluation uses the frozen common-coordinate and donor plans, train-only scaler, 128 balanced targets, seven primary endpoints, and the separate secondary SSIM.
- Synthetic targets are canonicalized once; JSON/CSV boxes agree. G1/G2/D5 share the complete ordered donor/alpha/epsilon/condition plan and seed allocation.
- Coefficient variants use common-reference RMS; native-coefficient RMS is secondary. No historical checkpoint or fitted scaler is reused.

### Detector behavior

- Exactly 30 complete validation candidates precede ratio selection. Ties and D5's selected-ratio union follow the frozen rules.
- B4 has four real images at every update and additional synthetic count `floor(4*r*t)-floor(4*r*(t-1))`; it performs one expanded-batch update and matches B2's learning-rate clock.
- Exposure counting uses explicit `source_kind`, never path-prefix inference. B1/B3 full epochs and B2 mixed-batch counters are logged.
- Cross-axis D0/D1 aliases use the same verified trajectory rather than pretending to be independent runs.
- All milestones and full resumable states are retained. Completed unfavorable seeds cannot be replaced; OOM cannot silently change the batch protocol.
- RetinaNet head adaptation, internal/reporting label mapping, and official COCO companion metrics must be checked on toy inputs.

### Memorization and inference

- A fixed 64×64 canonical representation is used for every query and reference. Flip-aware calibration precedes the immutable threshold lock, and that lock precedes controls and pools.
- Latent interpolation controls use the declared frozen G2 checkpoint, validation-only donors, matched alpha pairs, and stored latent epsilon; D5 is the separate noiseless pixel blend.
- Alpha-specific counts/Wilson intervals and the predeclared contribution-demotion rule are retained without retuning.
- Exact sign flip, degeneracy handling, Holm and nominal-42 BH placeholder accounting require independently checked toy edge cases.
- Crossed bootstrap preserves seed/specimen pairing; contrast signs and 48/16 multiplicity families match the protocol. Unadjusted intervals are not family-wise significance claims.

## Interpretation caveats to retain in the manuscript

- A fresh internal re-split cannot erase the corpus's prior analytic exposure.
- G2N matches architecture and inference capacity, not every loss scale; conclusions concern this prespecified smoothing/blob control.
- Real-image exposure matching does not also match compute, effective batch size, gradient noise, or the loss weight of real images.
- Sign-flip enumeration is not assumption-free. A non-significant result, an interval crossing zero, or the descriptive seed-SD rule is not equivalence.
- A lowest-grid ratio selection does not establish the absence of a beneficial unmeasured ratio.

No quantitative finding or experiment-completion claim is made by this document.

## Test execution record

The data-free protocol suite passed with `21 passed, 12 subtests passed in 0.10s` under the project environment. Its first draft had two reviewer-test assumptions corrected (different explicit D5 key names in T2/T6B, and an unnecessary literal-word assertion on the metadata-steward description). Neither correction changed a frozen protocol or scientific choice.

Command: `python -m pytest -q revision_rev02_20260902/tests/test_rev02_protocol_contract.py -p no:cacheprovider`, with Python bytecode writing disabled.

After adding three implementation-gate negative tests using temporary fabricated metadata, the first combined run reported **3 failed, 21 passed, 12 subtests passed**. The root owner corrected the implementation without modifying the scientific protocol. Re-running the independent contract suite together with the common-module tests produced **31 passed, 12 subtests passed in 0.21s**.

Two additional independent tests import only side-effect-free coordinator/common code and exercise the pure initial-job and budget-roster builders. All 25 possible D2/D4 selected-ratio pairs preserve the unique 45/57-trajectory roster, D0/D1 aliasing and second-family scope. The updated combined result was **33 passed, 37 subtests passed in 0.20s**. These tests do not call source/data/result loaders or begin execution.

A further independent fabricated-specimen test exposed a newly introduced detector guard that demanded 32 distinct `view` values instead of 32 distinct image IDs. It intentionally supplies 32 unique sample IDs across the two camera categories, plus a duplicate-ID negative case. The first run was **1 failed, 33 passed, 37 subtests passed in 1.70s**. This must pass after owner correction; real data were not used to discover the issue.

The generator, audit/statistics and pipeline component toy suites were independently run together after the reported source corrections: **51 passed in 3.58s**. These suites use synthetic arrays/temporary fixtures and do not access study test data.

After the detector specimen correction, the complete synthetic/data-free suite passed independently: **111 passed, 37 subtests passed in 4.86s**, with one expected torchvision warning from the deliberately non-pretrained toy head test. A later lexicographic T7 command-order regression was added by the coordinator owner.

A final independent negative fixture then exposed that an unlocked test manifest was not yet compared with the immutable public split's `sealed_test_hashes`. Campaign permission alone must not permit a changed partition. This fabricated fixture initially produced **1 failed, 27 passed, 37 subtests passed in 1.61s** and is retained as an independent regression. It creates no study identifiers or payloads.

The owner corrected sealed-content verification without changing the frozen scientific protocol. The reviewer inspected the permission-first guard, public split/receipt verification, exact two-entry sealed ledger and both content hashes, then independently reran every test: **116 passed, 37 subtests passed, 1 expected toy-only warning in 4.73s**. The final command was `python -m pytest -q revision_rev02_20260902/tests -p no:cacheprovider`, with bytecode writing disabled. The warning concerns torchvision's trainable-backbone fallback in a deliberately non-pretrained toy construction, not a failed experiment.

## Initial common/preparation implementation findings

| ID | Severity/status | Finding and acceptance criterion |
| --- | --- | --- |
| GATE-01 | Initial defect corrected | `verify_protocols()` now requires the full fourteen-file specification and verifies each content hash. The independent contract additionally pins the reviewed hash-receipt bytes. |
| GATE-02 | Initial defect corrected | `verify_implementation_lock()` now requires every experiment component, valid commit identity, protocol binding, and source hashes; run provenance checks the environment metadata and pip-freeze hashes. |
| GATE-03 | Corrected; static coordinator review complete | `require_test_unlocked()` rejects empty/partial task sets and checks all nine task receipt identities, counts, completion states, implementation binding, complete unique model rosters and chronology. Exact source checkpoint/completion hashes are additionally checked by model evaluators before test data access. |
| PREP-01 | Corrected; static metadata preparation review PASS | `prepare_split()` now requires clean committed preparation/common source and protocol-commit ancestry, and records both source hashes and preparation commit before split creation. This is a separate preparation-source receipt; full model execution still requires the later complete implementation lock. |
| PREP-02 | Reviewed boundary | In the inspected preparation loop, test rows copy existing metadata geometry/hashes but do not enter the branch that opens image bytes or label JSON. Canonical label writes are limited to train/validation. This is a static code observation, not an execution receipt. |
| LOAD-01 | Corrected | `manifest_path()` now validates the public split receipt and the selected train/validation manifest hash before returning a path to a model loader. |

The three gate tests create only toy metadata under temporary directories. They do not create or access the real study split, sealed test IDs, images, or annotations. Findings were sent to the root implementation owner; no owner code was modified by the independent reviewer.

## Model and analysis implementation review in progress

| ID | Status | Observation / required closure |
| --- | --- | --- |
| GEN-01 | Static review and toy tests complete | Generator loss weights, common coefficient reference, paired per-step random streams, seven-primary/SSIM-secondary evaluation, train-only scaler, balanced diversity, donor-plan correspondence and coordinator summary interfaces match the inspected frozen contracts. |
| GEN-02 | Owner correction inspected | `load_fitted()` now recomputes the expected current run identity/provenance rather than trusting the identity stored in the same checkpoint directory. |
| GEN-03 | Owner correction inspected | Generator source-image loading now checks the recorded image hash, and training/pool/scaler creation have explicit post-unlock restrictions. |
| GEN-04 | Owner correction inspected | `verify_test_roster_entry()` requires exactly one canonical arm/seed entry and matching checkpoint path/hash before test-plan preparation. Test blur controls verify all ten F00 roster entries first. |
| DET-01 | Core static review completed | B4's exact rational synthetic count, one expanded-batch update, learning-rate clock, D0/D1 canonical trajectory aliases, ratio-grid tie selection, and RetinaNet foreground mapping agree with the protocol. |
| DET-02 | Owner correction inspected | Component-specific `PRETEST_COMPLETE.json` writes were removed. `pretest_evidence()` now supplies the coordinator-owned canonical receipt builder. |
| DET-03 | Owner correction inspected | Test run-ID membership and exact frozen completion SHA-256 are checked before test-row loading. Post-unlock training/selection guards and completed pool artifact ledgers are present. Empty/malformed ledgers are rejected. |
| DET-04 | Owner correction inspected and independently tested | `_validate_specimen_roster()` now requires 32 distinct nonempty `sample_id` values and permits repeated ON_AXIS/OFF_AXIS categories. The independent fabricated 16+16-camera fixture passes and a duplicate sample ID is rejected. |
| STAT-01 | Core numerical static review completed | Exact sign flips, degenerate handling, Holm, nominal-42 BH, crossed bootstrap, contrast directions, and 48/16 primary multiplicity families match the inspected protocol. |
| STAT-02 | Owner correction inspected | `contextual_D1()` now emits D1-minus-D0 effects/intervals on all twelve main-family budget cells per split, explicitly outside primary Holm families. |
| STAT-03 | Owner correction inspected | Statistics now validates completed generator artifact identities, nonempty detector aggregate/selection ledgers and their bound inputs before computing outputs. |
| AUDIT-01 | Core static review completed | The common 64-pixel pipeline, orientation-aware nearest lookup, SSIM-at-feature-nearest convention, q01/q99 rule, nine control classes, latent-alpha pairing, Wilson intervals and contribution-demotion rule agree with T4. |
| AUDIT-02 | Owner correction inspected | Audit source bytes now match frozen row hashes; pool identities/rows and the F11 control fit are checked against current completed generator ledgers. Vacuous receipt maps and initiation of new audit stages after T7 unlock are rejected. Required task-root raw aliases and their hash index are written immutably. |
| COORD-01 | Static review and pure tests PASS | Initial stage has 78 generator fits plus validations, fixed controls/pools/audit, thirty ratio candidates and one selection action. Budget stage has exactly 45 or 57 unique trajectories. Stage-prefixed command receipts prevent a posttest summarizer from being skipped because of a pretest summarizer receipt. |
| COORD-02 | Owner correction inspected | `unlock()` now calls public freeze validation before writing the permanent marker. `freeze_pretest()` preflights component evidence, identities, timestamps and model roster, and accepts interrupted partial task receipts only when their content is identical apart from newly proposed creation time. |
| COORD-03 | Owner correction inspected | T7 generator and detector commands are now ordered lexicographically by their stable run IDs through the pure `test_jobs()` builder; milestone order remains ascending. |
| LOAD-02 | Owner correction inspected and independently tested | After permission succeeds, `manifest_path(test)` verifies the immutable public split/receipt and both sealed manifest/specimen hashes before returning the test path. Missing/incomplete hashes or changed content are rejected by fabricated regression tests. |

At the eventual T7 acceptance audit, independently rehash every target referenced by `PRETEST_COMPLETE.output_hashes`; common runtime validation verifies the receipt hashes, while model loaders additionally bind the exact frozen checkpoint/completion hashes. Figures, complete real-data rosters/results, failure logs and manuscript claims still need the separate post-training acceptance audit. No source review substitutes for those checks.

## Final bounded decision and operational limitations

All blocking findings raised in this source review are cleared in the inspected implementation. The exact reviewed sources are ready for the coordinator's commit/implementation lock and pretest-only execution. This PASS is not a success receipt for an unrun experiment, a T7 acceptance audit, a manuscript acceptance decision, or permission to revise the frozen scientific design.

- No real-data training, validation outcomes, REV02 test IDs, test manifests, test pixels or test labels were inspected by this reviewer. Full-training numerical stability, actual GPU resource demand and real-input integration therefore remain runtime checks, not established findings.
- The shared setup enables deterministic algorithms with `warn_only=True`; it does not guarantee bitwise-identical CUDA execution. The coordinator reports an ROIAlign-backward nondeterminism warning in its separate synthetic GPU smoke check. Retain that warning and its environment details in the runtime record, and describe seeded/reproducible protocol controls without claiming bitwise determinism.
- The access controls are procedural application-level gates. They are not OS isolation, external blinding or evidence that the previously studied corpus has become historically unseen.
- Training failures, incomplete evaluations, source/environment drift and changed artifacts must stop the coordinator and retain their logs. Recovery is an explicit technical decision on the same frozen cell/seed; no favorable-seed replacement, silent batch change or new split is authorized. Some incomplete-evaluation paths deliberately require manual documented recovery rather than an automatic retry.
- At T7, the coordinator's independent acceptance audit must rehash all referenced outputs, verify every actual required unit, inspect figures and required storage outputs, reconcile all failure/recovery records, verify the sealed split, and record true chronology before unlock. The protocol-required unsealed split export belongs to that post-unlock metadata workflow, not to this review.
- T8 layout, citations, author/AI-use metadata, submission package compilation/rendering and requirement-by-requirement evidence reconciliation remain separate downstream tasks. Detector utility and G2N/memorization conclusions remain conditional on the complete prespecified results.
