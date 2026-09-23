# REV02 pre-execution interpretation record

This record resolves contradictions and over-strong claim templates in the supplied work order, `WO-LPBF-WEAKPHYS-REV02`, before experiment implementation. It preserves the requested controls, new internal split, multi-seed comparisons, locked evaluation, and manuscript revision. It does not prescribe favorable or unfavorable empirical results.

The numbered task protocols define executable settings. This record defines the interpretation and reporting boundaries those settings must satisfy. Any unresolved inconsistency must be corrected and all affected protocol hashes reissued before implementation or data-dependent execution.

## 1. Previously studied corpus and a prospectively locked internal re-split

The same 176-specimen corpus has already informed earlier analysis. Randomly assigning these specimens to a new 112/32/32 train/validation/test partition cannot erase that historical exposure or create a new independent corpus.

REV02 will therefore be described as a **prospectively locked internal re-split evaluation within a previously studied corpus**. The new test specimens are excluded from REV02 training, donor selection, threshold calibration, ratio selection, checkpoint selection, and other model-development decisions. They are not claimed to have been historically unseen by the study as a whole.

The manuscript may remove cumbersome historical evidence-class codes, including Route C and C2/C3/C4, while retaining plain-language disclosure of prior corpus analysis and the absence of an external or historically untouched reserve. Removal of codes must not become removal of material provenance.

New G3/D3 experiments are omitted prospectively from the REV02 arm roster. Historical G3/D3 failures remain in the preserved historical record; omission is not a successful eligibility re-evaluation and cannot retrospectively promote their earlier evidence.

## 2. Historical artifacts remain preserved in place

Original historical files will not be moved or overwritten because their paths and hashes are part of existing provenance. `results/historical/` will provide an index or snapshot referencing preserved v49 and earlier source records, with source hashes and an explicit preliminary/historical label. This implements the work order's separation objective without breaking the original evidence chain.

No completed unfavorable historical or REV02 run may be silently deleted, replaced, or reclassified as unexecuted. Failed runs remain represented in the run manifest, including the reason for any permitted same-seed technical recovery.

## 3. Split-steward access is distinct from modeling access

The deterministic split-steward process necessarily handles specimen identities while creating and sealing the partition. Its creation/write access is an explicit, narrowly scoped exception to the instruction not to access test IDs before T7. It must not render test images, inspect test labels for modeling, calculate test endpoints, or pass the sealed test payload to model-development code.

The chronology must separately record:

1. Protocol freeze and split-steward creation/sealing events.
2. T1–T6 training, validation, calibration, selection, and summary-freeze events.
3. T7 unlock and the first modeling/evaluation access to the sealed test payload.
4. Completion or failure of the single evaluation campaign.

File-system access timestamps or absence of a selection log alone do not prove non-access. The evidence is a reproducible execution record with content hashes, declared access boundaries, and explicit access events. Because a deterministic split seed and the complement of train/validation identities can reveal test identities, this is procedural test isolation, not an independently blinded third-party study.

## 4. Pretest completion and the one-campaign interpretation

`pretest_complete` for T1–T6 means that all required training and validation work, fixed checkpoints, ratios, thresholds, arms, milestones, endpoint definitions, inference methods, and analysis software are complete and sealed. Test fields remain explicitly `locked_pending_T7`; their absence at this stage is not an undisclosed missing result.

Only after these pretest requirements are satisfied may T7 evaluate the test payload. T7 is **one sealed evaluation campaign containing all predeclared models, seeds, and milestones**, not a requirement that only one model make one inference call. Model selection, budget choice, ratio choice, threshold choice, seed replacement, and endpoint expansion based on this campaign are prohibited.

Any technically necessary repeat must retain the failed campaign record, document the reason, and disclose the repeat. A restart must not be used to replace an unfavorable completed evaluation.

Final two-split generator statistics and detector-family conclusion comparisons are completed after T7; they cannot be prerequisites that force test access during T1–T6.

## 5. B4 uses the work order's explicit option (b)

The work order contains mutually inconsistent B4 prose: it requests matched optimizer steps, describes step expansion, and explicitly permits either step expansion (a) or expanded batches (b). REV02 chooses the explicitly permitted **option (b)**.

At each B4 optimizer step, the real batch contains four images, matching D0. For a synthetic-to-real ratio `r`, cumulative synthetic exposure after step `k` is `floor(4*r*k)`; the additional synthetic count at that step is the difference between consecutive cumulative counts. This permits exact cumulative ratios at the declared milestones without silently rounding small ratios to zero. The total batch can vary when the synthetic count is fractional per step. Its images are processed under the frozen training-loss normalization and optimizer rules.

B4 uses the same optimizer-step milestones and learning-rate clock as B2. At step `k`, each B4 arm has `4*k` real-image exposures and `k` optimizer updates. Augmented arms have additional synthetic images and therefore additional batch/compute work, **not additional optimizer updates**. Real and synthetic exposures, total updates, effective batch sizes, and runtime must be reported separately.

The budget-bias table must consequently state:

| Budget axis | Matched quantity | Remaining imbalance |
| --- | --- | --- |
| B1/B3 | Epoch count; each complete mixed-data traversal exposes every real image once | Augmented datasets require more total image processing and generally more updates per epoch |
| B2 | Optimizer steps and fixed total batch size | Augmented arms receive fewer real-image exposures |
| B4(b) | Real-image exposure and optimizer steps | Augmented arms process larger effective batches and more total images/compute |

Option (a), which expands steps by `(1+r)`, is not used. It would overlap with the existing complete-epoch real-exposure comparison. B4(b) is not a compute-matched experiment. Equal real exposure also does not imply identical weighting of real examples in the mixed-batch training loss, identical gradient noise, or an isolated causal effect of synthetic content.

## 6. Seven primary generator endpoints and required secondary SSIM

The established seven scalar primary endpoints are:

1. Fresh-collocation reference proxy RMS.
2. Reconstruction PSNR.
3. Standardized morphology-feature Fréchet distance (PFFD_std).
4. Within-condition `1-SSIM` diversity.
5. Boundary energy.
6. Moment centroid error.
7. Moment spread error.

Reconstruction SSIM is additionally reported as a required **secondary** endpoint. It is distinct from `1-SSIM` diversity and is not silently added to the primary factorial family. Thus the planned primary factorial roster remains `3 effects × 7 endpoints × 2 splits = 42 tests`. If any cells are excluded as degenerate or cannot be completed, both the planned count and the actually defined count must be disclosed.

G2-versus-G2N comparisons, coefficient sensitivity, and detector comparisons are separate analyses whose contrast rosters and multiplicity conventions must be stated in their task protocols; they are not hidden additions to the 42-test family.

## 7. Statistical non-separation is not equivalence

The work order's seed-SD criterion for G2N is a descriptive heuristic, not an equivalence test. The task protocol must state exactly which paired-seed difference and SD enter the criterion. Passing every declared endpoint permits wording such as **“not separated by the prespecified seed-variability criterion”**, not “proven equivalent,” “the PDE is unnecessary,” or a claim about all possible generators.

Likewise, a D5 contrast interval crossing zero, or a non-significant detector test, does not establish zero added utility. The manuscript must report the estimate and uncertainty and distinguish failure to establish benefit from proof of its absence. The unsupported historical TOST margin of ±0.014 is removed; no new practical-equivalence margin is inferred from these results.

Exact sign-flip inference is primary under its paired-effect sign-exchangeability/symmetry assumption. It is not described as assumption-free. For ten nondegenerate paired seed effects, the smallest attainable nonzero two-sided exhaustive sign-flip p-value is `2/1024`. Under the work order's conservative reporting rule, degenerate cells are labeled `undefined (degenerate)` and excluded from within-endpoint Holm adjustment rather than printed as `p=0`. The global BH sensitivity retains the nominal 42-hypothesis denominator using conservative internal placeholders of one for undefined cells; their public p and q values remain undefined and their count is disclosed. This is a declared reporting convention, not a claim that an exact sign-flip enumeration is mathematically impossible for every constant nonzero effect.

## 8. Conclusions remain conditional on observed evidence

The negative conclusion sentences supplied in the work order are examples of possible reporting branches, not mandatory experimental outcomes. The manuscript will be rewritten according to the completed results, including any observed augmentation benefit or detector-family disagreement.

Selecting the lowest tested ratio, 0.125, establishes a boundary selection within the frozen grid. It does not prove that no beneficial interval exists below the grid or between sampled ratios. “No benefit established within the evaluated configurations” is permitted only when supported by the corresponding estimates and uncertainty.

Agreement of two detector families supports scope limited to those two evaluated families and this corpus. It does not establish architecture-universal or external-build generalization. Disagreement must be reported and reflected in the title and conclusions without selectively dropping the less favorable family.

## 9. Mechanistic controls and audit interpretation

G2N removes the condition-dependent advection–diffusion–reaction residual but retains the explicitly defined isotropic Laplacian regularizer. Therefore “all derivatives removed” is not an accurate description. Its blob target, Laplacian formula, loss weights, attention policy, and sampling conventions must be fixed before training. Equal architecture can establish equal parameter counts and inference FLOPs, but does not automatically establish equal training cost.

Coefficient variants must be compared using a common, frozen reference-coefficient residual and fresh collocation points. Variant-own residuals may be reported as secondary diagnostics; changing a coefficient changes the residual scale and therefore cannot by itself demonstrate improved common-reference consistency.

The generator's `0.08*epsilon` perturbation is latent-space noise. T4 therefore constructs its interpolation-positive controls by interpolating validation-donor posterior means and adding the recorded latent vector before decoding with the frozen G2 checkpoint. These are known-donor-derived controls, not guaranteed pixel-near copies. A pixel-space blend with normalized-intensity Gaussian noise of standard deviation 0.08 would be a different mechanism and is not substituted for this control. The audit must disclose the space, units, clipping, and transformation order. The separate D5 pixel-blending arm has no added latent or pixel noise.

Flip-invariant retrieval must be applied consistently to calibration negatives, positive controls, and pool queries before threshold locking. Alpha-specific sensitivity, uncertainty, and aggregate sensitivity must all be shown. A weak interpolation-positive sensitivity downgrades a zero-flag pool result to an audit limitation according to the frozen gate; it does not certify novelty or absence of memorization. FPR measured on the same validation distribution used to calibrate thresholds is labeled a calibration/apparent FPR.

## 10. Implementation and acceptance boundaries

- Split membership is at the specimen level, including all associated images and views. Each exact stratum must support the required train/validation/test allocation before the split is accepted.
- New training-only normalization, donor banks, fitted feature transforms, synthetic pools, and checkpoints must follow the new split. Historical data-dependent caches must not leak across roles.
- A shared frozen donor-triple ledger is required for G1/G2/D5 correspondence. The new generator-seed composition and integer allocation of all 7,168 pool rows must be explicit.
- When independently selected D2 and D4 ratios differ, D5 cannot be described as simultaneously ratio-matched to both unless both matched configurations are evaluated. The task protocols must define the comparison-specific ratio policy before execution.
- Incomplete or failed required cells remain visible. An honest incomplete status takes precedence over a false “zero missing cells” claim; only predefined technical recovery may resume the same seed.
- No experiment result is asserted in this interpretation record. No test payload is needed to apply it.

## Sources inspected for this interpretation

- Supplied work order: `C:/Users/WIN/Downloads/작업지시서_LPBF_WEAK_PHYSICS_REV02.md`.
- Historical frozen protocol: `revision_20260902/02_protocol/REVISION_PROTOCOL_v1.yaml`.
- Historical factorial inference amendment: `revision_20260902/02_protocol/AMENDMENT_20260902_001_FACTORIAL_INFERENCE.yaml`.
- Historical detector execution contract and loader/training implementation in `revision_20260902/05_detector_robustness/`.
- Historical generator endpoint definitions in `revision_20260902/04_generator_ablation/scripts/evaluate_factorial.py`.
- Source definitions in `src/metal_spatter_pinn/physics.py` and `src/metal_spatter_pinn/inference.py`.
- Historical pool allocation in `scripts/generate_confirmatory_pools.py`.

Only instructions, protocols, and implementation definitions were inspected for this record. No new REV02 test identities, test payloads, or historical result tables were read.
