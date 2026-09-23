# Scientific Reports argument map

Status: outcome-neutral map frozen before N2 held-out outcomes  
Scope authority: user-approved W08 protocol with W04 independent-rater review
and W13 corrected-label interaction excluded

This map connects each manuscript-level statement to the experiment, endpoint,
machine-readable source, and language boundary that can support it. It does not
preselect an effect direction. Final prose must be regenerated from the listed
sources after W17 and must pass the numerical-hash audit.

## Central argument

The study evaluates whether a dimensionless optical-residual regularizer changes
image-domain generator behavior and downstream detection of the inherited
`inherited_bbox_region` target when architecture, donor support, real exposure,
training budget, and evaluation access are controlled. The work does not
validate a thermal or fluid model, the physical semantics of the inherited
boxes, or external generalization.

## Claim-to-evidence map

| Manuscript claim class | Required experiment or control | Primary endpoint | Authoritative source after completion | Permitted result language | Prohibited extension |
|---|---|---|---|---|---|
| Optical regularization changes the fitted image field | W09 paired 2x2 generator factorial | Paired validation MSE, optical-operator and morphology effects across ten seeds | `evidence/generator_factorial/W09_ANALYSIS.json`; `factorial_seed_metrics.parquet` | Direction, interval, and multiplicity-adjusted result for the tested operator and weight | Physical fidelity, thermal accuracy, causal LPBF dynamics |
| The selected optical weight lies on a measurable regularization path | W10 six-level optical-weight sweep | Validation reconstruction, common/own residual, gradient, Laplacian, radial PSD | `evidence/prior_sweep/W10_ANALYSIS.json`; W10 source tables | Descriptive path and uncertainty for tested weights | Globally optimal weight or physics calibration |
| Non-PDE smoothing does or does not reproduce the selected trade-off | W10 frozen Gaussian and heat controls | Reconstruction-match gap and corresponding morphology/operator metrics | `evidence/prior_sweep/W10_ANALYSIS.json` | Matched comparison only if the 5% gate passes; otherwise `unmatched closest control` | Equivalent control when the match gate fails |
| Generator families differ in image-domain quality | W11 NB/NF/NP/NS/ND common-roster evaluation | PFFD10, FID and frozen auxiliary metrics with shared target roster | `evidence/stronger_generator/W11_QUALITY_COMPLETENESS.json`; W11 source matrices | Conditional comparison on the common roster with uncertainty | Human-perceived realism or physical validity without human/physical evidence |
| Synthetic supervision changes held-out operational-target detection | W12 primary detector matrix and the single W17 test campaign | Pooled COCO AP@[.50:.95] on held-out real images | `evidence/detector_core/CORE_COMPLETENESS.json`; `evidence/test_campaign/W17_TEST_COMPLETENESS.json`; `evidence/final_statistics/ALL_COMPARISONS.parquet` | AP and paired interval for the frozen inherited target | Verified particle, cluster, or physical-spatter detection accuracy |
| NP differs from donor, data-only, augmentation, or exposure controls | W12 plus four frozen W17 primary contrasts | NP-NB, NP-NF, NP-N1, NP-NR paired effects; 95% and Bonferroni 98.75% intervals | `evidence/final_statistics/W17_STATISTICS_COMPLETENESS.json`; `evidence/final_statistics/ALL_COMPARISONS.parquet` | Positive, adverse, small, or unresolved according to the frozen estimand and intervals | Equivalence/no effect from a zero-crossing interval; superiority outside tested arms |
| Findings depend on detector architecture | W12 Faster R-CNN primary and RetinaNet secondary cells | Architecture-specific pooled AP and paired contrasts | W12 completeness records; W17 comparison matrix | Architecture-specific consistency or divergence | Architecture-independent generality |
| Findings vary with real-data availability | W14 three nested sizes and three outcome-blind realizations | Size-specific pooled AP and paired contrasts | `evidence/low_data/W14_GENERATOR_COMPLETENESS.json`; `evidence/low_data/W14_DETECTOR_COMPLETENESS.json`; W17 low-data tables | Conditional low-data sensitivity across the tested rosters | General sample-efficiency law or extrapolation beyond 14/28/56 specimens |
| Findings are sensitive to the resolution path | W15 fixed-checkpoint native-versus-downsample audit | Validation-only pooled AP difference for the declared transform | `evidence/resolution/W15_COMPLETENESS.json` | Validation-only resolution-path sensitivity | Benefit of unavailable native 1024-pixel information |
| Findings persist or vary across development groups | W16 three seeded, three-fold grouped OOF analysis | Out-of-fold pooled AP and paired effects | `evidence/grouped_internal/W16_GENERATOR_COMPLETENESS.json`; `evidence/grouped_internal/W16_DETECTOR_COMPLETENESS.json`; grouped source tables | Protocol-frozen internal grouped evaluation | Independent cohort or external validation |
| Test conclusions were not selected after outcome access | W08 protocol lock, access ledger, W17 pretest freeze and one unlock | Hash identity and access chronology | `configs/PROTOCOL_LOCK.sha256`; `docs/TEST_ACCESS_LEDGER.csv`; `evidence/pretest/PRETEST_FREEZE.json`; `evidence/pretest/TEST_UNLOCK.json` | Protocol-frozen held-out evaluation with the documented identifier-display incident | Pristine never-seen cohort or absence of all prior identifier exposure |
| Results are reproducible from shareable evidence | W18-W21 numerical mapping, clean build, archive extraction and checksum tests | Exact source-row hashes, generated-fragment hashes, package SHA-256 verification | `evidence/manuscript/NUMERIC_QA.json`; `BUILD_QA.json`; release manifests | Recalculation from released aggregate/source matrices and authorized-data reconstruction path | Public availability of licensed pixels, private paths, predictions, or checkpoints |

## Result-branch rules

- A simultaneous confidence interval wholly above or below zero may support a
  directional statement for that frozen contrast; practical importance must
  still be discussed in the scale of pooled AP.
- An interval crossing zero is reported as resolving effects in both directions
  or as inconclusive at the declared precision. It is not evidence of
  equivalence or no effect.
- A failed or missing cell remains in `FAILURE_LEDGER.csv`; no seed, arm, or
  endpoint may be removed because of its performance.
- If generator quality and detector utility disagree, both observations remain
  in the argument. Image fidelity is not used as a surrogate for detection.
- If grouped OOF or low-data results differ from the held-out result, the
  heterogeneity is reported rather than averaged into a single broad claim.

## Fixed limitations that must survive editing

1. The inherited bounding region is an operational dataset target. Independent
   human rating, corrected labels, inter-rater agreement, and verified
   physical-object semantics are outside the user-approved scope.
2. The 32-specimen test split is held out from N2 development but belongs to a
   historically accessed corpus. The bounded display of two group-view test
   identifiers is disclosed in the access ledger and protocol-incident report;
   no N2 test pixels, labels, predictions, or metrics were used before unlock.
3. No same-task independent external cohort was located. W16 is internal
   grouped out-of-fold evaluation only.
4. The optical residual is dimensionless and image-domain. It is not a
   calibrated governing equation or evidence of thermofluid correctness.
5. Public reproducibility material excludes governed AI-Hub source pixels and
   other restricted runtime payloads; access and reconstruction instructions
   must state this plainly.

## Manuscript assembly order

Methods remain tied to frozen configuration and receipt hashes. After W17,
generate Results, Discussion, Introduction edits, and Abstract in that order;
then rebuild all figures, editable supplementary tables, legends, data/code
statements, cover letter, and the numerical source map. W20 may classify the
package as minor-revision-ready only after every headline number maps to one of
the sources above and all final PDF pages pass visual inspection.
