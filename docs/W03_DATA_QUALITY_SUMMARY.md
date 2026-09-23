# W03 data-quality decision summary

## Decision

The recovered 5,632-row corpus is usable for bounded historical reproduction
and for designing a specimen-grouped N2 protocol. It is not yet usable as
human-validated synthetic supervision or as an external validation cohort.

Automated status: `PASS_WITH_LIMITATION`; critical failures: 0.

## Findings by severity

### Critical

None in the automated custody, manifest, split, file, hash, readability, JSON,
or bounding-box checks.

### High

- The 1,024-image historical test split has already been accessed. It cannot
  support a pristine or external-validation claim.
- No manufacturing-build identifier is evidenced. Specimen grouping cannot be
  promoted to independent-build grouping.

### Medium

- The semantic unit and visible correctness of the original box are unresolved
  and require W04 human calibration.
- Exact 64-bit dHash equality flags 2,422 rows, including cross-split groups.
  All image SHA-256 values remain unique; the dominant coarse-hash groups are
  sparse off-axis patterns. Treat this as a review queue, not as duplicate
  proof.
- Thirty-six identical label hashes cover 72 rows. Each group stays within one
  specimen and split, so no split leakage is implied; human/lineage review is
  still appropriate.

### Low / documented

- Optional annotation position fields are null in 2,816 rows; the required box
  coordinates are complete and valid in all 5,632 rows.
- `TS.z05`, `TS.z06`, and `TS.z07` remain unavailable, and the corpus is an
  accessible validation-archive subset rather than the full source dataset.

## Quality dimensions

- Completeness: required analysis fields and files complete; optional position
  fields partially absent.
- Uniqueness: unique sample IDs, manifest grain, and image byte hashes.
- Validity: split/view domains, dimensions, numeric fields, boxes, files, and
  content hashes pass.
- Consistency: all 32 strata occur in all three historical splits; source and
  derived manifests reconcile.
- Timeliness: not applicable to a static historical corpus; custody timestamp,
  version receipts, and test-access state are recorded instead.
- Integrity: specimen and specimen–view groups do not cross splits.

## Required action

Use the W04 blind review to resolve label semantics and to inspect a stratified
sample that includes dHash-flagged and unflagged images. Keep manufacturing
build blank unless new source evidence is supplied. Do not start full N2
training before W08.

## Subsequent scope disposition

The instructions above record the outcome-neutral W03 recommendation at the
time of that audit. Before N2 outcome access, the project decision owner
subsequently excluded W04 and approved W08 with the operational
`inherited_bbox_region` target, unverified physical semantics, grouped internal
evaluation, and the corresponding claim limitations. W04 reviewer assignment
is therefore not an outstanding gate. Full N2 execution proceeds only under
the frozen W08 protocol recorded in `docs/W08_APPROVAL_FORM.md` and
`configs/PROTOCOL_LOCK.yaml`.
