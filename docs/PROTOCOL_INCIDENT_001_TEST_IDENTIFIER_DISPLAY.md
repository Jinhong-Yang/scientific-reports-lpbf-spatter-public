# Protocol incident 001 — bounded display of historical test identifiers

Recorded: 2026-09-14T22:24:00+09:00  
Status: `CONTAINED_NO_N2_OUTCOME_ACCESS`

While locating the already materialized training and validation manifests, an
operator command displayed the header and first five rows of
`data/manifests/group_manifest.csv`. Two displayed rows named one specimen in
the historical `test` split. The displayed fields were group-level identifiers,
row counts, stratum metadata, and historical provenance fields. No test image,
label coordinate, prediction, metric, checkpoint evaluation, or N2 outcome was
read or computed.

The display was not used to choose a model, seed, fold, arm, endpoint, threshold,
or analysis. W16 development folds are therefore constructed exclusively from
the separate W09 train and validation manifests, and the N2 held-out image and
label payload remains locked. The already documented historical test split was
not pristine before this project; this incident nevertheless remains in the
audit trail because the new-study protocol prohibited test-manifest inspection.

Containment actions:

- no further use of the combined group or image manifest for pretest code;
- development-only W16 inputs are explicitly allow-listed as W09 train and
  validation manifests;
- the final pretest receipt must cite this incident and distinguish identifier
  display from the first N2 test image/label/prediction access;
- no scientific claim may describe the held-out roster as identifier-blind.

Scientific impact: the incident does not reveal or alter an N2 endpoint and does
not contaminate outcome-blind model selection, but it is a protocol deviation
that must be disclosed in the internal audit and reproducibility record.
