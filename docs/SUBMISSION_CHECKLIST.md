# Scientific Reports submission checklist

Status: `AUTOMATED ITEMS PREPARED; AUTHOR ATTESTATIONS PENDING`

## Portal metadata

- Article type: `Article`
- Title: `Optical residual regularization for LPBF spatter synthesis and detection`
- Keywords: additive manufacturing; laser powder bed fusion; synthetic data;
  optical residual; object detection; reproducibility
- Corresponding author currently shown in the draft: Jinhong Yang,
  `jinhong@inje.ac.kr`
- Authors currently shown in the draft, in order: Hyojin Park; Nam-Hyun Yoo;
  Jinhong Yang
- Ethics declarations: no human participants, identifiable human data, animals,
  or clinical intervention are involved in the described image-computing study.
  The portal's exact not-applicable selections remain an author action.
- Data repository: AI-Hub dataset 71476 is the governed source; licensed source
  pixels are not redistributed. The final GitHub release supplies shareable
  numerical evidence, code, manifests, and reconstruction instructions.
- Code repository and version: final public GitHub URL and immutable `v1.0.0`
  release, to be verified after W19 audit.

## Files prepared for upload

- `Scientific_Reports_LPBF_Manuscript.pdf`
- `Scientific_Reports_LPBF_Supplementary_Information.pdf`
- `Scientific_Reports_LPBF_Cover_Letter.pdf`
- `Scientific_Reports_LPBF_Initial_Submission_Overleaf.zip`
- High-resolution vector PDF figure files 1--4 inside both the Overleaf source
  package and the journal-upload package
- Numerical-evidence and reproducibility archives with SHA-256 manifests
- `docs/FINAL_HANDOFF_REPORT.md` and `evidence/manuscript/W21_HANDOFF.json`
  generated after the final release URL is fixed

The journal-upload ZIP contains only the three submission PDFs and four
separately numbered vector-PDF figures. Internal review reports, readiness
records, checklists, and working Markdown files remain outside that ZIP.

## Automated gates before release

- [x] W11 quality evidence complete (`PASS_WITH_SCOPE_LIMITATION`; all 50
  common-roster cells complete, with adverse ND diagnostics disclosed)
- [x] W12 detector matrix complete (100/100 frozen trajectories; full integrity
  audit passed; held-out test access absent)
- [x] W14 low-data matrix complete (27/27 generator fits and 108/108 detector trajectories complete; full integrity audit passed; held-out test access absent)
- [x] W15 resolution audit complete (40 validation-only fixed-checkpoint audits;
  no new training and no held-out test access)
- [ ] W16 grouped out-of-fold matrix complete (9 training-fold-only generator
  refits, 36 development-only detector trajectories, and full integrity audit
  passed; held-out test access absent)
- [ ] W17 single held-out campaign complete
- [ ] W17 paired statistics complete
- [ ] Results, abstract, discussion, tables, and four figures regenerated
- [ ] Numerical map and generated-fragment hashes pass
- [ ] All three PDFs build without unresolved citations or major layout errors
- [ ] Every final PDF page receives visual inspection
- [ ] Internal, non-independent review reports minor-revision-or-better readiness
- [ ] Public archive audit excludes source pixels, checkpoints, predictions,
  private paths, credentials, and unused independent-review materials
- [ ] Public Git URL, tag, release assets, and hashes resolve after upload
- [ ] Every public release asset has been freshly downloaded and matched to the
  final local file by SHA-256 (`scripts/verify_public_release.py --write`)
- [ ] Integrated project-completion audit passes every copy, experiment,
  manuscript, package, and public-release check

## Source-form readiness audit (outcomes still locked)

Checked against the current Scientific Reports submission guidance on
2026-09-16. This is a non-independent, pre-results formatting audit; it does
not replace the final W20 internal review after W17 statistics and figures are
sealed.

- [x] Title is a single scientifically accurate sentence of 9 words (within the
  journal's 20-word guidance).
- [x] The generated abstract is unstructured, contains 51 words (within the
  200-word guidance), and contains no citations or figure references.
- [x] Six indexing keywords are supplied (within the journal's six-keyword
  allowance).
- [x] The main-text source includes Introduction, Results, Discussion, and
  Methods; the remaining generated result sections remain deliberately
  outcome-neutral until W17 unlock.
- [x] Separate Data Availability and Code availability sections are present.
  They accurately defer the public release URL/version until the post-audit
  release is immutable, and disclose the licensed source-image restriction.
- [x] A Methods disclosure identifies the scope of AI assistance and preserves
  human author accountability.
- [x] Statistical methods specify the primary endpoint, comparison family,
  grouped resampling unit, draw count, interval policy, and non-claim rule.
- [ ] Final reference-style, PDF-layout, figure-file, and supplementary-file
  checks remain pending regenerated W17 outcomes and final rendering.

## Author-only confirmations before journal submission

These items cannot be inferred or attested by automation. They do not prevent
preparation of the scientific package, but they prevent submission on the
authors' behalf until confirmed in the journal portal.

- [ ] Author names, order, affiliations, and corresponding-author details
- [ ] CRediT/contribution statement and approval by every author
- [ ] Funding agency wording and grant number
- [ ] Competing-interest declaration from every author
- [ ] Exclusive submission, prior publication, related manuscripts, preprints,
  theses, and prior Editorial Board discussions
- [ ] Suggested reviewers and reviewers to exclude, or an explicit choice of none
- [ ] AI-assistance disclosure accepted by all authors
- [ ] Final manuscript, supplement, figures, data/code statements, and cover letter
- [ ] Any portal-specific data-access, ethics, and licensing attestations

The actual journal-submit action is not performed by this project automation.
