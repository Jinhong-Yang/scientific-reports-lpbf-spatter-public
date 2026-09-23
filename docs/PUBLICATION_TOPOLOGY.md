# Public repository topology

Decision time: 2026-09-15T00:04:00+09:00  
Decision basis: pre-publication inspection of the complete Git object history

## Decision

The development repository
`Jinhong-Yang/scientific-reports-lpbf-spatter` remains private. The final public
record will be created at
`Jinhong-Yang/scientific-reports-lpbf-spatter-public` from the audited,
source-data-free allow-listed archive only.

## Reason

The private development history contains seven tracked recovery inputs:
historical manuscript/source ZIP and PDF files, a recovered implementation ZIP,
and the three user-supplied instruction documents. It also contains detailed
development-run receipts that are not selected for the public package. Removing
those paths from the current branch would not remove them from earlier Git
commits. Making the development repository public would therefore expose files
outside the W19 publication allow-list.

## Clean publication contract

1. Build `Scientific_Reports_LPBF_Reproducibility_v1.0.0.zip` exclusively from
   the tracked W19 allow-list and pass its private-path, credential, source-data,
   prediction, checkpoint, and excluded-review audits.
2. Extract the archive to a new temporary directory, verify every entry against
   its SHA-256 ledger, and run the public-package tests there.
3. Initialize a new root Git history from that verified extraction. Do not copy
   `.git`, `inputs/`, `historical/`, `runs/`, raw links, prediction payloads, or
   model checkpoints from the development repository.
4. Record the private source-study commit in the public release manifest. The
   new public repository necessarily has a different root commit; the manifest
   provides the auditable link without making private history reachable.
5. Tag the clean public commit `v1.0.0`, attach the three final PDFs, three
   submission ZIPs, and reproducibility ZIP, and verify all seven assets by a
   fresh download and SHA-256 comparison.
6. Keep the development remote private after publication. No history rewrite or
   deletion of the user's recovery material is required.

The gated publisher is `scripts/publish_clean_public_repo.py --execute`. It
refuses to run before W20, rejects an existing repository name, safely extracts
the archive without traversal or Git metadata, verifies every checksum and the
public test suite, creates the new root history, uploads the seven assets, and
then runs the fresh-download release verifier.

This topology satisfies the requested Git publication while preserving the
licensed, historical, and working-material boundaries documented in W19.
