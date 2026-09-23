# W19 public-release audit policy

Status: `CODE_READY_AWAITING_FINAL_RESULTS`

Provisional audit v0.0.2 passed on 14 September 2026: 224 tracked files were
selected, no source pixels or checkpoints were present, the archive was
extracted into an isolated directory, all 224 SHA-256 entries verified, and the
seven public-package tests passed. This is a tooling rehearsal, not the final
v1.0.0 release; the audit will be repeated after the frozen results, manuscript,
figures, and W20 review evidence are committed.

The Git remote is private during execution. The final public archive is built
from an explicit allow-list of tracked code, configurations, manuscript files,
tests, documentation, and shareable numerical evidence. It excludes historical
snapshots, licensed source pixels, raw-data links, human-review material,
secrets, local environments, dependency vendors, feature caches, model weights,
checkpoints, path-level custody records, user approval forms, task-completion
instructions, and internal recovery-management records.

The builder scans every selected payload regardless of filename suffix, so
embedded notebook, HTML, PDF, archive, and binary metadata are subject to the
same checks. It fails on Windows user paths, the local workspace path, common
GitHub and OpenAI token patterns, bearer credentials, AWS access keys, private
key headers, files above 100 MiB, or unexpected binary model/image payloads.
The resulting ZIP contains its own file manifest; the external release receipt
records the ZIP hash, Git commit, tag, and public URL.

Before reading any payload, the builder requires a clean Git worktree and an
exact match between local `HEAD` and its pushed upstream commit. This binds the
manifest's source-study commit to the precise bytes selected for publication
instead of allowing uncommitted working-copy content to inherit an unrelated
commit identifier.

No third-party source-data license is granted. The repository may be made public
under the user's upload instruction after this audit passes, but an open-source
license will not be invented. Absence of a code license is disclosed unless the
authors provide one.

The publisher also queries the development repository immediately before and
after publication and refuses to proceed unless its visibility remains
`PRIVATE`. The fresh-download publication receipt records that visibility, and
the integrated completion audit requires it together with the new clean
repository's `PUBLIC` visibility.

Immediately before any remote write, the publisher also rechecks the clean,
pushed source state, all three submission-package hashes, and the three final
PDF hashes against the passed page-by-page visual-QA receipt. A PDF or bundle
changed after review therefore cannot be uploaded under the earlier approval.
