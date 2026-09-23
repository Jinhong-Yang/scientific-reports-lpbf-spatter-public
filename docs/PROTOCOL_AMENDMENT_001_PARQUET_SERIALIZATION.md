# Protocol amendment 001 — Parquet serialization runtime

Date: 2026-09-14  
Classification: non-scientific technical recovery  
Affected stage: derived analysis-file serialization beginning with W09

## Trigger

The first W09 analysis attempt completed all source-row validation but stopped
before writing Parquet because the frozen core training environment did not
contain a Parquet engine. The failure did not access held-out test data and did
not alter any completed generator fit, checkpoint, metric, or roster.

## Change

Add a separately vendored, pinned `pyarrow==21.0.0` serialization runtime under
the ignored local `vendor/parquet/` directory. Analysis programs prepend only
that directory when reading or writing Parquet. The training environment and
its frozen package set are not modified. The pinned requirement is recorded in
`configs/analysis_serialization_requirements.txt` and will be installed by the
clean reproduction procedure.

## Scientific invariants

- No input, split, label, arm, seed, loss, checkpoint rule, endpoint,
  comparison, or statistical method changes.
- W09 remains 40/40 completed under its original source and environment hashes.
- CSV and Parquet are parallel serializations of the same derived rows.
- Test access remains prohibited.
- The failed attempt and retry remain in the execution ledger.

This amendment resolves a file-format dependency only and does not authorize a
scientific protocol change.
