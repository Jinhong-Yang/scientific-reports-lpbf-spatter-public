# REV02 implementation boundaries

This is a pre-code coordination contract, not an amendment to the statistical protocol.

The new scripts live in `revision_rev02_20260902/scripts`. Historical source modules may be imported for pure calculations, but historical broad entry points must not run. No code opens a combined manifest except the one-time `prepare_rev02.py` data steward. Every modelling command receives a revision root and explicit train/validation split. The common loader is the only route to the sealed test manifest and requires a global T7 unlock receipt.

## Shared Python module (root-owned)

`rev02_common.py` exports:

- `REV_ROOT`, `PROJECT_ROOT`, `ARTIFACT_ROOT` as pathlib Paths, loaded from the master protocol.
- `utc_now()`, `sha256_file(path)`, `load_json(path)`, `write_json(path, value, overwrite=False)`, `read_csv(path)`, `write_csv(path, rows, fieldnames=None, overwrite=False)`; atomic writes, JSON finite/null only, explicit overwrite for live status.
- `load_protocol(task)` loads `protocol/REV02_<task>.yaml` after verifying the protocol hash receipt.
- `task_dir(task)` creates/returns local `results/REV02/<task>` with `raw` and `figures`; initial config is a copy of the frozen task YAML.
- `manifest_path(split)` returns the public train/validation CSV, or a gated test path. `load_rows(split)` uses it and validates every row's split, source and paths. No split has a default value.
- `resolve_path(value)` resolves an absolute path unchanged or a project-relative source path under `PROJECT_ROOT`.
- `record_event(event, payload)` appends an actual UTC event with a previous-event hash to the single serialized coordinator ledger; no concurrent GPU commands are allowed. `access_chronology.json` is a readable projection.
- `generator_run_dir(arm, seed)` returns `ARTIFACT_ROOT/runs/generators/<arm>_s<seed>`; generator best weights are `best.pt`, full resume state `resume.pt`, and summary `summary.json`.
- `detector_run_dir(run_id)` returns `ARTIFACT_ROOT/runs/detectors/<run_id>`.
- `pool_dir(arm)` returns `ARTIFACT_ROOT/pools/<arm>`, with `manifest.csv`, images and labels. Pool IDs are `G1`, `G2`, `D5`.
- `run_provenance(task, config, inputs)` returns frozen protocol/code/environment/input hashes; `inputs` maps human-readable names to paths.
- `write_task_summary(task, summary)` writes current explicit status and logs with `overwrite=True`; completed receipts are separate immutable files.
- `seed_everything(seed)` seeds Python, NumPy, torch/CUDA with the frozen deterministic settings.
- `require_test_unlocked()` raises unless a verified T7 campaign receipt exists; only the root coordinator can write that receipt after validating every pretest receipt.

Modellers may use `SpatterPatchDataset(PROJECT_ROOT, manifest_path('train'), ...)` or equivalent with safe separate CSVs. Do not call `dataset_splits` or `evaluate_checkpoint` default-test paths. Public row paths may be absolute; every record has explicit `source_kind=real|synthetic`. This is the canonical provenance-field name in all manifests.

## Component ownership and command contracts

Root owns preparation, common helpers, integrity gates, the serial managed coordinator, environment/source freeze and global completion audit. Agents own separate files/tests:

1. `rev02_generator.py`: `train --arm ARM --seed N`; `evaluate --arm ARM --seed N --split validation|test`; `make-plans`; `make-pool --arm G1|G2|D5`; `controls --split validation|test`; `summarize`. Sensitivity arms are `F11_kappa_low`, `F11_kappa_high`, `F11_source_low`, `F11_source_high`, `F11_decay_low`, `F11_decay_high`. Importable `main(argv=None)` supports in-process coordinator calls.
2. `rev02_detector.py`: `ratio --arm D2|D4 --ratio R --seed N`; `select-ratios`; `budget --family fasterrcnn|retinanet --arm D0|D1|D2|D4|D5 --axis B1B3|B2|B4 --ratio R --seed N`; `evaluate --run-id ID --split test`; `summarize`. The budget run ID is a deterministic function exported as `budget_run_id(family,arm,axis,ratio,seed)`. The evaluator processes all frozen milestones from that run. `main(argv=None)` is importable.
3. `rev02_memorization.py`: `calibrate`; `controls`; `pools`; `summarize`, all train/validation only. A calibration lock must exist before controls/pools and is immutable.
4. `rev02_statistics.py`: pure inference functions and `summarize --split validation|all`; `all` requires T7 and complete test records. Do not import historical reporting entry points.

CLI entry points always return nonzero/raise on incomplete integrity or missing required artifacts. They skip only a completed, hash-verified identical run; log resumptions/failures, never silently retrain. Training commands must leave the GPU free on return. The coordinator can spawn one command per process to isolate CUDA state.

Common protocol values, rather than duplicated magic constants, govern all model choices. If an interface needs a technical adjustment, coordinate first; do not change the frozen scientific choices. Synthetic-unit tests may use temporary toy images and do not access any real test paths. Performance smoke tests use train data only, a distinct smoke seed and disposable smoke outputs, never a completed planned seed.
