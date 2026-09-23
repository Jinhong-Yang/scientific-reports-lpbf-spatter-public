# Original implementation map

Status: W02 recovery map. This map identifies what can be reproduced exactly,
what is only historical evidence, and what must be implemented as N2 work.

## Recovered exact Rev02 implementation

The nested Git repository at
`historical/metal_spatter_pinn/revision_rev02_20260902` resolves commit
`7b20c45788b2020142cd2de7ed47b866253f851d`. A Git archive of that commit is
preserved as `inputs/rev02_implementation_7b20c457.zip` and expanded under
`src/reproduction/rev02_implementation_7b20c457`.

| Historical component | Recovered implementation | Verification |
|---|---|---|
| Data preparation | `scripts/prepare_rev02.py` | SHA-256 matches `IMPLEMENTATION_LOCK.json` |
| Shared path/config utilities | `scripts/rev02_common.py` | SHA-256 match |
| Generator training | `scripts/rev02_generator.py` | SHA-256 match |
| Detector training/evaluation | `scripts/rev02_detector.py` | SHA-256 match |
| Memorization audit | `scripts/rev02_memorization.py` | SHA-256 match |
| Statistical aggregation | `scripts/rev02_statistics.py` | SHA-256 match |
| Pipeline orchestration | `scripts/run_rev02_pipeline.py` | SHA-256 match |
| Historical smoke test | `scripts/smoke_rev02.py` | SHA-256 match |

The base package named `metal_spatter_pinn` is present in the copied historical
root. Its locked module hashes are checked separately because those files live
outside the nested Rev02 Git repository.

## Historical artifacts retained without reinterpretation

- `historical/LPBF_REV02_20260902/runs`: generator and detector checkpoints,
  configs, metrics, and per-run identity receipts.
- `historical/LPBF_REV02_20260902/evaluation`: validation/test evaluation
  artifacts.
- `historical/LPBF_REV02_20260902/pools`: synthetic-data pools.
- Raw detector outputs include 420 compressed prediction files and 420 matched
  specimen-level CSV files. These enable recomputation checks, but they remain
  H0 until reproduced.

## Boundaries of exact reproduction

- The historical environment is available and CUDA-capable on this host.
- Exact source recovery does not imply numerical identity across devices or
  library builds; the historical lock explicitly limits determinism to a
  seeded single-host setting.
- Existing working trees were dirty at discovery. The commit archive, source
  hashes, and copied snapshots are therefore the authoritative recovery
  evidence, not the mutable historical checkout state alone.
- The later Applied Sciences source package is missing. The locally available
  IEEE Access source/PDF/SI are catalogued only as older manuscript evidence.

## N2 components not present in the original implementation

Human-label calibration, corrected-box experiments, the proposed compact
diffusion baseline, new matched exposure controls, external-cohort evaluation,
and Scientific Reports packaging require new implementations after the
corresponding gates. None is counted as completed historical work.
