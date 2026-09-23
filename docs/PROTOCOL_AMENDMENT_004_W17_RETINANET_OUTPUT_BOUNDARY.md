# Protocol amendment 004 — W17 RetinaNet output boundary recovery

Date: 2026-09-23
Classification: post-unlock technical recovery; no outcome-based selection

## Trigger and preserved evidence

During the single named W17 campaign, the fixed W12 secondary RetinaNet cell
`secondary_retinanet_N1_s62002` stopped at the frozen COCO serialization
boundary with `Invalid COCO detection`.  The failure occurred after authorized
test access, is preserved in its original run directory, and is not removed or
overwritten.  No pooled metric, specimen-level metric, performance comparison,
or statistical output was consulted to define this recovery.

The error is the same non-finite/non-positive prediction-geometry condition
already handled before test access by Amendment 003.  It is an inability to
represent a model output as a legal COCO detection, not a model-selection or
scientific-outcome signal.

## Recovery

The frozen W17 runner, common detector adapter, configuration, checkpoints,
test roster, split, endpoints, and statistical plan remain byte-for-byte
unchanged.  Before each resumed session, the recovery wrapper verifies every
source hash sealed in `evidence/pretest/PRETEST_FREEZE.json`.

Only W12 secondary RetinaNet cells receive the already-defined Amendment 003
prediction boundary adapter.  It:

1. retains every finite canonical-foreground detection with `x2 > x1` and
   `y2 > y1` unchanged;
2. discards only outputs that cannot be represented as a legal COCO detection;
3. records candidate and discarded counts inside the ordinary prediction
   artifact; and
4. leaves every other W17 cell on the frozen unwrapped prediction path.

The resumed transaction retains the original checkpoint, source block, run
identifier, test manifest, and frozen W17 source hash.  It does not replace a
seed, alter a threshold, clip/relabel/reorder a valid detection, overwrite a
completed result, add a test analysis, or use a test result to choose any
scientific option.

## Interpretation and disclosure

This amendment is a technical output-serialization recovery.  It must remain
visible in the W17 failure accounting, reproducibility archive, Supplementary
Information, and final internal author-side audit.  It does not support an
external-validity claim or a claim that invalid model output has physical
meaning.
