"""Resume the frozen W17 campaign with the predeclared RetinaNet boundary adapter.

This wrapper deliberately leaves ``run_locked_test_campaign.py`` untouched.
It applies the existing W12 Amendment 003 sanitization only to the frozen W12
secondary RetinaNet cells, after proving that the underlying W17 source hashes
still match the pretest freeze.  It is an output-serialization recovery, not a
model, threshold, seed, checkpoint, or outcome-selection change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "detectors"))

import run_locked_test_campaign as w17  # noqa: E402
from run_w12_retinanet_recovery import RECOVERY_ID, sanitize_records  # noqa: E402


RECOVERY_ID_W17 = "PROTOCOL_AMENDMENT_004_W17_RETINANET_OUTPUT_BOUNDARY"
ORIGINAL_PREDICT = w17.predict
ORIGINAL_RUN_ONE = w17.run_one
_ACTIVE_SPEC: dict[str, Any] | None = None


def _is_recovery_cell(spec: dict[str, Any] | None, family: str) -> bool:
    """Restrict the adapter to the documented W12 secondary RetinaNet cells."""
    return bool(
        spec
        and family == "retinanet"
        and spec.get("source_block") == "W12_core"
        and str(spec.get("run_id", "")).startswith("secondary_retinanet_")
    )


def recovered_predict(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Pass all non-amendment cells through the frozen W17 prediction path."""
    predictions = ORIGINAL_PREDICT(*args, **kwargs)
    dataset = args[1] if len(args) > 1 else kwargs["dataset"]
    if not _is_recovery_cell(_ACTIVE_SPEC, dataset.family):
        return predictions
    return sanitize_records(predictions)


def recovered_run_one(spec: dict[str, Any], rows: list[dict[str, str]], test_manifest_sha256: str) -> dict[str, Any]:
    """Keep the original frozen run identity while annotating the active cell."""
    global _ACTIVE_SPEC
    _ACTIVE_SPEC = spec
    try:
        return ORIGINAL_RUN_ONE(spec, rows, test_manifest_sha256)
    finally:
        _ACTIVE_SPEC = None


def verify_recovery_binding() -> None:
    """Refuse recovery unless every frozen W17 source still binds identically."""
    freeze = w17.json.loads((w17.PRETEST_ROOT / "PRETEST_FREEZE.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "PRETEST_COMPLETE":
        raise RuntimeError("W17 pretest freeze is not complete")
    for relative, expected in freeze["source_hashes"].items():
        actual = w17.sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen W17 source differs: {relative}")
    if RECOVERY_ID != "PROTOCOL_AMENDMENT_003_W12_RETINANET_INVALID_PREDICTIONS":
        raise RuntimeError("Unexpected W12 recovery binding")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--session-hours", type=float, default=10.5)
    args = parser.parse_args()
    if not args.run_all:
        parser.error("--run-all is required")
    verify_recovery_binding()
    w17.predict = recovered_predict
    w17.run_one = recovered_run_one
    result = w17.run_all(args.session_hours)
    print(w17.json.dumps({"recovery_id": RECOVERY_ID_W17, "W17": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
