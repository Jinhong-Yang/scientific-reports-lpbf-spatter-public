"""Create a hash-bound, full-trajectory integrity receipt for W12.

The W12 core campaign is pre-registered as exactly 100 trajectories.  This
utility is deliberately standalone: it reads each completion receipt, hashes
each receipt-listed artifact relative to the managed project root, and writes
one machine-readable final audit receipt without changing any study artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
W12_ROOT = ROOT / "runs" / "new_study" / "W12_detector_core"
CORE_RECEIPT = ROOT / "evidence" / "detector_core" / "CORE_COMPLETENESS.json"
OUTPUT = ROOT / "evidence" / "detector_core" / "W12_FULL_INTEGRITY_AUDIT.json"
EXPECTED_TRAJECTORIES = 100
EXPECTED_UPDATES = 8960


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    trajectory_reports: list[dict[str, Any]] = []
    for receipt_path in sorted(W12_ROOT.rglob("completion.json")):
        receipt = read_json(receipt_path)
        artifact_hashes = receipt.get("artifact_hashes", {})
        mismatches: list[dict[str, str]] = []
        for relative_path, expected_hash in sorted(artifact_hashes.items()):
            artifact_path = ROOT / relative_path
            if not artifact_path.is_file():
                mismatches.append({"path": relative_path, "reason": "MISSING"})
                continue
            actual_hash = sha256(artifact_path)
            if actual_hash.lower() != str(expected_hash).lower():
                mismatches.append({"path": relative_path, "reason": "SHA256_MISMATCH"})
        trajectory_reports.append(
            {
                "run_id": receipt_path.parent.name,
                "status": receipt.get("status"),
                "optimizer_updates": receipt.get("optimizer_updates"),
                "test_payload_accessed": receipt.get("test_payload_accessed"),
                "artifact_hash_count": len(artifact_hashes),
                "artifact_mismatches": mismatches,
            }
        )

    core = read_json(CORE_RECEIPT)
    bad_trajectories = [
        report
        for report in trajectory_reports
        if report["status"] != "COMPLETED"
        or report["optimizer_updates"] != EXPECTED_UPDATES
        or report["test_payload_accessed"] is not False
        or not report["artifact_hash_count"]
        or report["artifact_mismatches"]
    ]
    pass_conditions = {
        "exactly_100_receipts": len(trajectory_reports) == EXPECTED_TRAJECTORIES,
        "every_trajectory_completed": not bad_trajectories,
        "core_receipt_complete": core.get("status") == "COMPLETE",
        "core_receipt_exact_count": core.get("planned_trajectories") == EXPECTED_TRAJECTORIES
        and core.get("completed_trajectories") == EXPECTED_TRAJECTORIES,
        "core_receipt_no_missing": core.get("missing_run_ids") == [],
        "core_receipt_no_test_access": core.get("test_payload_accessed") is False,
    }
    return {
        "schema_version": 1,
        "audit_name": "W12_FULL_INTEGRITY_AUDIT",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "all W12 completion receipts and every receipt-listed artifact hash",
        "status": "PASS" if all(pass_conditions.values()) else "FAIL",
        "conditions": pass_conditions,
        "summary": {
            "trajectory_receipts": len(trajectory_reports),
            "expected_trajectories": EXPECTED_TRAJECTORIES,
            "expected_optimizer_updates": EXPECTED_UPDATES,
            "trajectories_with_problems": len(bad_trajectories),
            "artifact_hashes_checked": sum(report["artifact_hash_count"] for report in trajectory_reports),
        },
        "core_receipt": {
            "status": core.get("status"),
            "planned_trajectories": core.get("planned_trajectories"),
            "completed_trajectories": core.get("completed_trajectories"),
            "missing_run_ids": core.get("missing_run_ids"),
            "test_payload_accessed": core.get("test_payload_accessed"),
        },
        "problems": bad_trajectories,
    }


def main() -> int:
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    print(f"W12_FULL_INTEGRITY_AUDIT={result['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
